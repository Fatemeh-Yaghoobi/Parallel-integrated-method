import jax
import jax.numpy as jnp
import numpy as np
from matplotlib import pyplot as plt

from integrated._utils import none_or_concat

jax.config.update('jax_platform_name', 'cpu')

jax.config.update("jax_enable_x64", True)

from integrated._base import MVNStandard
from integrated.inegrated_params import slow_rate_params, fast_rate_params
from integrated.sequential import seq_filtering_slow_rate, seq_smoothing_slow_rate, filtering_fast_rate
from integrated.parallel import par_filtering_slow_rate, par_smoothing_slow_rate
from tests.linear.model import DistillationSSM


# We form a large prior system like this first:
#
#               x_0 ~ N(m_0,P_0)
#   x_{k+1} - A x_k ~ N(B u, Q)
#   =>
#   Psi X = W,  where W ~ N([m0;B u;...;B u], blkdiag(P0,Q,...,Q))
#
# with X = [x_0;x_1;...;x_T], which defines the prior
#
#   X ~ N(Psi^{-1} m_W, Psi^{-1} P_W Psi^{-T})
#
# Then the observation model has the form
#
#   Y = Eta X + R,  R ~ N([0;...;0], blkdiag(R,...,R))
#
# where
#
#   Eta = [0 C ... C 0 ... 0
#          0 0 ... 0 C ... C 0 ... 0
#          0 0 ... 0 0 ... 0 C ... C 0 ... 0 ...
#          ...
#                                            ... C ... C] / ell
def create_batch_model(model):
    _t_model = model.TranParams()
    _o_model = model.ObsParams()

    _l = model.l
    _nx = model.nx
    _ny = model.ny
    _interval = model.interval
    _m0 = model.prior_x.mean
    _p0 = model.prior_x.cov
    _A = _t_model.A
    _Q = _t_model.Q
    _B = _t_model.B
    _u = _t_model.u
    _C = _o_model.C
    _R = _o_model.R

    batch_nx = _nx * (_l * _interval + 1)
    batch_ny = _ny * _interval
    MW = np.zeros((batch_nx,), dtype=np.float64)
    Psi = np.zeros((batch_nx, batch_nx), dtype=np.float64)
    PW = np.zeros((batch_nx, batch_nx), dtype=np.float64)
    Eta = np.zeros((batch_ny, batch_nx), dtype=np.float64)
    Sigma = np.zeros((batch_ny, batch_ny), dtype=np.float64)

    # Create the prior model
    for fast_k in range(_l * _interval + 1):
        Psi[(fast_k * _nx): ((fast_k + 1) * _nx), (fast_k * _nx): ((fast_k + 1) * _nx)] = np.eye(_nx, dtype=np.float64)
        if fast_k > 0:
            Psi[(fast_k * _nx) : ((fast_k + 1) * _nx), ((fast_k - 1) * _nx) : (fast_k * _nx)] = -_A
            MW[(fast_k * _nx) : ((fast_k + 1) * _nx)] = (_B @ _u)[..., 0]
            PW[(fast_k * _nx): ((fast_k + 1) * _nx), (fast_k * _nx): ((fast_k + 1) * _nx)] = _Q
        else:
            MW[(fast_k * _nx) : ((fast_k + 1) * _nx)] = _m0
            PW[(fast_k * _nx): ((fast_k + 1) * _nx), (fast_k * _nx): ((fast_k + 1) * _nx)] = _p0

    # Create the observation model
    for slow_k in range(_interval):
        for fast_rel_k in range(_l):
            fast_k = slow_k * _l + fast_rel_k + 1
            Eta[(slow_k * _ny) : ((slow_k + 1) * _ny), (fast_k * _nx) : ((fast_k + 1) * _nx)] = _C / _l
            Sigma[(slow_k * _ny) : ((slow_k + 1) * _ny), (slow_k * _ny) : ((slow_k + 1) * _ny)] = _R

    return MW, PW, Psi, Eta, Sigma



def batch_full_smoother(MW, PW, Psi, Eta, Sigma, Y):
    iPsi = np.linalg.inv(Psi)  # Bit fishy, but fine for reference solution
    MX0 = iPsi @ MW
    PX0 = iPsi @ PW @ iPsi.T

    # Compute posterior mean and covariance
    S = Eta @ PX0 @ Eta.T + Sigma
    iS = np.linalg.inv(S)  # Again fishy, but fine for reference
    K = PX0 @ Eta.T @ iS
    MX = MX0 + K @ (Y - Eta @ MX0)
    PX = PX0 - K @ S @ K.T

    return MX, PX

def extract_sms_sPs(MX, PX, nx):
    fast_sms = np.reshape(MX, (np.shape(MX)[0] // nx, nx))
    fast_sPs = np.zeros((np.shape(fast_sms)[0], np.shape(fast_sms)[1], np.shape(fast_sms)[1]),
                       dtype=np.float64)
    for k in range(np.shape(fast_sms)[0]):
        fast_sPs[k, :, :] = PX[(k * nx): ((k + 1) * nx), (k * nx): ((k + 1) * nx)]

    return fast_sms, fast_sPs


# Compute fast rate smoother result for the model given the measurements y
def batch_fast_smoother(model, y):
    MW, PW, Psi, Eta, Sigma = create_batch_model(model)
    Y = y.reshape(-1)
    MX, PX = batch_full_smoother(MW, PW, Psi, Eta, Sigma, Y)
    fast_sms, fast_sPs = extract_sms_sPs(MX, PX, model.nx)
    return fast_sms, fast_sPs

# Compute slow rate smoother result for the model given the measurements y
def batch_slow_smoother(model, y):
    fast_sms, fast_sPs = batch_fast_smoother(model, y)
    slow_sms = fast_sms[1::model.l]  # Take x_{k,1}
    slow_sPs = fast_sPs[1::model.l]
    return slow_sms, slow_sPs

# Compute fast rate filter result for the model given the measurements y
# This is a very inefficient implementation but hopefully works
def batch_fast_filter(model, y):
    MW, PW, Psi, Eta, Sigma = create_batch_model(model)

    fast_fms = np.zeros((model.interval * model.l + 1, model.nx),
                        dtype=np.float64)
    fast_fPs = np.zeros((model.interval * model.l + 1, model.nx, model.nx),
                        dtype=np.float64)
    fast_fms[0] = model.prior_x.mean
    fast_fPs[0] = model.prior_x.cov
    Y = y.reshape(-1)

    for yi in range(model.interval):
        xi1 = 1 + yi * model.l    # Start index of the interval
        xi2 = (yi + 1) * model.l  # End index of the interval

        fMW = MW[0 : ((xi2 + 1) * model.nx)]
        fPW = PW[0 : ((xi2 + 1) * model.nx), 0 : ((xi2 + 1) * model.nx)]
        fPsi = Psi[0 : ((xi2 + 1) * model.nx), 0 : ((xi2 + 1) * model.nx)]
        fEta = Eta[0 : ((yi + 1) * model.ny), 0 : ((xi2 + 1) * model.nx)]
        fY = Y[0 : ((yi + 1) * model.ny)]
        fSigma = Sigma[0 : ((yi + 1) * model.ny), 0 : ((yi + 1) * model.ny)]

        fMX, fPX = batch_full_smoother(fMW, fPW, fPsi, fEta, fSigma, fY)

        for xi in range(xi1, xi2+1):
            fast_fms[xi] = fMX[(xi * model.nx) : ((xi + 1) * model.nx)]
            fast_fPs[xi] = fPX[(xi * model.nx) : ((xi + 1) * model.nx), (xi * model.nx) : ((xi + 1) * model.nx)]

    return fast_fms, fast_fPs


# Compute joint fast rate filter result for the model given the measurements y
# This is a very inefficient implementation but hopefully works. The initial
# distribution is left out as it has a different size.
def batch_joint_fast_filter(model, y):
    MW, PW, Psi, Eta, Sigma = create_batch_model(model)

    fast_fms = np.zeros((model.interval, model.nx * model.l),
                        dtype=np.float64)
    fast_fPs = np.zeros((model.interval, model.nx * model.l, model.nx * model.l),
                        dtype=np.float64)
    Y = y.reshape(-1)

    for yi in range(model.interval):
        xi1 = 1 + yi * model.l    # Start index of the interval
        xi2 = (yi + 1) * model.l  # End index of the interval

        fMW = MW[0 : ((xi2 + 1) * model.nx)]
        fPW = PW[0 : ((xi2 + 1) * model.nx), 0 : ((xi2 + 1) * model.nx)]
        fPsi = Psi[0 : ((xi2 + 1) * model.nx), 0 : ((xi2 + 1) * model.nx)]
        fEta = Eta[0 : ((yi + 1) * model.ny), 0 : ((xi2 + 1) * model.nx)]
        fY = Y[0 : ((yi + 1) * model.ny)]
        fSigma = Sigma[0 : ((yi + 1) * model.ny), 0 : ((yi + 1) * model.ny)]

        fMX, fPX = batch_full_smoother(fMW, fPW, fPsi, fEta, fSigma, fY)

        fast_fms[yi] = fMX[xi1 * model.nx : (xi2 + 1) * model.nx]
        fast_fPs[yi] = fPX[xi1 * model.nx : (xi2 + 1) * model.nx, xi1 * model.nx : (xi2+1) * model.nx]

    return fast_fms, fast_fPs


# Compute fast rate filter result for the model given the measurements y
# This is a very inefficient implementation but hopefully works
def batch_slow_filter(model, y):
    fast_fms, fast_fPs = batch_fast_filter(model, y)
    slow_fms = fast_fms[0::model.l]
    slow_fPs = fast_fPs[0::model.l]
    return slow_fms, slow_fPs
