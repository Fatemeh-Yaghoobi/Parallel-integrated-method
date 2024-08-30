###
# Test all the parameters and methods for l = 2, N = 1
#

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update('jax_platform_name', 'cpu')
jax.config.update("jax_enable_x64", True)

from integrated._base import MVNStandard
from integrated.inegrated_params import slow_rate_params, fast_rate_params
from integrated.sequential import seq_filtering_slow_rate, seq_smoothing_slow_rate, filtering_fast_rate
from integrated.parallel import par_filtering_slow_rate, par_smoothing_slow_rate
from tests.linear.model import DistillationSSM

from integrated.batch import *
################################### Parameters ########################################
l = 2
N = 1
nx = 4
ny = 2
Q = 1
R = 1
prior_x = MVNStandard(jnp.zeros(4), jnp.eye(4))
seed = 0
################################ Model: Distillation Column ##########################
model = DistillationSSM(l=l, interval=N, nx=nx, ny=ny, Q=Q, R=R, prior_x=prior_x, seed=seed)  # noqa
x, h, y = model.get_data()

################################### Parameters ########################################
transition_model = model.TranParams()
observation_model = model.ObsParams()
Params_SR = slow_rate_params(transition_model, observation_model, l)
Params_FR = fast_rate_params(transition_model, observation_model, l)
A, B, u, Q1 = transition_model
C, R = observation_model
fAbar, fBbar, fGbar, fBbar_u, fQbar, C_bar, M_bar_minus_1, fD_bar, Rx, Q2, fDu_bar = Params_FR
sA_bar, sG_bar, sB_bar, u_bar, sBu_bar, sQ_bar, _, M_bar, _, _, Q, _ = Params_SR
### test parameters for l = 2 , N = 1
def _test_parameters():
    np.testing.assert_allclose(fAbar[0], A, rtol=1e-06, atol=1e-06)
    np.testing.assert_allclose(C_bar, C/l @ (A + A @ A), rtol=1e-06, atol=1e-06)
    np.testing.assert_allclose(fD_bar[0], C / l @ (A @ B + B), rtol=1e-06, atol=1e-06)
    np.testing.assert_allclose(fD_bar[1], C / l @ (B), rtol=1e-06, atol=1e-06)
    np.testing.assert_allclose(M_bar[0], C/l @ (A + jnp.eye(A.shape[0])), rtol=1e-06, atol=1e-06)
    np.testing.assert_allclose(M_bar[1], C/l , rtol=1e-06, atol=1e-06)
    np.testing.assert_allclose(M_bar[0], M_bar_minus_1[0], rtol=1e-06, atol=1e-06)
    np.testing.assert_allclose(Q, Q2, rtol=1e-06, atol=1e-06)
    np.testing.assert_allclose(fGbar[0][0], jnp.eye(A.shape[0]), rtol=1e-06, atol=1e-06)
    np.testing.assert_allclose(fQbar[0], fGbar[0][0] @ Q1 @ fGbar[0][0].T, rtol=1e-06, atol=1e-06)
    np.testing.assert_allclose(Rx, M_bar[0] @ Q1 @ M_bar[0].T + M_bar[1] @ Q1 @ M_bar[1].T + R, rtol=1e-06, atol=1e-06)
_test_parameters()
print("All parameters are correct")

###






