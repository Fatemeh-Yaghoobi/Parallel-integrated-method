import jax
import jax.numpy as jnp
import numpy as np
from matplotlib import pyplot as plt

from integrated._base import MVNStandard

jax.config.update("jax_enable_x64", True)

from integrated.inegrated_params import new_params, all_params
from tests.linear.model import DistillationSSM
# from integrated.sequential._filtering_new import _integrated_update

################################### Parameters ########################################
l = 2
N = 10
nx = 4
ny = 2
Q = 0.1
R = 0.1
prior_x = MVNStandard(jnp.zeros(4), jnp.eye(4))
seed = 3
################################ Model: Distillation Column ##########################
model = DistillationSSM(l=l, interval=N, nx=nx, ny=ny, Q=Q, R=R, prior_x=prior_x, seed=seed)  # noqa
# x, h, y = model.get_data()
################################### Filtering ########################################
transition_model = model.TranParams()
observation_model = model.ObsParams()
parameters_NEW = new_params(transition_model, observation_model, l)
parameters_OLD = all_params(transition_model, observation_model, l)
#########################################################################################
A, B, u, cov_Q = transition_model
C, cov_R = observation_model
#########################################################################################
def _integrated_predict_NEW(all_params, x):
    m, P = x
    Abar, _, _, Bbar_u, Qbar, _, _ = all_params

    m = Abar @ m + Bbar_u                                      # dim = l x nx
    P = Abar @ P @ jnp.transpose(Abar, axes=(0, 2, 1)) + Qbar  # dim = l x nx x nx
    return MVNStandard(m, P)
def _integrated_update_NEW(all_params, x_predict_interval, y):
    m_, P_ = x_predict_interval
    Abar, Bbar, Gbar, Bbar_u, Qbar, H, R = all_params

    y_diff = y - jnp.einsum('ijk,ik->j', H, m_)

    S = H[0] @ jnp.sum(P_, axis=0) @ H[0].T + R
    temp = P_ @ H[0].T
    vmap_solve = jax.vmap(jnp.linalg.solve, in_axes=(None, 0))
    KT = vmap_solve(S, jnp.transpose(temp, axes=(0, 2, 1)))
    K = jnp.transpose(KT, axes=(0, 2, 1))
    m = m_ + jnp.einsum('ijk,k->ij', K, y_diff)
    P = P_ - K @ H @ P_

    return  H[0] @ jnp.sum(P_, axis=0) @ H[0].T, jnp.sum(P_, axis=0)
##################################
def _integrated_predict_OLD(all_params, x):
    m, P = x
    Abar, _, _, Bbar_u, Qbar, _, _, _, _, _, _, _ = all_params

    m = Abar @ m + Bbar_u                                      # dim = l x nx
    P = Abar @ P @ jnp.transpose(Abar, axes=(0, 2, 1)) + Qbar  # dim = l x nx x nx
    return MVNStandard(m, P)

def _integrated_update_OLD(all_params, x_predict_interval, xl_k_1, y):
    m_, P_ = x_predict_interval
    m_k_1, P_k_1 = xl_k_1
    Abar, Bbar, Gbar, Bbar_u, Qbar, C_bar, M_bar, D_bar, Rx, Q, Du_bar, r1 = all_params
    S = C_bar @ P_k_1 @ C_bar.T + Rx                                                      # dim = ny x ny
    temp = (Abar @ P_k_1 @ C_bar.T
            + jnp.einsum('ijkl,jlm->ikm',
                         Gbar @ Q, jnp.transpose(M_bar, axes=(0, 2, 1))))                 # dim = l x nx x ny


    vmap_func = jax.vmap(jax.scipy.linalg.solve, in_axes=(None, 0))
    TranL = vmap_func(S, jnp.transpose(temp, axes=(0, 2, 1)))
    L = jnp.transpose(TranL, axes=(0, 2, 1))                                                 # dim = l x nx x ny

    m = m_ + jnp.einsum('ijk,k->ij', L, y -  C_bar @ m_k_1 - Du_bar)                         # dim = l x nx
    tempT = jnp.transpose(temp, axes=(0, 2, 1))                                              # dim = l x ny x nx
    P = P_ - jnp.einsum('ijk,ikl->ijl', L, tempT)                                            # dim = l x nx x nx

    return C_bar @ P_k_1 @ C_bar.T + r1, r1
################################################### l=3
m_1 = jax.random.normal(jax.random.PRNGKey(0), shape=(nx,))
P_1 = jax.random.normal(jax.random.PRNGKey(1), shape=(nx, nx))
x0 = MVNStandard(m_1, P_1)
y = jax.random.normal(jax.random.PRNGKey(2), shape=(ny,))
prior_OLD = _integrated_predict_OLD(parameters_OLD, x0)
prior_NEW = _integrated_predict_NEW(parameters_NEW, x0)
np.testing.assert_allclose(prior_OLD.mean, prior_NEW.mean, rtol=1e-06, atol=0)
np.testing.assert_allclose(prior_OLD.cov, prior_NEW.cov, rtol=1e-06, atol=0)

old_update_res = _integrated_update_OLD(parameters_OLD, prior_NEW, x0, y)
new_update_res = _integrated_update_NEW(parameters_NEW, prior_OLD, y)
np.testing.assert_allclose(old_update_res[1], 1/l * (C) @ (A@cov_Q@A.T + cov_Q + cov_Q) @ (C).T * (1/l), rtol=1e-06, atol=0)