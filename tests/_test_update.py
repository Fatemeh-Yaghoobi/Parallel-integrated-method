import jax
import jax.numpy as jnp
import numpy as np
from matplotlib import pyplot as plt

from integrated._base import MVNStandard

jax.config.update("jax_enable_x64", True)

from integrated.inegrated_params import new_params
from tests.linear.model import DistillationSSM
# from integrated.sequential._filtering_new import _integrated_update

################################### Parameters ########################################
l = 3
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
parameters_all = new_params(transition_model, observation_model, l)
#########################################################################################
A, B, u, cov = transition_model
C, R = observation_model
Abar, Bbar, Gbar, Bbar_u, Qbar, H, R = parameters_all
#########################################################################################
m_ = jax.random.normal(jax.random.PRNGKey(0), shape=(l, nx))
P_ = jax.random.normal(jax.random.PRNGKey(1), shape=(l, nx, nx))
y = jax.random.normal(jax.random.PRNGKey(2), shape=(ny,))
Prior = MVNStandard(m_, P_)
def _integrated_update(all_params, x_predict_interval, y):
    m_, P_ = x_predict_interval
    Abar, Bbar, Gbar, Bbar_u, Qbar, H, R = all_params
    y_diff= y - jnp.einsum('ijk,ik->j', H, m_)
    S = H[0] @ jnp.sum(P_, axis=0) @ H[0].T + R
    temp = P_ @ H[0].T
    KT = jnp.linalg.solve(S, jnp.transpose(temp, axes=(0, 2, 1)))
    K = jnp.transpose(KT, axes=(0, 2, 1))
    print(K.shape)
    m = m_ + jnp.einsum('ijk,k->ij', K, y_diff)
    P = P_ - (K @ H) @ P_
    temp2 = jnp.stack([K[0] @ H[0] @ P_[0], K[1] @ H[1] @ P_[1], K[2] @ H[2] @ P_[2]])
    # print(P_.shape)
    np.testing.assert_allclose(P, P_ - jnp.stack([K[0] @ H[0] @ P_[0], K[1] @ H[1] @ P_[1], K[2] @ H[2] @ P_[2]]))

    return MVNStandard(m, P)


update_results = _integrated_update(parameters_all, Prior, y)
