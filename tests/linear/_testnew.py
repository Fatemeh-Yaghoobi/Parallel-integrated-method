import jax
import jax.numpy as jnp
import numpy as np
from jax import lax, jit
jax.config.update("jax_enable_x64", True)
from integrated._base import MVNStandard
from integrated.inegrated_params import all_params, slow_rate_params
from integrated.sequential import integrated_filtering, all_filtering
from tests.linear.model import DistillationSSM
################################### Parameters ########################################
l = 4
N = 5
nx = 4
ny = 2
Q = 1
R = 1
prior_x = MVNStandard(jnp.zeros(4), jnp.eye(4))
seed = 0
################################ Model: Distillation Column ##########################
model = DistillationSSM(l=l, interval=N, nx=nx, ny=ny, Q=Q, R=R, prior_x=prior_x, seed=seed)  # noqa
x, h, y = model.get_data()
# print(x.shape, h.shape, y.shape)
################################### Filtering ########################################
transition_model = model.TranParams()
observation_model = model.ObsParams()
i = l - 2
slow_rate_params = slow_rate_params(transition_model, observation_model, i)
parameters_all = all_params(transition_model, observation_model, l)

def all_parameters_comparison():
    Abar, Bbar, Gbar, Bbar_u, Qbar, C_bar, M_bar, D_bar, Rx, Q, Du_bar = parameters_all
    return Abar, Bbar, Gbar, Bbar_u, Qbar, C_bar, M_bar, D_bar, Rx, Q, Du_bar
def slow_parameters_comparison():
    A_bar, G_bar, B_bar, u_bar, Bu_bar, Q_bar, C_bar, M_bar, D_bar, Rx, Q, Du_bar = slow_rate_params
    return A_bar, G_bar, B_bar, u_bar, Bu_bar, Q_bar, C_bar, M_bar, D_bar, Rx, Q, Du_bar

A_all, B_all, G_all, B_u_all, Qbar_all, C_all, M_all, D_all, Rx_all, Q_all, Du_all = all_parameters_comparison()
A_slow, G_slow, B_slow, u_slow, Bu_slow, Qbar_slow, C_slow, M_slow, D_slow, Rx_slow, Q_slow, Du_slow = slow_parameters_comparison()

np.testing.assert_allclose(A_all[i-l-1], A_slow, rtol=1e-06, atol=0)
np.testing.assert_allclose(B_all[i-l-1, :i-l, :, :], B_slow, rtol=1e-06, atol=0)
np.testing.assert_allclose(G_all[i-l-1, :i-l, :, :], G_slow, rtol=1e-06, atol=0)
np.testing.assert_allclose(B_u_all[i-l-1], Bu_slow, rtol=1e-06, atol=0)
np.testing.assert_allclose(Qbar_all[i-l-1], Qbar_slow, rtol=1e-06, atol=0)

sequential_filtered_i = integrated_filtering(y, prior_x, slow_rate_params)
sequential_filtered_all = all_filtering(y, prior_x, parameters_all)
print(sequential_filtered_i.mean.shape)
print(sequential_filtered_i.mean[1:, 0])
print(sequential_filtered_all.mean.shape)
print(sequential_filtered_all.mean.reshape(-1, 4)[:, 0])
