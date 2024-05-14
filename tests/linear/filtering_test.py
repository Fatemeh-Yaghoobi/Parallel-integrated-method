import jax
import jax.numpy as jnp
import numpy as np
from matplotlib import pyplot as plt

jax.config.update("jax_enable_x64", True)

from integrated._base import MVNStandard
from integrated.inegrated_params import slow_rate_params, fast_rate_params
from integrated.sequential import integrated_filtering, integrated_smoothing
from integrated.parallel import parallel_filtering, parallel_smoothing
from tests.linear.model import DistillationSSM

################################### Parameters ########################################
l = 10
N = 50
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
Params_SR = slow_rate_params(transition_model, observation_model, l)
Params_FR = fast_rate_params(transition_model, l)


sequential_filtered = integrated_filtering(y, prior_x, Params_SR)
sequential_smoothed = integrated_smoothing(sequential_filtered, Params_SR)
#
# parallel_filtered = parallel_filtering(y, prior_x, Params_SR)
# parallel_smoothed = parallel_smoothing(parallel_filtered, Params_SR)
def fast_rate(fast_params, slow_result_k_1, slow_result_k):
    m_0, P_0 = slow_result_k_1
    m_l, P_l = slow_result_k
    Abar, Bbar, Gbar, Bbar_u, Qbar = fast_params
    m = Abar @ m_0 + Bbar_u
    P = Abar @ P_0 @ jnp.transpose(Abar, axes=(0, 2, 1)) + Qbar
    m = jnp.concatenate([m, m_l[None, :]], axis=0)
    P = jnp.concatenate([P, P_l[None, :, :]], axis=0)
    return MVNStandard(m, P)

fast_vmap = jax.vmap(fast_rate, in_axes=(None, 0, 0))
slow_result_k_1 = MVNStandard(sequential_filtered.mean[:-1, :], sequential_filtered.cov[:-1, :])
slow_result_k = MVNStandard(sequential_filtered.mean[1:, :], sequential_filtered.cov[1:, :])
fast_result = fast_vmap(Params_FR, slow_result_k_1, slow_result_k)
fast_rate_result_seq_filter = fast_result.mean.reshape(-1, nx)

smooth_slow_k_1 = MVNStandard(sequential_smoothed.mean[:-1, :], sequential_smoothed.cov[:-1, :])
smooth_slow_k = MVNStandard(sequential_smoothed.mean[1:, :], sequential_smoothed.cov[1:, :])
fast_result_smooth = fast_vmap(Params_FR, smooth_slow_k_1, smooth_slow_k)
fast_rate_result_seq_smooth = fast_result_smooth.mean.reshape(-1, nx)

plt.plot(fast_rate_result_seq_smooth[:, 0], label='fast rate x - smooth')
plt.plot(fast_rate_result_seq_filter[:, 0], label='fast rate x - filter')
plt.plot(range(l-1, len(x) - 1, l), sequential_filtered.mean[1:, 0], '--', color='y', label='filtered x')
# plt.plot(range(l, len(x), l), sequential_smoothed.mean[1:, 0], '--', color='g', label='smoothed x')
# # plt.plot(range(l, len(x), l), parallel_smoothed.mean[1:, 0], '--', color='r', label='parallel smoothed x')
plt.plot(range(l, len(x), l), x[1::l, 0], '--', color='b', label='true x')
# # plt.plot(range(l, len(x), l), parallel_filtered.mean[1:, 0], '*--', color='r', label='parallel filtered x')
plt.legend()
plt.grid()
plt.show()


