import jax
import jax.numpy as jnp
import numpy as np
from matplotlib import pyplot as plt

jax.config.update("jax_enable_x64", True)

from integrated._base import MVNStandard
from integrated.inegrated_params import all_params, slow_rate_params
from integrated.sequential import all_filtering, integrated_filtering
from integrated.parallel import parallel_filtering_fast_slow, parallel_filtering
from tests.linear.model import DistillationSSM

################################### Parameters ########################################
l = 2
N = 2
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
parameters_all = all_params(transition_model, observation_model, l)
parameters_slow = slow_rate_params(transition_model, observation_model, l)

Abar, Bbar, Gbar, Bbar_u, Qbar, C_bar, M_bar, D_bar, Rx, Q, Du_bar = parameters_all
A_bar, G_bar, B_bar, u_bar, Bu_bar, Q_bar, _, _, _, _, _, _ = parameters_slow


parallel_filtered = parallel_filtering_fast_slow(y, prior_x, parameters_all)
sequential_filtered = all_filtering(y, prior_x, parameters_all)
# parallel_filtered_slow = parallel_filtering(y, prior_x, parameters_slow)

seq_res_all = sequential_filtered.mean.reshape(-1, 4)
par_res_all = parallel_filtered.mean.reshape(-1, 4)
# print(seq_res_all.shape)
# print(seq_res_all[:, 0])
# print(par_res_all.shape)
# print(par_res_all[:, 0])
# # seq_filter_slow = integrated_filtering(y, prior_x, parameters_slow)
# # plt.plot(x[1:, 0], '--', color='b', label='true x')
plt.plot(seq_res_all[:, 0], '.--', color='r', label='sequential filtered x')
plt.plot(par_res_all[:, 0], '.--', color='g', label='parallel filtered x')
# plt.plot(range(l-1, len(x) - 1, l), parallel_filtered_slow.mean[1:, 0], '*--', color='y', label='parallel x - slow')
plt.grid()
plt.legend()
plt.show()


