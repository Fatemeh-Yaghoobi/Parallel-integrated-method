import jax
import jax.numpy as jnp
import numpy as np
from matplotlib import pyplot as plt

from integrated._utils import none_or_concat

jax.config.update('jax_platform_name', 'cpu')
jax.config.update("jax_enable_x64", True)

from integrated._base import MVNStandard
from integrated.inegrated_params import slow_rate_params, fast_rate_params, full_transition_params
from integrated.sequential import seq_filtering_slow_rate, seq_smoothing_slow_rate, filtering_fast_rate, seq_smoothing_full
from integrated.parallel import par_filtering_slow_rate, par_smoothing_slow_rate
from tests.linear.model import DistillationSSM

from integrated.batch import *

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

################################### Parameters ########################################
transition_model = model.TranParams()
observation_model = model.ObsParams()
Params_SR = slow_rate_params(transition_model, observation_model, l)
Params_FR = fast_rate_params(transition_model, observation_model, l)
Params_full = full_transition_params(transition_model, l)

Ahat, Bhat_u, Qhat = Params_full
A, B, u, Q1 = transition_model
# np.set_printoptions(precision=3)
# print(f"{Bhat_u = }")
# print(f"{A @ B + B = }")

### Filtering - Slow rate - Sequential and Parallel ###
sequential_filtered = seq_filtering_slow_rate(y, prior_x, Params_SR)
parallel_filtered = par_filtering_slow_rate(y, prior_x, Params_SR)
np.testing.assert_allclose(sequential_filtered.mean, parallel_filtered.mean, rtol=1e-06, atol=1e-03)

## Fast-rate Filtering ###
m_l_k_1 = sequential_filtered.mean[:-1]
P_l_k_1 = sequential_filtered.cov[:-1]
m_l_k = sequential_filtered.mean[1:]
P_l_k = sequential_filtered.cov[1:]
vmap_func = jax.vmap(filtering_fast_rate, in_axes=(0, None, 0, 0))
fast_rate_result_filtered_, new_fast = vmap_func(y, Params_FR, MVNStandard(m_l_k_1, P_l_k_1), MVNStandard(m_l_k, P_l_k))
fast_rate_result_filtered = none_or_concat(MVNStandard(fast_rate_result_filtered_.mean.reshape(-1, 4), fast_rate_result_filtered_.cov.reshape(-1, 4, 4)),
                                           MVNStandard(sequential_filtered.mean[-1], sequential_filtered.cov[-1]), 0)

new_mean = fast_rate_result_filtered.mean[1:].reshape(N, l*nx)
new_cov = jnp.zeros((N*l*nx, N*l*nx))
old_cov = fast_rate_result_filtered.cov[1:]
for i in range(l*N):
    new_cov = new_cov.at[i*nx: (i + 1)*nx, i*nx: (i + 1)*nx].set(old_cov[i])
reshaped_new_cov = jnp.zeros((N, l*nx, l*nx))
for i in range(N):
    reshaped_new_cov = reshaped_new_cov.at[i, :, :].set(new_cov[i*l*nx: (i + 1)*l*nx, i*l*nx: (i + 1)*l*nx])
batch_filtered_results = MVNStandard(new_mean, reshaped_new_cov)
### smoothing - full ###
sequential_smoothed_full = seq_smoothing_full(batch_filtered_results, Params_full)

# np.set_printoptions(precision=3)
fast_fms, fast_fPs = batch_fast_filter(model, y)
fast_sms, fast_sPs = batch_fast_smoother(model, y)
print(f"{fast_sms = }")
print(f"{sequential_smoothed_full.mean = }")



# P_k_1_l = reshaped_new_cov[0, :, :]  # 8 x 8
# M_k_1_l = new_mean[0, :]  # 8
# print(f"{M_k_1_l.shape = }")
# print(f"{fast_fms[3:5, :].reshape(-1,) = }")
# G_star = P_k_1_l @ Ahat.T @ jnp.linalg.inv(Ahat @ P_k_1_l @ Ahat.T + Qhat)
# M_bar = M_k_1_l + G_star @ (fast_fms[3:5, :].reshape(-1,) - Ahat @ M_k_1_l - Bhat_u)
# print(f"{M_bar = }")
# print(sequential_smoothed_full.mean)