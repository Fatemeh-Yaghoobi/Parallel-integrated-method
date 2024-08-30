###
# Testing of the implementation of a batch solution to the problem
#


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

from integrated.batch import *

## %% (new cell starts here)

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

### Filtering - Slow rate - Sequential and Parallel ###
sequential_filtered = seq_filtering_slow_rate(y, prior_x, Params_SR)
parallel_filtered = par_filtering_slow_rate(y, prior_x, Params_SR)
np.testing.assert_allclose(sequential_filtered.mean, parallel_filtered.mean, rtol=1e-06, atol=1e-03)

### Fast-rate Filtering ###
m_l_k_1 = sequential_filtered.mean[:-1]
P_l_k_1 = sequential_filtered.cov[:-1]
vmap_func = jax.vmap(filtering_fast_rate, in_axes=(0, None, 0))
fast_rate_result_filtered_ = vmap_func(y, Params_FR, MVNStandard(m_l_k_1, P_l_k_1))
fast_rate_result_filtered = none_or_concat(MVNStandard(fast_rate_result_filtered_.mean.reshape(-1, 4), fast_rate_result_filtered_.cov.reshape(-1, 4, 4)),
                                           MVNStandard(sequential_filtered.mean[-1], sequential_filtered.cov[-1]), 0)

### Smoothing - Slow rate - Sequential and Parallel ###
sequential_smoothed = seq_smoothing_slow_rate(sequential_filtered, Params_SR)
parallel_smoothed = par_smoothing_slow_rate(parallel_filtered, Params_SR)
np.testing.assert_allclose(sequential_smoothed.mean, parallel_smoothed.mean, rtol=1e-06, atol=1e-03)



##%%

#fast_sms, fast_sPs = batch_fast_smoother(model, y)
#print(f"{fast_sms = }")

# slow_sms, slow_sPs = batch_slow_smoother(model, y)
# print(f"{slow_sms = }")
#
# print(f"{sequential_smoothed.mean = }")
#
fast_fms, fast_fPs = batch_fast_filter(model, y)
print(f"{fast_fms = }")
#
# print(f"{fast_rate_result_filtered.mean = }")
#
# slow_fms, slow_fPs = batch_slow_filter(model, y)
# print(f"{slow_fms = }")
#
# print(f"{sequential_filtered.mean = }")


### test for l = 2 , N = 1
A, B, u, Q1 = transition_model
C, R = observation_model
fAbar, fBbar, fGbar, fBbar_u, fQbar, C_bar, M_bar_minus_1, fD_bar, Rx, Q2, fDu_bar = Params_FR
sA_bar, sG_bar, sB_bar, u_bar, sBu_bar, sQ_bar, _, M_bar, _, _, Q, _ = Params_SR
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



m0, P0 = prior_x.mean, prior_x.cov
x_1_filtered_mean = (A @ m0 + B @ u[0]
                + (A @ P0 @ C_bar.T + Q @ M_bar[0].T) @ jnp.linalg.inv(C_bar @ P0 @ C_bar.T +  Rx) @ (y[0] - C_bar @ m0 - fDu_bar))
print(f"{x_1_filtered_mean = }")











# # fast rate filtering one step
# def _prediction_one_step(Params_fr, x0):
#     m0, P0 = x0
#     Abar, _, _, Bbar_u, Qbar, _, _, _, _, _, _ = Params_fr
#     m_1_1 = Abar[0] @ m0 + Bbar_u
#     P_1_1 = Abar[0] @ P0 @ Abar[0].T + Qbar[0]
#     return MVNStandard(m_1_1, P_1_1)
#
# from integrated.sequential._filtering_fast_rate import _integrated_predict as _integrated_predict_fast
#
# predicted_1 = _prediction_one_step(Params_FR, (fast_rate_result_filtered.mean[0], fast_rate_result_filtered.cov[0]))
# predicted_2 = _integrated_predict_fast(Params_FR, (fast_rate_result_filtered.mean[0], fast_rate_result_filtered.cov[0]))
#
# def _update_one_step(Params_fr, predicted_result, x0, y):
#     m0, P0 = x0
#     m_, P_ = predicted_result
#
#     Abar, Bbar, Gbar, Bbar_u, Qbar, C_bar, M_bar, D_bar, Rx, Q, Du_bar = Params_fr
#     # check Rx
#     S = C_bar @ P0 @ C_bar.T + Rx
#     temp = (Abar[0] @ P0 @ C_bar.T + Gbar[0][0] @ Q @ M_bar[0].T)
#     L = jnp.linalg.solve(S, temp.T).T
#     m_1_1 = m_[0] +  L @ (y - C_bar @ m0 - Du_bar)
#     P_1_1 = P0 - L @ S @ L.T
#     return MVNStandard(m_1_1, P_1_1)
#
# from integrated.sequential._filtering_fast_rate import _integrated_update as _integrated_update_fast
#
# updated_1 = _update_one_step(Params_FR, (predicted_1.mean, predicted_1.cov),
#                              (fast_rate_result_filtered.mean[0], fast_rate_result_filtered.cov[0]), y[0])
# updated_2 = _integrated_update_fast(Params_FR, (predicted_2.mean, predicted_2.cov),
#                                     (fast_rate_result_filtered.mean[0], fast_rate_result_filtered.cov[0]), y[0])
# print(f"{updated_1.mean = }")
# print(f"{updated_2.mean = }")