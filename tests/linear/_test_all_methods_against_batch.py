#
# Testing of the implementation of a batch solution to the problem
#
from ssl import PROTOCOL_SSLv23

import jax
import jax.numpy as jnp
import numpy as np
from fontTools.misc.psOperators import PSOperators
from matplotlib import pyplot as plt

from integrated._utils import none_or_concat, none_or_shift

jax.config.update('jax_platform_name', 'cpu')

jax.config.update("jax_enable_x64", True)

from integrated._base import MVNStandard
from integrated.inegrated_params import slow_rate_params, fast_rate_params
from integrated.sequential import seq_filtering_slow_rate, seq_smoothing_slow_rate, filtering_fast_rate
from integrated.parallel import par_filtering_slow_rate, par_smoothing_slow_rate
from tests.linear.model import DistillationSSM

from integrated.batch import *

#%% (new cell starts here)

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
def selected_fast_filtered(filtered_results, l):
    size = len(filtered_results.mean)
    def body(_, i):
        return _, MVNStandard(filtered_results.mean[i], filtered_results.cov[i])

    _, selected =  jax.lax.scan(body, init = None, xs=jnp.arange(1, size, l))
    return selected
sr_filtered_k1 = selected_fast_filtered(fast_rate_result_filtered, l)
# print(f"{sr_filtered_k1.mean = }")
### Smoothing - Slow rate - Sequential and Parallel ###
sequential_smoothed = seq_smoothing_slow_rate(sr_filtered_k1, Params_SR)
print(f"{sequential_smoothed.mean = }")
# parallel_smoothed = par_smoothing_slow_rate(parallel_filtered, Params_SR)
# np.testing.assert_allclose(sequential_smoothed.mean, parallel_smoothed.mean, rtol=1e-06, atol=1e-03)

fast_fms, fast_fPs = batch_fast_filter(model, y)
fast_sms, fast_sPs = batch_fast_smoother(model, y)
# print(f"{fast_sms = }")
##### smoothing test for l=2 and N=2, states are [x_0, x_1, x_2, x_3, x_4] ####
### states in the last interval are [x_3, x_4]  and they have the same filtered and smoothed values
ms4 = fast_sms[4]
mf4 = fast_fms[4]
ms3 = fast_sms[3]
mf3 = fast_fms[3]
np.testing.assert_allclose(ms4, mf4, rtol=1e-06, atol=1e-03)
np.testing.assert_allclose(ms3, mf3, rtol=1e-06, atol=1e-03)
Ps3 = fast_sPs[3]
### we want to find the smoothed values for x_2 and x_1
def x_smoothed(mf, Pf, ms, Ps):
    A, B, u, Q1 = transition_model
    m_ = A @ mf + (B @ u).reshape(-1)
    P_ = A @ Pf @ A.T + Q1
    G = Pf @ A.T @ jnp.linalg.inv(P_)
    ms = mf + G @ (ms - m_)
    Ps = Pf + G @ (Ps - P_) @ G.T
    return MVNStandard(ms, Ps)
### smoothed value for x_2 from x_3
mf2 = fast_fms[2]
Pf2 = fast_fPs[2]
ms2 = x_smoothed(mf2, Pf2, ms3, Ps3).mean
Ps2 = x_smoothed(mf2, Pf2, ms3, Ps3).cov
np.testing.assert_allclose(ms2, fast_sms[2], rtol=1e-06, atol=1e-03)
np.testing.assert_allclose(Ps2, fast_sPs[2], rtol=1e-06, atol=1e-03)
### smoothed value for x_1 from x_2
mf1 = fast_fms[1]
Pf1 = fast_fPs[1]
ms1 = x_smoothed(mf1, Pf1, ms2, Ps2).mean
Ps1 = x_smoothed(mf1, Pf1, ms2, Ps2).cov
np.testing.assert_allclose(ms1, fast_sms[1], rtol=1e-06, atol=1e-03)
np.testing.assert_allclose(Ps1, fast_sPs[1], rtol=1e-06, atol=1e-03)

#### smoothed values for x_1 and x_3
def x_smoothed_sr(mf, Pf, ms, Ps):
    A, B, u, Q1 = transition_model
    Abar = A @ A
    Qbar = A @ Q1 @ A.T + Q1
    m_ = Abar @ mf + (A @ B @ u).reshape(-1) + (B @ u).reshape(-1)
    P_ = Abar @ Pf @ Abar.T + Qbar
    G = Pf @ Abar.T @ jnp.linalg.inv(P_)
    ms = mf + G @ (ms - m_)
    Ps = Pf + G @ (Ps - P_) @ G.T
    return MVNStandard(ms, Ps)





#%%

# fast_sms, fast_sPs = batch_fast_smoother(model, y)
# print(f"{fast_sms = }")

# slow_sms, slow_sPs = batch_slow_smoother(model, y)
# print(f"{slow_sms = }")

# print(f"{sequential_smoothed.mean = }")

# fast_fms, fast_fPs = batch_fast_filter(model, y)
# print(f"{fast_fms = }")
#
# print(f"{fast_rate_result_filtered.mean = }")
#
# slow_fms, slow_fPs = batch_slow_filter(model, y)
# print(f"{slow_fms = }")
#
# print(f"{sequential_filtered.mean = }")
