import jax
import jax.numpy as jnp
import numpy as np
from matplotlib import pyplot as plt

from integrated._utils import none_or_concat

jax.config.update('jax_platform_name', 'cpu')
jax.config.update("jax_enable_x64", True)

from integrated._base import MVNStandard
from integrated.inegrated_params import slow_rate_params, fast_rate_params, full_filtering_params
from integrated.sequential import seq_filtering_slow_rate, seq_smoothing_slow_rate, filtering_fast_rate, filtering_all_states
from integrated.parallel import par_filtering_slow_rate, par_smoothing_slow_rate
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

################################### Parameters ########################################
transition_model = model.TranParams()
observation_model = model.ObsParams()
Params_SR = slow_rate_params(transition_model, observation_model, l)
Params_FR = fast_rate_params(transition_model, observation_model, l)
Params_all_states = full_filtering_params(transition_model, model.ObsParams(), l)

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
### Filtering - All States ###
vmap_func_all_states = jax.vmap(filtering_all_states, in_axes=(0, None, 0))
all_states_filtered_ = vmap_func_all_states(y, Params_all_states, MVNStandard(m_l_k_1, P_l_k_1))
all_filtering_means, all_filtering_covs = all_states_filtered_.mean, all_states_filtered_.cov
np.testing.assert_allclose(fast_rate_result_filtered.mean[1:], all_filtering_means.reshape(-1, 4), rtol=1e-06, atol=1e-03)

from integrated.batch import *
mf_all, Pf_all = batch_joint_fast_filter(model, y)
np.testing.assert_allclose(mf_all, all_filtering_means, rtol=1e-06, atol=1e-03)
np.testing.assert_allclose(Pf_all, all_filtering_covs, rtol=1e-06, atol=1e-03)

### Smoothing - Slow rate - Sequential and Parallel ###
# sequential_smoothed = seq_smoothing_slow_rate(sequential_filtered, Params_SR)
# parallel_smoothed = par_smoothing_slow_rate(parallel_filtered, Params_SR)
# np.testing.assert_allclose(sequential_smoothed.mean, parallel_smoothed.mean, rtol=1e-06, atol=1e-03)

### Fast-rate Smoothing ###



