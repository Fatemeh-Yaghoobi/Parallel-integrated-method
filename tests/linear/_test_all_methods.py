import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental.host_callback import id_print
from matplotlib import pyplot as plt

from integrated._utils import none_or_concat

jax.config.update('jax_platform_name', 'cpu')
jax.config.update("jax_enable_x64", True)

from integrated._base import MVNStandard
from integrated.inegrated_params import slow_rate_params, fast_rate_params, full_filtering_params
from integrated.sequential import seq_filtering_slow_rate, seq_smoothing_slow_rate, filtering_fast_rate, filtering_all_states, smoothing_fast_rate
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

############################################# Filtering - Slow rate - Sequential and Parallel ########################################
sequential_filtered = seq_filtering_slow_rate(y, prior_x, Params_SR)
parallel_filtered = par_filtering_slow_rate(y, prior_x, Params_SR)
np.testing.assert_allclose(sequential_filtered.mean, parallel_filtered.mean, rtol=1e-06, atol=1e-03)

####################################################### Filtering - Fast-rate ############################################################
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

############################################### Smoothing - Slow rate - Sequential ########################################################
# Selection the filtering cross-covariances between the first and last states in each interval, i.e., P^f_{k, 1, l}.
def selected_Pf_k1l(Pf_all_, l):
    nx = int(Pf_all.shape[-1] / l)
    N = Pf_all.shape[0]
    def body(_, i):
        return _, Pf_all_[i, 0:nx, (l-1)*nx:l*nx]

    _, selected =  jax.lax.scan(body, init = None, xs=jnp.arange(N))
    return selected
Pf_k1l = selected_Pf_k1l(all_filtering_covs, l)

# Selecting the first states of the filtering result in each interval, i.e., m^f_{1:N, 1} and P^f_{1:N, 1}.
def selected_fast_filtered(filtered_results, l):
    size = len(filtered_results.mean)
    def body(_, i):
        return _, (MVNStandard(filtered_results.mean[i], filtered_results.cov[i]),
                   MVNStandard(filtered_results.mean[i + l - 1], filtered_results.cov[i + l - 1]))

    _, (selected_1, selected_l)  =  jax.lax.scan(body, init = None, xs=jnp.arange(1, size, l))
    return selected_1, selected_l
sr_filtered_k1, sr_filtered_kl = selected_fast_filtered(fast_rate_result_filtered, l)
# Slow-rate Smoothing
fast_sms, fast_sPs = batch_fast_smoother(model, y)
sequential_smoothed_slow_rate = seq_smoothing_slow_rate(sr_filtered_k1, sr_filtered_kl, transition_model, Pf_k1l)

################################################## Smoothing - Fast rate - Sequential ##################################################
# Selection the filtering cross-covariances between each state and last states in each interval, i.e., P^f_{k, i, l} i = 1, ... , l.
def selected_Pf_k_all_l(Pf_all_, l):
    nx = int(Pf_all.shape[-1] / l)
    N = Pf_all.shape[0]
    def body(_, i):
        return _, Pf_all_[i, :, (l-1)*nx:l*nx]

    _, selected =  jax.lax.scan(body, init = None, xs=jnp.arange(N))
    return selected
Pf_k_all_l = selected_Pf_k_all_l(all_filtering_covs, l)
# fast rate smoothing
vmap_function = jax.vmap(smoothing_fast_rate, in_axes=(0, 0, 0, None, 0))
sequential_smoothed_fast_rate = vmap_function(MVNStandard(all_filtering_means[:-1, :], all_filtering_covs[:-1, :, :]),
                                              MVNStandard(sr_filtered_kl.mean[:-1, :], sr_filtered_kl.cov[:-1, :, :]),
                                              MVNStandard(sequential_smoothed_slow_rate.mean[1:, :], sequential_smoothed_slow_rate.cov[1:, :, :]),
                                              transition_model,
                                              Pf_k_all_l[:-1, :, :])

sequential_smoothed_fast_rate = none_or_concat(sequential_smoothed_fast_rate,
                                               MVNStandard(all_filtering_means[-1, :], all_filtering_covs[-1, :, :]),
                                               0)
seq_smoothed_fr_means = sequential_smoothed_fast_rate.mean.reshape(-1, nx)  # shape (N * l, nx)
seq_smoothed_fr_covs = jnp.zeros((N * l, nx, nx))
for i in range(N):
    for j in range(l):
        seq_smoothed_fr_covs = seq_smoothed_fr_covs.at[i * l + j].set(sequential_smoothed_fast_rate.cov[i, j*nx:(j+1)*nx, j*nx:(j+1)*nx])

np.testing.assert_allclose(seq_smoothed_fr_means, fast_sms[1:], rtol=1e-06, atol=1e-03)
np.testing.assert_allclose(seq_smoothed_fr_covs, fast_sPs[1:], rtol=1e-06, atol=1e-03)
######################################################### par_smoothing_all_rate########################################################################
par_smoothing = par_smoothing_slow_rate(transition_model,
                                        MVNStandard(all_filtering_means, all_filtering_covs),
                                        MVNStandard(sr_filtered_kl.mean[:-1, :], sr_filtered_kl.cov[:-1, :, :]),
                                        Pf_k_all_l[:-1, :, :])

print(par_smoothing.mean.shape)
