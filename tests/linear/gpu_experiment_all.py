import jax
import numpy as np
import jax.numpy as jnp
import time

from jax import jit
from matplotlib import pyplot as plt

from integrated._utils import none_or_concat

jax.config.update("jax_enable_x64", True)

from integrated._base import MVNStandard
from integrated.inegrated_params import slow_rate_params, fast_rate_params, full_filtering_params
from integrated.sequential import (seq_filtering_slow_rate, seq_smoothing_slow_rate,
                                   filtering_all_states, smoothing_fast_rate, filtering_fast_rate)
from integrated.parallel import par_filtering_slow_rate, par_smoothing_slow_rate
from tests.linear.model import DistillationSSM

################################### Parameters ########################################
n_length_space = 4
l = 2**n_length_space
N = 4000
T = N * l
nx = 4
ny = 2
Q = 1
R = 1
prior_x = MVNStandard(jnp.zeros(4), jnp.eye(4))
seed = 0
################################ Model: Distillation Column ##########################
model = DistillationSSM(l=l, interval=N, nx=nx, ny=ny, Q=Q, R=R, prior_x=prior_x, seed=seed)  # noqa
x, _, y = model.get_data()
transition_model = model.TranParams()
observation_model = model.ObsParams()
Params_SR = slow_rate_params(transition_model, observation_model, l)
Params_FR = fast_rate_params(transition_model, observation_model, l)
Params_all_states = full_filtering_params(transition_model, model.ObsParams(), l)


def func(method, lengths, n_runs=100):
    res_mean = []
    res_median = []
    for k, j in enumerate(lengths):
        print(f"Iteration {k + 1} out of {len(lengths)}\r", end="")
        observations_slice = y[:j]
        s = method(observations_slice)
        s.mean.block_until_ready()
        run_times = []
        for i in range(n_runs):
            tic = time.time()
            s_states = method(observations_slice)
            s_states.mean.block_until_ready()
            toc = time.time()
            run_times.append(toc - tic)
            print(f"run {i + 1} out of {n_runs}\r", end="")
        res_mean.append(np.mean(run_times))
        res_median.append(np.median(run_times))
    print()
    return np.array(res_mean), np.array(res_median)


lengths_space = np.logspace(n_length_space, int(np.log2(T)), num=10, base=2, dtype=int)
print(lengths_space)

###############################################################################
def sequential_filtering(y):
    # slow-rate
    sequential_filtered_sr = seq_filtering_slow_rate(y, prior_x, Params_SR)
    # Filtering - All States # with considering the mutual cross-covariances
    m_l_k_1 = sequential_filtered_sr.mean[:-1]
    P_l_k_1 = sequential_filtered_sr.cov[:-1]
    vmap_func_all_states = jax.vmap(filtering_all_states, in_axes=(0, None, 0))
    all_states_filtered_ = vmap_func_all_states(y, Params_all_states, MVNStandard(m_l_k_1, P_l_k_1))
    all_filtering_means, all_filtering_covs = all_states_filtered_.mean, all_states_filtered_.cov
    fast_rate_result_filtered = MVNStandard(all_filtering_means, all_filtering_covs)
    return fast_rate_result_filtered


def sequential_smoothing(y):
    fast_rate_result_filtered = sequential_filtering(y)
    all_filtering_means, all_filtering_covs = fast_rate_result_filtered.mean, fast_rate_result_filtered.cov
    # params-preparation - same in sequential and parallel version
    def selected_Pf_k1l(Pf_all_, l):
        size = Pf_all_.shape[0]
        def body(_, i):
            return _, Pf_all_[i, 0:nx, (l - 1) * nx:l * nx]

        _, selected = jax.lax.scan(body, init=None, xs=jnp.arange(size))
        return selected
    Pf_k1l = selected_Pf_k1l(all_filtering_covs, l)

    # Selecting the first states of the filtering result in each interval, i.e., m^f_{1:N, 1} and P^f_{1:N, 1}.
    def selected_fast_filtered(filtered_results, l):
        size = filtered_results.mean.shape[0]
        def body(_, i):
            return _, (MVNStandard(filtered_results.mean[i, 0:nx], filtered_results.cov[i, 0:nx, 0:nx]),
                       MVNStandard(filtered_results.mean[i, (l - 1) * nx:l * nx],
                                   filtered_results.cov[i, (l - 1) * nx:l * nx, (l - 1) * nx:l * nx]))

        _, (selected_1, selected_l) = jax.lax.scan(body, init=None, xs=jnp.arange(0, size, 1))
        return selected_1, selected_l
    sr_filtered_k1, sr_filtered_kl = selected_fast_filtered(fast_rate_result_filtered, l)
    # slow-rate
    sequential_smoothed_slow_rate = seq_smoothing_slow_rate(sr_filtered_k1, sr_filtered_kl, transition_model, Pf_k1l)
    # params-preparation - same in sequential and parallel version
    def selected_Pf_k_all_l(Pf_all_, l):
        size = Pf_all_.shape[0]
        def body(_, i):
            return _, Pf_all_[i, :, (l - 1) * nx:l * nx]

        _, selected = jax.lax.scan(body, init=None, xs=jnp.arange(size))
        return selected

    Pf_k_all_l = selected_Pf_k_all_l(all_filtering_covs, l)
    # fast rate smoothing
    vmap_function = jax.vmap(smoothing_fast_rate, in_axes=(0, 0, 0, None, 0))
    sequential_smoothed_fast_rate = vmap_function(
        MVNStandard(all_filtering_means[:-1, :], all_filtering_covs[:-1, :, :]),
        MVNStandard(sr_filtered_kl.mean[:-1, :], sr_filtered_kl.cov[:-1, :, :]),
        MVNStandard(sequential_smoothed_slow_rate.mean[1:, :], sequential_smoothed_slow_rate.cov[1:, :, :]),
        transition_model,
        Pf_k_all_l[:-1, :, :])

    sequential_smoothed_fast_rate = none_or_concat(sequential_smoothed_fast_rate,
                                                   MVNStandard(all_filtering_means[-1, :],
                                                               all_filtering_covs[-1, :, :]),
                                                   0)
    return sequential_smoothed_fast_rate


def parallel_filtering_(y):
    # slow-rate
    parallel_filtered_sr = par_filtering_slow_rate(y, prior_x, Params_SR)
    # Filtering - All States # with considering the mutual cross-covariances
    m_l_k_1 = parallel_filtered_sr.mean[:-1]
    P_l_k_1 = parallel_filtered_sr.cov[:-1]
    vmap_func_all_states = jax.vmap(filtering_all_states, in_axes=(0, None, 0))
    all_states_filtered_ = vmap_func_all_states(y, Params_all_states, MVNStandard(m_l_k_1, P_l_k_1))
    all_filtering_means, all_filtering_covs = all_states_filtered_.mean, all_states_filtered_.cov
    fast_rate_result_filtered = MVNStandard(all_filtering_means, all_filtering_covs)
    return fast_rate_result_filtered


def parallel_smoothing_(y):
    fast_rate_result_filtered = parallel_filtering_(y)
    all_filtering_means, all_filtering_covs = fast_rate_result_filtered.mean, fast_rate_result_filtered.cov
    # params preparation - same in sequential and parallel version
    # Selecting the first states of the filtering result in each interval, i.e., m^f_{1:N, 1} and P^f_{1:N, 1}.
    def selected_fast_filtered(filtered_results, l):
        size = filtered_results.mean.shape[0]
        def body(_, i):
            return _, (MVNStandard(filtered_results.mean[i, 0:nx], filtered_results.cov[i, 0:nx, 0:nx]),
                       MVNStandard(filtered_results.mean[i, (l - 1) * nx:l * nx],
                                   filtered_results.cov[i, (l - 1) * nx:l * nx, (l - 1) * nx:l * nx]))

        _, (selected_1, selected_l) = jax.lax.scan(body, init=None, xs=jnp.arange(0, size, 1))
        return selected_1, selected_l

    sr_filtered_k1, sr_filtered_kl = selected_fast_filtered(fast_rate_result_filtered, l)

    def selected_Pf_k_all_l(Pf_all_, l):
        size = Pf_all_.shape[0]
        def body(_, i):
            return _, Pf_all_[i, :, (l - 1) * nx:l * nx]

        _, selected = jax.lax.scan(body, init=None, xs=jnp.arange(size))
        return selected

    Pf_k_all_l = selected_Pf_k_all_l(all_filtering_covs, l)
    # slow-rate
    par_smoothing_sr = par_smoothing_slow_rate(transition_model,
                                            MVNStandard(all_filtering_means[:, 0:4], all_filtering_covs[:, 0:4, 0:4]),
                                            MVNStandard(sr_filtered_kl.mean[:-1, :], sr_filtered_kl.cov[:-1, :, :]),
                                            Pf_k_all_l[:-1, 0:4, :])
    # fast-rate
    vmap_function = jax.vmap(smoothing_fast_rate, in_axes=(0, 0, 0, None, 0))
    smoothed_fast_rate = vmap_function(
        MVNStandard(all_filtering_means[:-1, :], all_filtering_covs[:-1, :, :]),
        MVNStandard(sr_filtered_kl.mean[:-1, :], sr_filtered_kl.cov[:-1, :, :]),
        MVNStandard(par_smoothing_sr.mean[1:, :], par_smoothing_sr.cov[1:, :, :]),
        transition_model,
        Pf_k_all_l[:-1, :, :])

    smoothed_fast_rate = none_or_concat(smoothed_fast_rate,
                                        MVNStandard(all_filtering_means[-1, :],all_filtering_covs[-1, :, :]),0)
    return smoothed_fast_rate
###############################################################################
gpu_sequential_filtering = jit(sequential_filtering, backend="gpu")
gpu_sequential_smoothing = jit(sequential_smoothing, backend="gpu")
gpu_parallel_filtering = jit(parallel_filtering_, backend="gpu")
gpu_parallel_smoothing = jit(parallel_smoothing_, backend="gpu")
###############################################################################
gpu_par_filter_mean_times, gpu_par_filter_median_times = func(gpu_parallel_filtering, lengths_space)
jnp.savez("results/results_all_methods_final_version_L16_N4000/gpu_par_filter_times",
          gpu_par_filter_mean_times=gpu_par_filter_mean_times,
          gpu_par_filter_median_times=gpu_par_filter_median_times)
gpu_par_smooth_mean_times, gpu_par_smooth_median_times = func(gpu_parallel_smoothing, lengths_space)
jnp.savez("results/results_all_methods_final_version_L16_N4000/gpu_par_smooth_times",
          gpu_par_smooth_mean_times=gpu_par_smooth_mean_times,
          gpu_par_smooth_median_times=gpu_par_smooth_median_times)
gpu_seq_filter_mean_times, gpu_seq_filter_median_times = func(gpu_sequential_filtering, lengths_space)
jnp.savez("results/results_all_methods_final_version_L16_N4000/gpu_seq_filter_times",
          gpu_seq_filter_mean_times=gpu_seq_filter_mean_times,
          gpu_seq_filter_median_times=gpu_seq_filter_median_times)
gpu_seq_smooth_mean_times, gpu_seq_smooth_median_times = func(gpu_sequential_smoothing, lengths_space)
jnp.savez("results/results_all_methods_final_version_L16_N4000/gpu_seq_smooth_times",
          gpu_seq_smooth_mean_times=gpu_seq_smooth_mean_times,
          gpu_seq_smooth_median_times=gpu_seq_smooth_median_times)
###############################################################################
plt.loglog(lengths_space, gpu_par_filter_mean_times, label="par filter", linestyle="-.", linewidth=3)
plt.loglog(lengths_space, gpu_par_smooth_mean_times, label="par smoother", linestyle="-.", linewidth=3)
plt.loglog(lengths_space, gpu_seq_filter_mean_times, label="seq filter", linestyle="-.", linewidth=3)
plt.loglog(lengths_space, gpu_seq_smooth_mean_times, label="seq smoother", linestyle="-.", linewidth=3)
plt.title("GPU Mean Run Times")
plt.grid(True, which="both")
plt.legend()
plt.show()
