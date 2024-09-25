import jax
import jax.numpy as jnp
import numpy as np

from integrated._utils import none_or_concat

jax.config.update('jax_platform_name', 'cpu')
jax.config.update("jax_enable_x64", True)

from integrated._base import MVNStandard
from integrated.inegrated_params import slow_rate_params, fast_rate_params, full_filtering_params
from integrated.sequential import seq_filtering_slow_rate, seq_smoothing_slow_rate, filtering_fast_rate, filtering_all_states, smoothing_fast_rate
from tests.linear.model import DistillationSSM
################################### Parameters ########################################
n_length_space = 4
l = 2**n_length_space
N = 200
nx = 4
ny = 2
Q = 1
R = 1
prior_x = MVNStandard(jnp.zeros(4), jnp.eye(4))
n_runs = 100
RMSE_filter = []
RMSE_smoother = []
for i in range(n_runs):
    ################################ Model: Distillation Column ##########################
    seed = np.random.randint(0, 1000)
    model = DistillationSSM(l=l, interval=N, nx=nx, ny=ny, Q=Q, R=R, prior_x=prior_x, seed=seed)  # noqa
    x, _, y = model.get_data()
    ################################### Parameters ########################################
    transition_model = model.TranParams()
    observation_model = model.ObsParams()
    Params_SR = slow_rate_params(transition_model, observation_model, l)
    Params_FR = fast_rate_params(transition_model, observation_model, l)
    Params_all_states = full_filtering_params(transition_model, model.ObsParams(), l)

    #################################### Filtering  ########################################
    sequential_filtered = seq_filtering_slow_rate(y, prior_x, Params_SR)
    ### Filtering - All States ###
    m_l_k_1 = sequential_filtered.mean[:-1]
    P_l_k_1 = sequential_filtered.cov[:-1]
    vmap_func_all_states = jax.vmap(filtering_all_states, in_axes=(0, None, 0))
    all_states_filtered_ = vmap_func_all_states(y, Params_all_states, MVNStandard(m_l_k_1, P_l_k_1))
    all_filtering_means, all_filtering_covs = all_states_filtered_.mean, all_states_filtered_.cov
    filtered_results = all_filtering_means.reshape(-1, 4)  # shape (N * l, nx)
    ############################################### Smoothing - Slow rate - Sequential ########################################################
    # Selection the filtering cross-covariances between the first and last states in each interval, i.e., P^f_{k, 1, l}.
    def selected_Pf_k1l(Pf_all_, l):
        def body(_, i):
            return _, Pf_all_[i, 0:nx, (l-1)*nx:l*nx]

        _, selected =  jax.lax.scan(body, init = None, xs=jnp.arange(N))
        return selected
    Pf_k1l = selected_Pf_k1l(all_filtering_covs, l)

    # Selecting the first states of the filtering result in each interval, i.e., m^f_{1:N, 1} and P^f_{1:N, 1}.
    def selected_fast_filtered(filtered_results, l):
        def body(_, i):
            return _, (MVNStandard(filtered_results.mean[i, 0:nx], filtered_results.cov[i, 0:nx, 0:nx]),
                       MVNStandard(filtered_results.mean[i, (l-1)*nx:l*nx], filtered_results.cov[i, (l-1)*nx:l*nx, (l-1)*nx:l*nx]))

        _, (selected_1, selected_l)  =  jax.lax.scan(body, init = None, xs=jnp.arange(0, N, 1))
        return selected_1, selected_l
    sr_filtered_k1, sr_filtered_kl = selected_fast_filtered(MVNStandard(all_filtering_means, all_filtering_covs), l)

    # Slow-rate Smoothing
    sequential_smoothed_slow_rate = seq_smoothing_slow_rate(sr_filtered_k1, sr_filtered_kl, transition_model, Pf_k1l)
    ########## Smoothing - Fast rate
    # Selection the filtering cross-covariances between each state and last states in each interval, i.e., P^f_{k, i, l} i = 1, ... , l.
    def selected_Pf_k_all_l(Pf_all_, l):
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
    smoothed_results = sequential_smoothed_fast_rate.mean.reshape(-1, nx)  # shape (N * l, nx)
    rmse_filter = jnp.sqrt(jnp.mean((filtered_results[:, 0] - x[1:, 0]) ** 2))
    rmse_smoother = jnp.sqrt(jnp.mean((smoothed_results[:, 0] - x[1:, 0]) ** 2))
    print(f"run {i + 1} out of {n_runs}", end="\n")
    RMSE_filter.append(rmse_filter)
    RMSE_smoother.append(rmse_smoother)


# print(f"{np.mean(RMSE_filter) = }")
# print(f"{np.mean(RMSE_smoother) = }")
# print(f"{np.std(RMSE_filter) = }")
# print(f"{np.std(RMSE_smoother) = }")
# jnp.savez("results/rmse_experiment_L16_N200_100runs.npz", RMSE_filter_mean=np.mean(RMSE_filter), RMSE_smoother_mean=np.mean(RMSE_smoother))
