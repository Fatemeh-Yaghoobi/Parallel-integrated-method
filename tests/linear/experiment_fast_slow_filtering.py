import jax
import jax.numpy as jnp
import numpy as np
import time

from jax import jit
from matplotlib import pyplot as plt

jax.config.update("jax_enable_x64", True)

from integrated._base import MVNStandard
from integrated.inegrated_params import all_params
from integrated.sequential import all_filtering
from integrated.parallel import parallel_filtering_fast_slow
from tests.linear.model import DistillationSSM

################################### Parameters ########################################
l = 10
N = 50000
T = N
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
fast_slow_parameters = all_params(transition_model, observation_model, l)


def func(method, lengths, n_runs=100):
    res_mean = []
    res_median = []
    for k, j in enumerate(lengths):
        print(f"Iteration {k + 1} out of {len(lengths)}", end='\n')
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
            print(f"run {i + 1} out of {n_runs}", end='\n')
        res_mean.append(np.mean(run_times))
        res_median.append(np.median(run_times))
    print()
    return np.array(res_mean), np.array(res_median)


lengths_space = np.logspace(2, int(np.log2(T)), num=10, base=2, dtype=int)


###############################################################################
def sequential_filtering(y):
    return all_filtering(y, prior_x, fast_slow_parameters)

def parallel_filtering_(y):
    return parallel_filtering_fast_slow(y, prior_x, fast_slow_parameters)


###############################################################################
gpu_sequential_filtering = jit(sequential_filtering, backend="gpu")
gpu_parallel_filtering = jit(parallel_filtering_, backend="gpu")
###############################################################################
gpu_par_filter_mean_times, gpu_par_filter_median_times = func(gpu_parallel_filtering, lengths_space)
# jnp.savez("linear/results/all_gpu_par_filter_times_l_30",
#           gpu_par_filter_mean_times=gpu_par_filter_mean_times,
#           gpu_par_filter_median_times=gpu_par_filter_median_times)
# ###############################################################################
#
# ###############################################################################
gpu_seq_filter_mean_times, gpu_seq_filter_median_times = func(gpu_sequential_filtering, lengths_space)
# jnp.savez("linear/results/all_gpu_seq_filter_times_l_30",
#           gpu_seq_filter_mean_times=gpu_seq_filter_mean_times,
#           gpu_seq_filter_median_times=gpu_seq_filter_median_times)
###############################################################################

###############################################################################
plt.loglog(lengths_space, gpu_par_filter_mean_times, label="par filter", linestyle="-.", linewidth=3)
plt.loglog(lengths_space, gpu_seq_filter_mean_times, label="seq filter", linestyle="-.", linewidth=3)
plt.title("GPU Mean Run Times")
plt.grid(True, which="both")
plt.legend()
plt.show()
