import jax
import jax.numpy as jnp
import numpy as np
from matplotlib import pyplot as plt

jax.config.update('jax_platform_name', 'cpu')
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

################################### Parameters ########################################
transition_model = model.TranParams()
observation_model = model.ObsParams()
Params_SR = slow_rate_params(transition_model, observation_model, l)
Params_FR = fast_rate_params(transition_model, l)

### Filtering - Slow rate - Sequential and Parallel ###
sequential_filtered = integrated_filtering(y, prior_x, Params_SR)
parallel_filtered = parallel_filtering(y, prior_x, Params_SR)
np.testing.assert_allclose(sequential_filtered.mean, parallel_filtered.mean, rtol=1e-06, atol=1e-03)

### Smoothing - Slow rate - Sequential and Parallel ###
sequential_smoothed = integrated_smoothing(sequential_filtered, Params_SR)
parallel_smoothed = parallel_smoothing(parallel_filtered, Params_SR)
np.testing.assert_allclose(sequential_smoothed.mean, parallel_smoothed.mean, rtol=1e-06, atol=1e-03)

### Fast-rate Filtering ###



