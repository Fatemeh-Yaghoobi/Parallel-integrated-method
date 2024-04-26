import jax
import jax.numpy as jnp
import numpy as np
from matplotlib import pyplot as plt

jax.config.update("jax_enable_x64", True)

from integrated._base import MVNStandard
from integrated.inegrated_params._slow_rate_params import _slow_rate_integrated_params
from integrated.sequential import integrated_filtering, integrated_smoothing
from integrated.parallel import parallel_filtering, parallel_smoothing
from tests.linear.model import DistillationSSM

################################### Parameters ########################################
l = 10
N = 1500
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
slow_rate_params = _slow_rate_integrated_params(transition_model, observation_model, l)


sequential_filtered = integrated_filtering(y, prior_x, slow_rate_params)
sequential_smoothed = integrated_smoothing(sequential_filtered, slow_rate_params)

parallel_filtered = parallel_filtering(y, prior_x, slow_rate_params)
parallel_smoothed = parallel_smoothing(parallel_filtered, slow_rate_params)

plt.semilogx(range(l, len(x), l), sequential_filtered.mean[1:, 0], '--', color='y', label='filtered x')
plt.semilogx(range(l, len(x), l), sequential_smoothed.mean[1:, 0], '--', color='g', label='smoothed x')
plt.semilogx(range(l, len(x), l), x[1::l, 0], '--', color='b', label='true x')
plt.plot(range(l, len(x), l), parallel_filtered.mean[1:, 0], '*--', color='r', label='parallel filtered x')
plt.legend()
plt.show()


