import jax
import jax.numpy as jnp
import numpy as np
from matplotlib import pyplot as plt

jax.config.update("jax_enable_x64", True)

from integrated._base import MVNStandard
from integrated.inegrated_params._all_params import _fast_and_slow_params
from integrated.sequential import all_filtering
from tests.linear.model import DistillationSSM

################################### Parameters ########################################
l = 10
N = 150
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
all_params = _fast_and_slow_params(transition_model, observation_model, l)


sequential_filtered = all_filtering(y, prior_x, all_params)
seq_res = sequential_filtered.mean.reshape(-1, 4)
print(x.shape)

plt.semilogx(x[1:, 0], '--', color='b', label='true x')
plt.semilogx(seq_res[:, 0], '.--', color='r', label='sequential filtered x')

# plt.semilogx(range(l, len(x), l), sequential_filtered.mean[1:, 0], '--', color='y', label='filtered x')
# plt.semilogx(range(l, len(x), l), sequential_smoothed.mean[1:, 0], '--', color='g', label='smoothed x')
# plt.semilogx(range(l, len(x), l), x[1::l, 0], '--', color='b', label='true x')
# plt.plot(range(l, len(x), l), parallel_filtered.mean[1:, 0], '*--', color='r', label='parallel filtered x')
plt.legend()
plt.show()


