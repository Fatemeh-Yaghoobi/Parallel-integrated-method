import jax.numpy as jnp
from matplotlib import pyplot as plt

from integrated._base import MVNStandard
from integrated.sequential._slow_rate_params import _slow_rate_integrated_params
from integrated.sequential import integrated_filtering
from tests.linear.model import DistillationSSM

################################### Parameters ########################################
l = 5
N = 100
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

result = integrated_filtering(y, prior_x, slow_rate_params)
print(x[1:, 0].shape)
print(result.mean[1:, 0].shape)
plt.plot(x[:, 0], 'o--', color='b', label='x')
# plt.plot(range(l, len(x), l), h[:, 0], 'o--', color='r', label='h')
# plt.plot(range(l, len(x), l), y[:, 0], 'o', color='g', label='y')
plt.plot(range(l, len(x), l), result.mean[1:, 0], 'o--', color='y', label='filtered x')
plt.legend()
plt.show()


