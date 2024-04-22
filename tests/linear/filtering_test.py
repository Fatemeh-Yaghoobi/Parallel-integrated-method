import jax
import jax.numpy as jnp
from jax import jit
from matplotlib import pyplot as plt

from integrated._base import MVNStandard
from integrated.sequential._integrated_params import _slow_rate_integrated_params
from integrated.sequential import integrated_filtering
from tests.linear.model import DistillationSSM

################################### Parameters ########################################
l = 10
N = 10
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
# plt.plot(x[:, 0], 'o', color='b', label='x')
# plt.plot(range(10, len(x), 10), h[:, 0], 'o--', color='r', label='h')
# plt.plot(range(10, len(x), 10), y[:, 0], 'o', color='g', label='y')
# plt.legend()
# plt.show()
################################### Filtering ########################################
transition_model = model.TranParams()
observation_model = model.ObsParams()
slow_rate_params = _slow_rate_integrated_params(transition_model, l)

from functools import partial
A, B, u, Q = transition_model
print(u)

@partial(jit, static_argnums=0)
def body(val):
    x, i = val
    return (jnp.stack(x, x), i + 1)


jax.lax.while_loop(lambda i: i[1] < l, body, (u, 0))

# @partial(jit, static_argnums=1)
# def body_scan(_, i):
#     out = jnp.array(jnp.stack([u] * i))
#     return _, out
#
# jax.lax.scan(body_scan, None, jnp.arange(l))
