import jax
import jax.numpy as jnp
import numpy
import numpy as np
from jax import jit
from jax.lax import while_loop, scan
from matplotlib import pyplot as plt

from integrated._base import MVNStandard
from integrated.sequential._integrated_params import _slow_rate_integrated_params
from integrated.sequential import integrated_filtering
from tests.linear.model import DistillationSSM

################################### Parameters ########################################
l = 5
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

# integrated_filtering(observations=y,
#                      x0=prior_x,
#                      transition_model=transition_model,
#                      observation_model=observation_model,
#                      slow_rate_params=slow_rate_params,
#                      l=l)

from functools import partial
A, B, u, Q = transition_model

# @partial(jit, static_argnums=(0,))
# def out_body(carry):
#     j, val = carry
#     u1 = np.zeros((j, 1, 1))
#
#     def body(carry):
#         i, x = carry
#         return i + 1, x.at[i].set(u)
#
#     def cond(carry):
#         i, _val = carry
#         return i < j
#
#     _, out = while_loop(cond, body, init_val=(0, u1))
#     return _, out
#
# jax.lax.while_loop(lambda carry: carry[0]<l + 1, out_body, init_val=(1, 0.))


@partial(jit, static_argnums=(0,))
def out_body(carry):
    j, val = carry

    def body(carry):
        i, x = carry
        return i + 1, x @ x

    def cond(carry):
        i, _val = carry
        return i < j

    _, out = while_loop(cond, body, init_val=(0, val))

    return j + 1, out

_, result = jax.lax.while_loop(lambda carry: carry[0]<l + 1, out_body, init_val=(1, jnp.eye(2)))
print(result.shape)