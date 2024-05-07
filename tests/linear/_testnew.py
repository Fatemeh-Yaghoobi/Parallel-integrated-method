import jax
import jax.numpy as jnp
import numpy as np
from jax import lax, jit
jax.config.update("jax_enable_x64", True)
from integrated._base import MVNStandard
from integrated.inegrated_params import all_params, slow_rate_params, new_params
from integrated.sequential import integrated_filtering, all_filtering
from tests.linear.model import DistillationSSM
################################### Parameters ########################################
l = 4
N = 5
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
parameters_all = all_params(transition_model, observation_model, l)
parameters_new = new_params(transition_model, observation_model, l)
