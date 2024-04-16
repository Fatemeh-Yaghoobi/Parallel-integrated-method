import jax.numpy as jnp

from integrated._base import MVNStandard
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
print(x.shape, h.shape, y.shape)


transition_model = model.TranParams()
observation_model = model.ObsParams()


x_hat, h_hat = integrated_filtering(y, prior_x, transition_model, observation_model, l)




