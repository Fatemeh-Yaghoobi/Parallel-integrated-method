import jax
import jax.numpy as jnp

jax.config.update('jax_platform_name', 'cpu')

jax.config.update("jax_enable_x64", True)

from integrated._base import MVNStandard
from tests.linear.model import DistillationSSM

from integrated.batch import *

################################### Parameters ########################################
l = 4
N = 20
nx = 4
ny = 2
Q = 1
R = 1
prior_x = MVNStandard(jnp.zeros(4), jnp.eye(4))
seed = 0
################################ Model: Distillation Column ##########################
model = DistillationSSM(l=l, interval=N, nx=nx, ny=ny, Q=Q, R=R, prior_x=prior_x, seed=seed)  # noqa
x, h, y = model.get_data()

fast_sms, fast_sPs = batch_fast_smoother(model, y)
print(f"{fast_sms.shape = }")

fast_fms, fast_fPs = batch_fast_filter(model, y)
print(f"{fast_fms.shape = }")
################################### RMSE test ########################################
rmse_filter = jnp.sqrt(jnp.mean((fast_fms[:, 0] - x[:, 0]) ** 2))
rmse_smoother = jnp.sqrt(jnp.mean((fast_sms[:, 0] - x[:, 0]) ** 2))
print(f"{rmse_filter = }")
print(f"{rmse_smoother = }")
