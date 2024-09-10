import jax
import jax.numpy as jnp
import numpy as np

jax.config.update('jax_platform_name', 'cpu')
jax.config.update("jax_enable_x64", True)

from integrated._base import MVNStandard
from tests.linear.model import DistillationSSM
from integrated.inegrated_params import full_filtering_params
################################### Parameters ########################################
l = 3
N = 2
nx = 4
ny = 2
Q = 1
R = 1
prior_x = MVNStandard(jnp.zeros(4), jnp.eye(4))
seed = 0
################################ Model: Distillation Column ##########################
model = DistillationSSM(l=l, interval=N, nx=nx, ny=ny, Q=Q, R=R, prior_x=prior_x, seed=seed)  # noqa
x, h, y = model.get_data()

transition_model = model.TranParams()
observation_model = model.ObsParams()
A, B, u, Q1 = transition_model
C, R = observation_model
################################### Parameters ########################################
full_params = full_filtering_params(transition_model, model.ObsParams(), l)
Abar, Gbar, Bbar, Bbar_u, Qbar, H, R = full_params

if l == 2:
    print("l is 2")
    np.testing.assert_allclose(Abar, jnp.concatenate([A, A @ A], axis=0), rtol=1e-06, atol=1e-03)
    temp1_Gbar = jnp.concatenate([jnp.eye(nx), jnp.zeros_like(A)], axis=1)
    temp2_Gbar = jnp.concatenate([A, jnp.eye(nx)], axis=1)
    np.testing.assert_allclose(Gbar, jnp.concatenate([temp1_Gbar, temp2_Gbar], axis=0), rtol=1e-06, atol=1e-03)
    temp1_Bbar = jnp.concatenate([B, jnp.zeros_like(B)], axis=1)
    temp2_Bbar = jnp.concatenate([A @ B, B], axis=1)
    np.testing.assert_allclose(Bbar, jnp.concatenate([temp1_Bbar, temp2_Bbar], axis=0), rtol=1e-06, atol=1e-03)
    temp1_Qbar = jnp.concatenate([Q1, Q1 @ A.T], axis=1)
    temp2_Qbar = jnp.concatenate([A @ Q1, A @ Q1 @ A.T + Q1], axis=1)
    np.testing.assert_allclose(Qbar, jnp.concatenate([temp1_Qbar, temp2_Qbar], axis=0), rtol=1e-06, atol=1e-03)
    np.testing.assert_allclose(H, jnp.concatenate([C/l, C/l], axis=1), rtol=1e-06, atol=1e-03)

elif l == 3:
    print("l is 3")
    np.testing.assert_allclose(Abar, jnp.concatenate([A, A @ A, A @ A @ A], axis=0), rtol=1e-06, atol=1e-03)
    temp1_Gbar = jnp.concatenate([jnp.eye(nx), jnp.zeros_like(A), jnp.zeros_like(A)], axis=1)
    temp2_Gbar = jnp.concatenate([A, jnp.eye(nx), jnp.zeros_like(A)], axis=1)
    temp3_Gbar = jnp.concatenate([A @ A, A, jnp.eye(nx)], axis=1)
    np.testing.assert_allclose(Gbar, jnp.concatenate([temp1_Gbar, temp2_Gbar, temp3_Gbar], axis=0), rtol=1e-06, atol=1e-03)
    temp1_Bbar = jnp.concatenate([B, jnp.zeros_like(B), jnp.zeros_like(B)], axis=1)
    temp2_Bbar = jnp.concatenate([A @ B, B, jnp.zeros_like(B)], axis=1)
    temp3_Bbar = jnp.concatenate([A @ A @ B, A @ B, B], axis=1)
    np.testing.assert_allclose(Bbar, jnp.concatenate([temp1_Bbar, temp2_Bbar, temp3_Bbar], axis=0), rtol=1e-06,
                               atol=1e-03)
    temp1_Qbar = jnp.concatenate([Q1, Q1 @ A.T, Q1 @ (A @ A).T], axis=1)
    temp2_Qbar = jnp.concatenate([A @ Q1, A @ Q1 @ A.T + Q1, A @ Q1 @ (A @ A).T + Q1 @ A.T], axis=1)
    temp3_Qbar = jnp.concatenate([A @ A @ Q1, A @ A @ Q1 @ A.T + A @ Q1, A @ A @ Q1 @ (A @ A).T + A @ Q1 @ A.T + Q1], axis=1)
    np.testing.assert_allclose(Gbar, jnp.concatenate([temp1_Gbar, temp2_Gbar, temp3_Gbar], axis=0), rtol=1e-06,
                               atol=1e-03)
    np.testing.assert_allclose(H, jnp.concatenate([C / l, C / l, C / l], axis=1), rtol=1e-06, atol=1e-03)

else:
    print("l is not 2 or 3")

