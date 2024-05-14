import jax
import jax.numpy as jnp
import numpy as np
from matplotlib import pyplot as plt

from integrated._base import MVNStandard

jax.config.update("jax_enable_x64", True)

from integrated.inegrated_params import new_params, all_params
from tests.linear.model import DistillationSSM

################################### Parameters ########################################
l = 2
N = 10
nx = 4
ny = 2
Q = 0.1
R = 0.1
prior_x = MVNStandard(jnp.zeros(4), jnp.eye(4))
seed = 3
################################ Model: Distillation Column ##########################
model = DistillationSSM(l=l, interval=N, nx=nx, ny=ny, Q=Q, R=R, prior_x=prior_x, seed=seed)  # noqa
# x, h, y = model.get_data()
################################### Filtering ########################################
transition_model = model.TranParams()
observation_model = model.ObsParams()
parameters_NEW = new_params(transition_model, observation_model, l)
parameters_OLD = all_params(transition_model, observation_model, l)
#########################################################################################
A, B, u, cov = transition_model
C, R = observation_model
#########################################################################################
Abar, Bbar, Gbar, Bbar_u, Qbar, H, R1 = parameters_NEW
Abar11, Bbar11, Gbar11, Bbar_u11, Qbar11, C_bar, M_bar, D_bar, Rx, Q, Du_bar, r1 = parameters_OLD
np.testing.assert_allclose(Abar, Abar11, rtol=1e-06, atol=0)
np.testing.assert_allclose(Bbar, Bbar11, rtol=1e-06, atol=0)
np.testing.assert_allclose(Gbar, Gbar11, rtol=1e-06, atol=0)
