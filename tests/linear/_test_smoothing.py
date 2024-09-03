 # This is a simple test in special case when l=2 and N=2

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update('jax_platform_name', 'cpu')
jax.config.update("jax_enable_x64", True)

from integrated._base import MVNStandard
from tests.linear.model import DistillationSSM

from integrated.batch import *

################################### Parameters ########################################
l = 2
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


### Batch solution for fast filtering and smoothing
fast_fms, fast_fPs = batch_fast_filter(model, y)
fast_sms, fast_sPs = batch_fast_smoother(model, y)

######################################### Test ##########################################
#### smoothing test for l=2 and N=2, states are [x_0, x_1, x_2, x_3, x_4] ####
### states in the last interval are [x_3, x_4]  and they have the same filtered and smoothed values
ms4 = fast_sms[4]
mf4 = fast_fms[4]
ms3 = fast_sms[3]
mf3 = fast_fms[3]
np.testing.assert_allclose(ms4, mf4, rtol=1e-06, atol=1e-03)
np.testing.assert_allclose(ms3, mf3, rtol=1e-06, atol=1e-03)
Ps3 = fast_sPs[3]
### we want to find the smoothed values for x_2 and x_1
def x_smoothed(mf, Pf, ms, Ps):
    A, B, u, Q1 = transition_model

    m_ = A @ mf + (B @ u).reshape(-1)
    P_ = A @ Pf @ A.T + Q1

    G = Pf @ A.T @ jnp.linalg.inv(P_)
    ms = mf + G @ (ms - m_)
    Ps = Pf + G @ (Ps - P_) @ G.T
    return MVNStandard(ms, Ps)
### smoothed value for x_2 from x_3
mf2 = fast_fms[2]
Pf2 = fast_fPs[2]
ms2 = x_smoothed(mf2, Pf2, ms3, Ps3).mean
Ps2 = x_smoothed(mf2, Pf2, ms3, Ps3).cov
np.testing.assert_allclose(ms2, fast_sms[2], rtol=1e-06, atol=1e-03)
np.testing.assert_allclose(Ps2, fast_sPs[2], rtol=1e-06, atol=1e-03)

print("smoothed value for x_2 is matched with the batch fast smoother")

### smoothed value for x_1 from x_2
mf1 = fast_fms[1]
Pf1 = fast_fPs[1]
ms1 = x_smoothed(mf1, Pf1, ms2, Ps2).mean
Ps1 = x_smoothed(mf1, Pf1, ms2, Ps2).cov
np.testing.assert_allclose(ms1, fast_sms[1], rtol=1e-06, atol=1e-03)
np.testing.assert_allclose(Ps1, fast_sPs[1], rtol=1e-06, atol=1e-03)


