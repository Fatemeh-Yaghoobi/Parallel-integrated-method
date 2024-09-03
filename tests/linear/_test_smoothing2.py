# This is a simple test in special case when l=2 and N=2



import jax
import jax.numpy as jnp
import numpy as np
from jax.lax import associative_scan

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
joint_fast_fms, joint_fast_fPs = batch_joint_fast_filter(model, y)
fast_sms, fast_sPs = batch_fast_smoother(model, y)

print(joint_fast_fms.shape)

#%%


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

# Parameters:
#    mf1, Pf1 = filtering mean and covariance for x_{k,1}
#    mfl, Pfl = filtering mean and covariance for x_{k,l}
#        Pf1l = joint covariance for x_{k,1} and x_{k,l} from filter
#    ms_next1, Ps_next1 = smoother mean and covariance for x_{k+1,1}
#
# Returns:
#    ms1, Ps1 = smoother mean and covariance for x_{k,1}
def slow_smoothed(mf1, Pf1, mfl, Pfl, Pf1l, ms_next1, Ps_next1):
    A, B, u, Q1 = transition_model

    m_ = A @ mfl + (B @ u).reshape(-1)
    P_ = A @ Pfl @ A.T + Q1

    G = Pf1l @ A.T @ jnp.linalg.inv(P_)
    ms1 = mf1 + G @ (ms_next1 - m_)
    Ps1 = Pf1 + G @ (Ps_next1 - P_) @ G.T
    return MVNStandard(ms1, Ps1)

### smoothed value for x_1 from x_3
mf1 = fast_fms[1]
Pf1 = fast_fPs[1]
mfl = fast_fms[2]
Pfl = fast_fPs[2]
Pf1l = joint_fast_fPs[0][:nx, nx:]

ss = slow_smoothed(mf1, Pf1, mfl, Pfl, Pf1l, ms3, Ps3)
ms1 = ss.mean
Ps1 = ss.cov
print(ms1)
print(fast_sms[1])
#print(Ps1)
#print(fast_sPs[1])

np.testing.assert_allclose(ms1, fast_sms[1], rtol=1e-06, atol=1e-03)
np.testing.assert_allclose(Ps1, fast_sPs[1], rtol=1e-06, atol=1e-03)

print("smoothed value for x_1 is matched with the batch fast smoother")

