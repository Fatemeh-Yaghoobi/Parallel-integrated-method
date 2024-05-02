import jax
import jax.numpy as jnp
import numpy as np
from matplotlib import pyplot as plt

jax.config.update("jax_enable_x64", True)

from integrated._base import MVNStandard
from integrated.inegrated_params import all_params, slow_rate_params
from integrated.sequential import all_filtering, integrated_filtering
from tests.linear.model import DistillationSSM

################################### Parameters ########################################
l = 2
N = 15
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
parameters_slow = slow_rate_params(transition_model, observation_model, l)

Abar, Bbar, Gbar, Bbar_u, Qbar, C_bar, M_bar, D_bar, Rx, Q, Du_bar = parameters_all
A_bar, G_bar, B_bar, u_bar, Bu_bar, Q_bar, _, _, _, _, _, _ = parameters_slow

def _integrated_predict(slow_rate_params, x):
    m, P = x
    A_bar, _, _, _, Bu_bar, Q_bar, _, _, _, _, _, _ = slow_rate_params

    m = A_bar @ m + Bu_bar
    P = A_bar @ P @ A_bar.T + Q_bar
    return MVNStandard(m, P)

x_predict_slow = _integrated_predict(parameters_slow, prior_x)
def _integrated_predict_all(all_params, x):
    m, P = x
    Abar, _, _, Bbar_u, Qbar, _, _, _, _, _, _ = all_params

    m = Abar @ m + Bbar_u                                      # dim = l x nx
    P = Abar @ P @ jnp.transpose(Abar, axes=(0, 2, 1)) + Qbar  # dim = l x nx x nx
    return MVNStandard(m, P)

x_predict_all = _integrated_predict_all(parameters_all, prior_x)
def _integrated_update(slow_rate_params, xl_k_pred, xl_k_1, y):
    m_, P_ = xl_k_pred
    m_k_1, P_k_1 = xl_k_1
    A_bar, G_bar, B_bar, u_bar, Bu_bar, Q_bar, C_bar, M_bar, D_bar, Rx, Q, Du_bar = slow_rate_params

    S = C_bar @ P_k_1 @ C_bar.T + Rx
    A = jnp.array([[0.8499, 0.0350, 0.0240, 0.0431],
                   [1.2081, 0.0738, 0.0763, 0.4087],
                   [0.7331, 0.0674, 0.0878, 0.8767],
                   [0.0172, 0.0047, 0.0114, 0.9123]])
    temp = A_bar @ P_k_1 @ C_bar.T + jnp.sum(G_bar @ Q @ jnp.transpose(M_bar, axes=(0, 2, 1)), axis=0)
    print(jnp.sum(G_bar @ Q @ jnp.transpose(M_bar, axes=(0, 2, 1)), axis=0) == A @ Q @ M_bar[0].T + Q @ M_bar[1].T)
    L = jax.scipy.linalg.solve(S, temp.T).T
    m = m_ + L @ (y - C_bar @ m_k_1 - Du_bar)
    P = P_ - L @ temp.T

    return MVNStandard(m, P)

x_update_slow = _integrated_update(parameters_slow, x_predict_slow, prior_x, y[0])


def _integrated_update_all(all_params, x_predict_interval, xl_k_1, y):
    m_, P_ = x_predict_interval
    m_k_1, P_k_1 = xl_k_1
    Abar, Bbar, Gbar, Bbar_u, Qbar, C_bar, M_bar, D_bar, Rx, Q, Du_bar = all_params

    S = C_bar @ P_k_1 @ C_bar.T + Rx  # dim = ny x ny

    temp = (Abar @ P_k_1 @ C_bar.T
            + jnp.einsum('ijkl,ilm->ikm', Gbar @ Q, jnp.transpose(M_bar, axes=(0, 2, 1))))   # dim = l x nx x ny
    print(jnp.einsum('ijkl,ilm->ikm', Gbar @ Q, jnp.transpose(M_bar, axes=(0, 2, 1))))
    vmap_func = jax.vmap(jax.scipy.linalg.solve, in_axes=(None, 0))
    TranL = vmap_func(S, jnp.transpose(temp, axes=(0, 2, 1)))
    L = jnp.transpose(TranL, axes=(0, 2, 1))                                                 # dim = l x nx x ny
    # Cbar_m = jnp.einsum('ij,lj->li', C_bar, m_k_1)                                           # dim = l x ny

    m = m_ + jnp.einsum('ijk,k->ij', L, y -  C_bar @ m_k_1 - Du_bar)                        # dim = l x nx
    tempT = jnp.transpose(temp, axes=(0, 2, 1))                                              # dim = l x ny x nx
    P = P_ - jnp.einsum('ijk,ikl->ijl', L, tempT)                                            # dim = l x nx x nx

    return MVNStandard(m, P)

x_update_all = _integrated_update_all(parameters_all, x_predict_all, prior_x, y[0])
# print(x_update_all.mean)
# print(x_update_slow.mean)
# sequential_filtered = all_filtering(y, prior_x, parameters_all)
# seq_res_all = sequential_filtered.mean.reshape(-1, 4)
#
# seq_filter_slow = integrated_filtering(y, prior_x, parameters_slow)
# # plt.semilogx(x[1:, 0], '--', color='b', label='true x')
# plt.plot(seq_res_all[:, 0], '.--', color='r', label='sequential filtered x')
# plt.plot(range(l-1, len(x) - 1, l), seq_filter_slow.mean[1:, 0], '*--', color='y', label='filtered x - slow')
# plt.grid()
# plt.legend()
# plt.show()


