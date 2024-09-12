import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)
from integrated._base import MVNStandard
from integrated._utils import none_or_concat

# This code provides the filtering means and covariances for the all states including the cross-covariances between the states.

def filtering(y: jnp.ndarray,
              all_params,
              xl_k_1):

    x_predict_interval = _integrated_predict(all_params, xl_k_1)
    x_update_interval = _integrated_update(all_params, x_predict_interval, y)

    return x_update_interval


def _integrated_predict(all_params, x):
    m, P = x
    Abar, Gbar, _, Bbar_u, Qbar, _, _ = all_params

    m = Abar @ m + Bbar_u                                      # dim = lnx,
    P = Abar @ P @ Abar.T + Qbar                               # dim = lnx x lnx
    return MVNStandard(m, P)


def _integrated_update(all_params, x_predict_interval, y):
    m_, P_ = x_predict_interval
    Abar, Gbar, Bbar, Bbar_u, Qbar, H, R = all_params

    S = H @ P_ @ H.T + R                                 # dim = ny x ny
    temp = P_ @ H.T
    L = jax.scipy.linalg.solve(S, temp.T).T
    m = m_ + L @ (y - H @ m_)
    P = P_ - L @ temp.T
    return MVNStandard(m, P)


