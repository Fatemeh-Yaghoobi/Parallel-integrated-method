import jax
import jax.numpy as jnp

from integrated._base import MVNStandard, LinearTran, LinearObs, SlowRateIntegratedParams
from integrated._utils import none_or_concat
from integrated.sequential._integrated_params import _fast_rate_integrated_params


def filtering(observations: jnp.ndarray,
              x0: MVNStandard,
              slow_rate_params,
              l: int):
    def body(carry, y):
        xl_k_pred, xl_k_1 = carry
        h = _integrated_predict(slow_rate_params, xl_k_1)



    return


    _, xs = jax.lax.scan(body, (x0, x0), observations)
    xs = none_or_concat(xs, x0, 1)
    return xs, hs


def _integrated_predict(slow_rate_params, x):
    m, P = x
    A_bar, _, _, _, Bu_bar, Q_bar, _, _, _, _, _ = slow_rate_params

    m = A_bar @ m + Bu_bar
    P = A_bar @ P @ A_bar.T + Q_bar
    return MVNStandard(m, P)


def _integrated_update(slow_rate_params, xl_k_pred, xl_k_1, y):
    m_, P_ = xl_k_pred
    m_k_1, P_k_1 = xl_k_1
    A_bar, G_bar, B_bar, u_bar, Bu_bar, Q_bar, C_bar, M_bar, D_bar, Rx, Q = slow_rate_params
    S = C_bar @ P_k_1 @ C_bar.T + Rx
    temp = A_bar @ P_k_1 @ C_bar.T + jnp.sum(G_bar @ Q @ jnp.transpose(M_bar, axes=(0, 2, 1)), axis=0)
    L = jax.scipy.linalg.solve(S, temp.T).T
    m = m_ + L @ (y - C_bar @ m_k_1 - D_bar @ u_bar)
    P = P_ - L @ temp.T

    return MVNStandard(m, P)


