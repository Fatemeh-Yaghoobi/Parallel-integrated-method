from typing import Optional

import jax
import jax.numpy as jnp
from jax.experimental.host_callback import id_print
from jax.lax import scan, associative_scan
from jax.scipy.linalg import cho_solve

from integrated._base import MVNStandard, LinearTran, LinearObs, SlowRateIntegratedParams
from integrated._utils import none_or_shift, none_or_concat


def filtering(observations: jnp.ndarray,
              x0: MVNStandard,
              transition_model: LinearTran,
              observation_model: LinearObs,
              slow_rate_params: SlowRateIntegratedParams,
              l: int):

    def body(x, y):
        h = _integrated_predict(slow_rate_params,  x)
        h, L = _integrated_update(observation_model, h, y)

        def fast_body(_, i):
            x = _fast_update(transition_model, observation_model, h, L, i)
            return x, x
        x_l, x = jax.lax.scan(fast_body, x, jnp.arange(l))

        return x_l, (x, h)

    _, (xs, hs) = jax.lax.scan(body, x0, observations)
    xs = none_or_concat(xs, x0, 1)
    return xs, hs


def _fast_update(transition_model, observation_model, h, L, i):
    i = j + 1
    m_h, P_h = h
    nx = m_h.shape[0]
    A, Bi_vec, A_bar, B_bar, b_bar, Q, Bb_bar, Q_bar = transition_model
    C, R = observation_model

    # Cbar_i =  ..

    # B_bar_i_f = jax.lax.dynamic_slice(Bi_vec, [0, 0, 0], [i, nx, nx])
    # b_bar_f = jax.lax.dynamic_slice(b_bar, [0], [i])
    # A_tilda_i = A**i @ jnp.linalg.inv(A_bar)
    # c_i = B_bar_i_f @ b_bar_f - A_tilda_i @ B_bar @ b_bar # check
    #
    # m_x = A_tilda_i @ m_h + c_i
    return MVNStandard(m_h, P_h)  # check this line not correct



# def _fast_update(transition_model, observation_model, h, j):
#     i = j + 1
#     m_h, P_h = h
#     nx = m_h.shape[0]
#     A, Bi_vec, A_bar, B_bar, b_bar, Q = transition_model
#     C, R = observation_model
#
#     B_bar_i_f = jax.lax.dynamic_slice(Bi_vec, [0, 0, 0], [i, nx, nx])
#     b_bar_f = jax.lax.dynamic_slice(b_bar, [0], [i])
#     A_tilda_i = A**i @ jnp.linalg.inv(A_bar)
#     c_i = B_bar_i_f @ b_bar_f - A_tilda_i @ B_bar @ b_bar # check
#
#     m_x = A_tilda_i @ m_h + c_i
#     # P_x = A_tilda_i @ P_h @ A_tilda_i.T + jnp.sum(B_bar_i_f @ Q @ B_bar_i_f.T, axis=0) # check this line not complete
#     return MVNStandard(m_h, P_h)  # check this line not correct


def _integrated_predict(slow_rate_params, x):
    m, P = x
    A_bar, _, _, G_bar, Bu_bar, Q_bar = slow_rate_params

    m = A_bar @ m + Bu_bar
    P = A_bar @ P @ A_bar.T + Q_bar

    return MVNStandard(m, P)


def _integrated_update(observation_model, h, y):
    m, P = h
    C, R = observation_model

    y_hat = C @ m
    y_diff = y - y_hat
    S = R + C @ P @ C.T
    L = jax.scipy.linalg.solve(S, C @ P).T

    m = m + L @ y_diff
    P = P - L @ C @ P
    return MVNStandard(m, P), L


