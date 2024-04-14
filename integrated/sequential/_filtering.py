from typing import Optional

import jax
import jax.numpy as jnp
from jax.experimental.host_callback import id_print
from jax.lax import scan, associative_scan
from jax.scipy.linalg import cho_solve

from integrated._base import MVNStandard, LinearIntegrated
from integrated._utils import none_or_shift, none_or_concat

def integrated_parameters(A, l):
    def body(_, j):
        i = l - j
        A_vec = A ** (i - 1)
        return _, A_vec
    _, A_vec = scan(body, None, jnp.arange(l))
    A_bar = (1 / l) * jnp.sum(A @ A_vec, axis=0)
    B_bar = (1 / l) * associative_scan(jnp.add, A_vec, reverse=True)
    return A_bar, B_bar


# A = jnp.array([[2]])
# l = 3
# A_bar, B_bar= integrated_parameters(A, l)
# print(A_bar.shape, B_bar.shape)

def filtering(observations: jnp.ndarray,
              x0: MVNStandard,
              transition_model: LinearIntegrated,
              observation_model,
              l: int):

    def body(x, y):
        h = _integrated_predict(transition_model,  x)
        h = _integrated_update(observation_model, h, y)

        def fast_body(_, i):
            x = _fast_update(transition_model, observation_model, h, i)
            return x, x
        x_l, x = jax.lax.scan(fast_body, x, jnp.arange(l))

        return x_l, (x, h)

    _, (xs, hs) = jax.lax.scan(body, x0, observations)
    xs = none_or_concat(xs, x0, 1)
    return xs, hs


def _fast_update(transition_model, observation_model, h, j):
    i = j + 1
    A, A_bar, B_bar, b_bar, Q = transition_model
    C, R = observation_model
    def B_i_f(A, i):
        def body(_, k):
            B_i_f_vec = A ** (i - k - 1)
            return _, B_i_f_vec
        _, B_bar_i_f = scan(body, None, jnp.arange(i))
        return B_bar_i_f

    B_bar_i_f = B_i_f(A, i)
    m_h, P_h = h
    A_tilda_i = A**i @ jnp.linalg.inv(A_bar)
    c_i = B_bar_i_f @ ...  - A_tilda_i @ B_bar @ b_bar # check
    m_x = A_tilda_i @ m_h + c_i
    P_x =
    return MVNStandard(m_x, P_x)


def _integrated_predict(transition_model, x):
    m, P = x
    A, A_bar, B_bar, b_bar, Q = transition_model

    m = A_bar @ m + B_bar @ b_bar # tensor dot attention
    P = A_bar @ P @ A_bar.T + jnp.sum(B_bar @ Q @ B_bar.T, axis=0) # check this line

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
    return MVNStandard(m, P)


