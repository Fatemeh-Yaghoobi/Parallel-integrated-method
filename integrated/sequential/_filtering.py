from typing import Optional

import jax
import jax.numpy as jnp
from jax.experimental.host_callback import id_print
from jax.lax import scan, associative_scan
from jax.scipy.linalg import cho_solve

from integrated._base import MVNStandard, LinearIntegrated
from integrated._utils import none_or_shift, none_or_concat

def integrated_parameters(A, b, Q, C, R, l):
    A_bar = A**l
    def body(_, j):
        i = l - j
        b_i = A**(i-1) @ b
        G_i = A**(i-1)
        A_tilde = A**(j + 1)
        A_new = A**(j)
        return _, (b_i, G_i, A_tilde, A_new)
    _, (b_bar, G_bar, A_tilde, A_new) = scan(body, None, jnp.arange(l))
    C_bar = (C / l) * jnp.sum(A_tilde, axis=0)
    M_bar = (C / l) * associative_scan(jnp.add, A_new) # reverse may be needed
    D_bar = M_bar @ b
    return A_bar, b_bar, G_bar, C_bar, A_tilde, A_new, M_bar, D_bar, Q, R


A = jnp.array([[2]])
b = jnp.array([5])
C = jnp.array([[1]])
l = 5
A_bar, b_bar, G_bar, C_bar, A_tilde, A_new, M_bar, D_bar, Q, R = integrated_parameters(A, b, None, C, None, l)
print(A_new, A_new.shape)
print(M_bar)

# def filtering(observations: jnp.ndarray,
#               x0: MVNStandard,
#               transition_model: LinearIntegrated,
#               observation_model: LinearIntegrated):
#
#     def body(x, y):
#         x = _integrated_predict(transition_model,  x)
#         x = _integrated_update(observation_model, transition_model.cov, x, y)
#         return x, x
#
#     _, xs = jax.lax.scan(body, x0, observations)
#     xs = none_or_concat(xs, x0, 1)
#     return xs
#
#
# def _integrated_predict(transition_model, x):
#     m, P = x
#     A, G, b, Q = transition_model
#
#     m = A @ m + b
#     P = A @ P @ A.T + G @ Q @ G.T
#
#     return MVNStandard(m, P)
#
#
# def _integrated_update(observation_model, Q, x, y):
#     m, P = x
#     C, M, d, R = observation_model
#
#     y_hat = C @ m + d
#     y_diff = y - y_hat
#     S = R + C @ P @ C.T + M @ Q @ M.T
#     K = jnp.scipy.linalg.solve(S, C @ P).T
#
#     m = m + G @ y_diff
#     P = P - G @ S @ G.T
#     return MVNStandard(m, P)
#
#
