from typing import Optional

import jax
import jax.numpy as jnp
from jax.scipy.linalg import cho_solve

from integrated._base import MVNStandard, LinearIntegrated
from integrated._utils import none_or_shift, none_or_concat


def filtering(observations: jnp.ndarray,
              x0: MVNStandard,
              transition_model: LinearIntegrated,
              observation_model: LinearIntegrated):

    def body(x, y):
        x = _integrated_predict(transition_model,  x)
        x = _integrated_update(observation_model, transition_model.cov, x, y)
        return x, x

    _, xs = jax.lax.scan(body, x0, observations)
    xs = none_or_concat(xs, x0, 1)
    return xs


def _integrated_predict(transition_model, x):
    m, P = x
    A, G, b, Q = transition_model

    m = A @ m + b
    P = A @ P @ A.T + G @ Q @ G.T

    return MVNStandard(m, P)


def _integrated_update(observation_model, Q, x, y):
    m, P = x
    C, M, d, R = observation_model

    y_hat = C @ m + d
    y_diff = y - y_hat
    S = R + C @ P @ C.T + M @ Q @ M.T
    K = jnp.scipy.linalg.solve(S, C @ P).T

    m = m + G @ y_diff
    P = P - G @ S @ G.T
    return MVNStandard(m, P)


