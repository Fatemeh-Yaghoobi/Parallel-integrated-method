import jax
import jax.numpy as jnp

from integrated._base import MVNStandard, LinearTran, LinearObs, SlowRateIntegratedParams
from integrated._utils import none_or_concat
from integrated.inegrated_params.bin._integrated_params import _fast_rate_integrated_params


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
            x = _fast_update(transition_model, observation_model, h, L, i, l)
            return x, x
        x_l, x = jax.lax.scan(fast_body, x, jnp.arange(l))

        return x_l, (x, h)

    _, (xs, hs) = jax.lax.scan(body, x0, observations)
    xs = none_or_concat(xs, x0, 1)
    return xs, hs


def _fast_update(transition_model, observation_model, h, L, i, l):
    m_h, P_h = h
    C, _ = observation_model
    ubar_f, Gbar_i_f, Bbar_i_f, At_i, c_i_f, G_bar_i, Q_f, Q_bar, Q = _fast_rate_integrated_params(transition_model, i, l)

    m_x_i = At_i @ m_h + c_i_f

    Cbar_i = At_i @ L @ C
    temp = jnp.transpose(G_bar_i, axes=(0, 2, 1)) @ Cbar_i.T
    Q_f_s = jnp.sum(Gbar_i_f @ Q @ temp, axis=0)
    P_x_i = (At_i @ P_h @ At_i.T - Q_f_s - Q_f_s.T
             + Cbar_i @ Q_bar @ At_i.T + At_i @ Q_bar @ Cbar_i.T
             + Q_f - At_i @ Q_bar @ At_i.T)
    return MVNStandard(m_x_i, P_x_i)


def _integrated_predict(slow_rate_params, x):
    m, P = x
    A_bar, _, _, _, Bu_bar, Q_bar = slow_rate_params

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


