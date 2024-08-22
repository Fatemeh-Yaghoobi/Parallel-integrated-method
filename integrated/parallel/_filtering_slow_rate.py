import jax
import jax.numpy as jnp

from integrated._base import MVNStandard
from integrated._utils import none_or_concat
from integrated.parallel._operators import filtering_operator


def filtering(observations: jnp.ndarray,
              x0: MVNStandard,
              slow_rate_params):
    m0, P0 = x0
    associative_params = _standard_associative_params(slow_rate_params, x0, observations)
    _, filtered_means, filtered_cov, _, _ = jax.lax.associative_scan(jax.vmap(filtering_operator), associative_params)

    filtered_means = none_or_concat(filtered_means, m0, position=1)
    filtered_cov = none_or_concat(filtered_cov, P0, position=1)

    res = jax.vmap(MVNStandard)(filtered_means, filtered_cov)

    return res


def _standard_associative_params(slow_rate_params, init_x, observations):
    T = observations.shape[0]

    def make_params(sl_params, x0, ys, i):
        predicate = i == 0

        def _first(_):
            return _standard_associative_params_first(sl_params, x0, ys)

        def _generic(_):
            return _standard_associative_params_generic(sl_params, ys)

        return jax.lax.cond(predicate, _first, _generic, None)

    vmapped_fn = jax.vmap(make_params, in_axes=[None, None, 0, 0])
    return vmapped_fn(slow_rate_params, init_x, observations, jnp.arange(T))


def _standard_associative_params_generic(slow_rate_params, y):
    A_bar, G_bar, B_bar, u_bar, Bu_bar, Q_bar, C_bar, M_bar, D_bar, Rx, Q, Du_bar = slow_rate_params

    tempt = jnp.sum(G_bar @ Q @ jnp.transpose(M_bar, axes=(0, 2, 1)), axis=0)
    K = jax.scipy.linalg.solve(Rx, tempt.T).T
    F = A_bar - K @ C_bar
    d = K @ y + Bu_bar - K @ Du_bar
    D = Q_bar - K @ tempt.T

    tempt2 = jax.scipy.linalg.solve(Rx, C_bar).T
    eta = tempt2 @ (y - Du_bar)
    J = tempt2 @ C_bar
    return F, d, D, eta, J


def _standard_associative_params_first(slow_rate_params, x0, y):
    A_bar, G_bar, B_bar, u_bar, Bu_bar, Q_bar, C_bar, M_bar, D_bar, Rx, Q, Du_bar = slow_rate_params
    m0, P0 = x0

    m_ = A_bar @ m0 + Bu_bar
    P_ = A_bar @ P0 @ A_bar.T + Q_bar

    S = C_bar @ P0 @ C_bar.T + Rx
    temp = A_bar @ P0 @ C_bar.T + jnp.sum(G_bar @ Q @ jnp.transpose(M_bar, axes=(0, 2, 1)), axis=0)
    L = jax.scipy.linalg.solve(S, temp.T).T

    F = jnp.zeros_like(P0)
    d = m_ + L @ (y - C_bar @ m0 - Du_bar)
    D = P_ - L @ temp.T

    tempt2 = jax.scipy.linalg.solve(Rx, C_bar).T

    eta = tempt2 @ (y - Du_bar)
    J = tempt2 @ C_bar
    return F, d, D, eta, J
