import jax
import jax.numpy as jnp
import jax.scipy.linalg as jlinalg

from integrated._base import MVNStandard
from integrated._utils import none_or_concat
from integrated.parallel._operators import standard_smoothing_operator


def smoothing(filter_trajectory: MVNStandard,
              slow_rate_params):
    associative_params = _associative_params(slow_rate_params, filter_trajectory)
    smoothed_means, _, smoothed_covs = jax.lax.associative_scan(jax.vmap(standard_smoothing_operator),
                                                                associative_params, reverse=True)
    res = jax.vmap(MVNStandard)(smoothed_means, smoothed_covs)
    return res


def _associative_params(slow_rate_params, filtering_trajectory):
    ms, Ps = filtering_trajectory
    vmapped_fn = jax.vmap(_standard_associative_params, in_axes=[None, 0, 0])

    gs, Es, Ls = vmapped_fn(slow_rate_params, ms[:-1], Ps[:-1])
    g_T, E_T, L_T = ms[-1], jnp.zeros_like(Ps[-1]), Ps[-1]

    return none_or_concat((gs, Es, Ls), (g_T, E_T, L_T), -1)


def _standard_associative_params(slow_rate_params, m, P):
    A_bar, _, _, _, Bu_bar, Q_bar, _, _, _, _, _, _ = slow_rate_params
    Pp = A_bar @ P @ A_bar.T + Q_bar

    E = jlinalg.solve(Pp, A_bar @ P, assume_a='pos').T

    g = m - E @ (A_bar @ m + Bu_bar)
    L = P - E @ Pp @ E.T

    return g, E, L
