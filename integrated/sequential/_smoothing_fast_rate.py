import jax
import jax.numpy as jnp

from integrated._base import MVNStandard


def smoothing(filter_trajectory: MVNStandard,
              x_fast_rate_interval: MVNStandard,
              fast_rate_params,
              x_smoothed_slow_rate: MVNStandard):
    xs = x_smoothed_slow_rate                                     # dim = (nx,),  nx x nx
    flip_x_mean = jnp.flip(x_fast_rate_interval.mean, axis=0)     # dim = (l - 1) x nx,          index: k, l-1:1
    flip_x_cov = jnp.flip(x_fast_rate_interval.cov, axis=0)       # dim = (l - 1) x nx x nx,     index: k, l-1:1
    xf = MVNStandard(flip_x_mean, flip_x_cov)

    smoothed_states_ = _standard_smooth(fast_rate_params, xf, xs)

    smoothed_states = jax.tree_map(lambda a, b: jnp.concatenate([b, a[None, ...]]),
                                   xs, smoothed_states_)
    return smoothed_states


def _standard_smooth(fast_rate_params, Xf, xs):
    Abar, Bbar, Gbar, Bbar_u, Qbar, C_bar, M_bar, D_bar, Rx, Q, Du_bar = fast_rate_params

    # note: the assumption is that all the inputs are the same.

    Mf, Pf = Xf
    ms, Ps = xs

    vmap_solve = jax.vmap(jax.scipy.linalg.solve, in_axes=(0, 0))
    S = Abar @ Pf @ jnp.transpose(Abar, axes=(0, 2, 1)) + Qbar
    tempT = vmap_solve(S, Abar)
    gain = Pf @ jnp.transpose(tempT, axes=(0, 2, 1))                 # dim = (l - 1) x nx x nx

    mean_diff = ms - (Bbar_u + Abar @ Mf)                            # dim = (l - 1) x nx
    cov_diff = Ps - S                                                # dim = (l - 1) x nx x nx

    Ms = Mf + jnp.einsum('ijk,ik->ij', gain, mean_diff)              # dim = (l - 1) x nx
    Ps = Pf + gain @ cov_diff @ jnp.transpose(gain, axes=(0, 2, 1))  # dim = (l - 1) x nx x nx

    return MVNStandard(jnp.flip(Ms, axis=0), jnp.flip(Ps, axis=0))

