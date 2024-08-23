import jax
import jax.numpy as jnp

from integrated._base import MVNStandard


def smoothing(filter_trajectory: MVNStandard,
              all_params):
    last_interval = MVNStandard(filter_trajectory.mean[-1],
                             filter_trajectory.cov[-1])

    other_intervals = MVNStandard(filter_trajectory.mean[:-1],
                            filter_trajectory.cov[:-1])
    def smooth_one(_all_params, xf, xs):
        return _standard_smooth(_all_params, xf, xs)

    def body(smoothed, inputs):
        filtered = inputs
        smoothed_state = smooth_one(all_params, filtered, smoothed)

        return smoothed_state, smoothed_state

    _, smoothed_states = jax.lax.scan(body,
                                      last_interval,
                                      other_intervals,
                                      reverse=True)

    smoothed_states = jax.tree_map(lambda a, b: jnp.concatenate([b, a[None, ...]]),
                                   last_interval, smoothed_states)
    return smoothed_states


def _standard_smooth(all_params, Xf, Xs):
    Mf, Pf = Xf
    Ms, Ps = Xs
    mf_l, Pf_l = Xf.mean[-1], Xf.cov[-1]
    Abar, _, _, Bbar_u, Qbar, _, _, _, _, _, _ = all_params

    mean_diff = Ms - (Bbar_u + Abar @ mf_l)    # dim = l x nx
    S = Abar @ Pf_l @ jnp.transpose(Abar, axes=(0, 2, 1)) + Qbar
    cov_diff = Ps - S

    vmap_solve = jax.vmap(jax.scipy.linalg.solve, in_axes=(0, 0))
    tempT = vmap_solve(S, Abar)
    gain = Pf_l @ jnp.transpose(tempT, axes=(0, 2, 1))  # dim = l x nx x nx (should be)
    Ms = Mf + jnp.einsum('ijk,ik->ij', gain, mean_diff) # dim = l x nx
    Ps = Pf + gain @ cov_diff @ jnp.transpose(gain, axes=(0, 2, 1)) # dim = l x nx x nx

    return MVNStandard(Ms, Ps)

