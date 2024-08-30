import jax
import jax.scipy.linalg as jlag

from integrated._base import MVNStandard
from integrated._utils import none_or_shift, none_or_concat


def smoothing(filter_trajectory: MVNStandard,
              transition_model):
    last_state = jax.tree_map(lambda z: z[-1], filter_trajectory)

    def smooth_one(_slow_rate_params, xf, xs):
        return _standard_smooth(_slow_rate_params, xf, xs)

    def body(smoothed, inputs):
        filtered = inputs
        smoothed_state = smooth_one(transition_model, filtered, smoothed)

        return smoothed_state, smoothed_state

    _, smoothed_states = jax.lax.scan(body,
                                      last_state,
                                      none_or_shift(filter_trajectory, -1),
                                      reverse=True)

    smoothed_states = none_or_concat(smoothed_states, last_state, -1)
    return smoothed_states


def _standard_smooth(transition_model, xf, xs):
    mf, Pf = xf
    ms, Ps = xs
    A, B, u, Q = transition_model

    mean_diff = ms - (B @ u + A @ mf)
    S = A @ Pf @ A.T + Q
    cov_diff = Ps - S

    gain = Pf @ jlag.solve(S, A, assume_a='pos').T
    ms = mf + gain @ mean_diff
    Ps = Pf + gain @ cov_diff @ gain.T

    return MVNStandard(ms, Ps)

