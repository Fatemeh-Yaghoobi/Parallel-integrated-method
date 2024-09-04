import jax
import jax.scipy.linalg as jlag

from integrated._base import MVNStandard
from integrated._utils import none_or_shift, none_or_concat


def smoothing(filtered_last_interval: MVNStandard,
              full_transition_params,
              filter_trajectory_sr):

    def smooth_one(_full_params, xf, xs):
        return _standard_smooth(_full_params, xf, xs)

    def body(smoothed, inputs):
        filtered = inputs
        smoothed_state = smooth_one(full_transition_params, filtered, smoothed)

        return smoothed_state, smoothed_state

    _, smoothed_states = jax.lax.scan(body,
                                      filtered_last_interval,
                                      none_or_shift(filter_trajectory_sr, -1),
                                      reverse=True)
    last_state = jax.tree_map(lambda z: z[-1], filter_trajectory_sr)
    smoothed_states = none_or_concat(smoothed_states, last_state, -1)
    return smoothed_states


def _standard_smooth(slow_rate_params, xf, xs):
    mf, Pf = xf
    ms, Ps = xs
    A_bar, _, _, _, Bu_bar, Q_bar, _, _, _, _, _, _ = slow_rate_params

    mean_diff = ms - (Bu_bar + A_bar @ mf)
    S = A_bar @ Pf @ A_bar.T + Q_bar
    cov_diff = Ps - S

    gain = Pf @ jlag.solve(S, A_bar, assume_a='pos').T
    ms = mf + gain @ mean_diff
    Ps = Pf + gain @ cov_diff @ gain.T

    return MVNStandard(ms, Ps)

