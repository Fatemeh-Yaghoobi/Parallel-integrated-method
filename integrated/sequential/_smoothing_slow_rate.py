import jax
import jax.scipy.linalg as jlag
from jax.experimental.host_callback import id_print

from integrated._base import MVNStandard
from integrated._utils import none_or_shift, none_or_concat


def smoothing(filter_trajectory_sr: MVNStandard,
              filter_trajectory_sr_l: MVNStandard,
              transition_model,
              cross_covariances_sr):
    last_state = jax.tree_map(lambda z: z[-1], filter_trajectory_sr)

    def smooth_one(_transition_model, xf1, xfl, xs, _Pf_k1l):
        return _standard_smooth(_transition_model, xf1, xfl, xs, _Pf_k1l)

    def body(smoothed, inputs):
        filtered_1, filtered_l, Pf_k1l = inputs
        smoothed_state = smooth_one(transition_model, filtered_1, filtered_l, smoothed, Pf_k1l)

        return smoothed_state, smoothed_state

    _, smoothed_states = jax.lax.scan(body,
                                      last_state,
                                      (none_or_shift(filter_trajectory_sr, -1),
                                       none_or_shift(filter_trajectory_sr_l, -1),
                                       none_or_shift(cross_covariances_sr, -1)),
                                      reverse=True)

    smoothed_states = none_or_concat(smoothed_states, last_state, -1)
    return smoothed_states


def _standard_smooth(transition_model, xf1, xfl, xs, Pf_k1l):
    mf1, Pf1 = xf1
    mfl, Pfl = xfl
    ms, Ps = xs
    A, B, u, Q = transition_model
    Bu_bar = (B @ u).reshape(-1)

    mean_diff = ms - (Bu_bar + A @ mfl)
    S = A @ Pfl @ A.T + Q
    cov_diff = Ps - S

    gain = Pf_k1l @ jlag.solve(S, A, assume_a='pos').T
    ms = mf1 + gain @ mean_diff
    Ps = Pf1 + gain @ cov_diff @ gain.T

    return MVNStandard(ms, Ps)

