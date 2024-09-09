import jax
import jax.scipy.linalg as jlag

from integrated._base import MVNStandard
from integrated._utils import none_or_shift, none_or_concat


def smoothing(batch_filtered_results: MVNStandard,
              full_transition_params):

    filtered_last_interval = jax.tree_map(lambda z: z[-1], batch_filtered_results)

    def smooth_one(_full_params, xf, xs):
        return _standard_smooth(_full_params, xf, xs)

    def body(smoothed, inputs):
        filtered = inputs
        smoothed_state = smooth_one(full_transition_params, filtered, smoothed)

        return smoothed_state, smoothed_state

    _, smoothed_states = jax.lax.scan(body,
                                      filtered_last_interval,
                                      none_or_shift(batch_filtered_results, -1),
                                      reverse=True)
    smoothed_states = none_or_concat(smoothed_states, filtered_last_interval, -1)
    return smoothed_states



def _standard_smooth(full_transition_params, xf, xs):
    a = xf
    mf, Pf = xf
    ms, Ps = xs
    Ahat, Bhat_u, Qhat = full_transition_params

    mean_diff = ms - (Bhat_u + Ahat @ mf)
    S = Ahat @ Pf @ Ahat.T + Qhat
    cov_diff = Ps - S

    gain = Pf @ jlag.solve(S, Ahat, assume_a='pos').T
    ms = mf + gain @ mean_diff
    Ps = Pf + gain @ cov_diff @ gain.T

    return MVNStandard(ms, Ps)

