import jax
import jax.numpy as jnp
import jax.scipy.linalg as jlag

from integrated._base import MVNStandard


def smoothing(filter_trajectory_interval: MVNStandard,
              filter_trajectory_sr_l: MVNStandard,
              smoothed_trajectory_sr: MVNStandard,
              transition_model,
              cross_covariances_fr):
    Xf = filter_trajectory_interval
    xfl = filter_trajectory_sr_l
    xs = smoothed_trajectory_sr
    smoothed_states = _standard_smooth(transition_model, Xf, xfl, xs, cross_covariances_fr)

    return smoothed_states


def _standard_smooth(transition_model, Xf, xfl, xs, cross_covariances_fr):
    A, B, u, Q = transition_model

    Mf, Pf = Xf
    ms, Ps = xs
    mfl, Pfl = xfl
    Pf_il = cross_covariances_fr

    Bu_bar = (B @ u).reshape(-1)

    mean_diff = ms - (Bu_bar + A @ mfl)
    S = A @ Pfl @ A.T + Q
    cov_diff = Ps - S

    gain = Pf_il @ jlag.solve(S, A, assume_a='pos').T

    Ms = Mf + gain @ mean_diff
    Ps = Pf + gain @ cov_diff @ gain.T

    return MVNStandard(Ms, Ps)

