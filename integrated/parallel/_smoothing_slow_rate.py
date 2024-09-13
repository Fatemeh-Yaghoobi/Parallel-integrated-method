import jax
import jax.numpy as jnp
import jax.scipy.linalg as jlinalg
from jax.experimental.host_callback import id_print

from integrated._base import MVNStandard
from integrated._utils import none_or_concat
from integrated.parallel._operators import standard_smoothing_operator


def smoothing(transition_model,
              filter_trajectory: MVNStandard,
              filter_trajectory_sr_l: MVNStandard,
              cross_covs_f_1_to_l):
    # transition_model: (A, B, u, Q)
    # filter_trajectory: (lnx), (lnx, lnx)            --> should be given from 1 to N
    # cross_covs_f_1_to_l: (lnx, nx)                  --> should be given from 1 to N - 1
    # filter_trajectory_sr_l: (nx), (nx, nx)          --> should be given from 1 to N - 1

    associative_params = _associative_params(transition_model, filter_trajectory, filter_trajectory_sr_l, cross_covs_f_1_to_l)
    smoothed_means, _, smoothed_covs = jax.lax.associative_scan(jax.vmap(standard_smoothing_operator),
                                                                associative_params, reverse=True)
    res = jax.vmap(MVNStandard)(smoothed_means, smoothed_covs)
    return res


def _associative_params(tran_model, filtering_trajectory, last_filtered_state, cross_covs_f_1_to_l):
    A, _, _, _ = tran_model
    nx = A.shape[0]

    ms, Ps = filtering_trajectory
    m_kl, P_kl = last_filtered_state
    lnx = ms.shape[-1]
    vmapped_fn = jax.vmap(_standard_associative_params, in_axes=[None, 0, 0, 0, 0 ,0])
    gs, Es, Ls = vmapped_fn(tran_model, ms[:-1, :], Ps[:-1, :, :], m_kl, P_kl, cross_covs_f_1_to_l)
    g_T, E_T, L_T = ms[-1, :], jnp.zeros((lnx, nx)), Ps[-1, :, :]
    return none_or_concat((gs, Es, Ls), (g_T, E_T, L_T), -1)


def _standard_associative_params(tran_model, m_k_f, P_k_f, m_kl, P_kl, Pk_1_to_l):
    A, B, u, Q = tran_model
    Pp = A @ P_kl @ A.T + Q

    Bu_bar = (B @ u).reshape(-1)
    E = Pk_1_to_l @ jlinalg.solve(Pp, A, assume_a='pos').T

    g = m_k_f - E @ (A @ m_kl + Bu_bar)
    L = P_k_f - E @ Pp @ E.T

    return g, E, L
