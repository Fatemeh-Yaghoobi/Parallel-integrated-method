import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)
from integrated._base import MVNStandard
from integrated._utils import none_or_concat


def filtering(observations: jnp.ndarray,
              x0: MVNStandard,
              all_params):

    def body(carry, y):
        xl_k_1 = carry
        x_predict_interval = _integrated_predict(all_params, xl_k_1)
        x_update_interval = _integrated_update(all_params, x_predict_interval, xl_k_1, y)
        xl_k = MVNStandard(x_update_interval.mean[-1], x_update_interval.cov[-1])
        return xl_k, x_update_interval

    _, xs = jax.lax.scan(body, x0, observations)
    # xs = none_or_concat(xs, x0, 1)
    return xs


def _integrated_predict(all_params, x):
    m, P = x
    Abar, _, _, Bbar_u, Qbar, _, _, _, _, _, _ = all_params

    m = Abar @ m + Bbar_u                                      # dim = l x nx
    P = Abar @ P @ jnp.transpose(Abar, axes=(0, 2, 1)) + Qbar  # dim = l x nx x nx
    return MVNStandard(m, P)


def _integrated_update(all_params, x_predict_interval, xl_k_1, y):
    m_, P_ = x_predict_interval
    m_k_1, P_k_1 = xl_k_1
    Abar, Bbar, Gbar, Bbar_u, Qbar, C_bar, M_bar, D_bar, Rx, Q, Du_bar = all_params

    S = C_bar @ P_k_1 @ C_bar.T + Rx                                                      # dim = ny x ny
    temp = (Abar @ P_k_1 @ C_bar.T
            + jnp.einsum('ijkl,jlm->ikm',
                         Gbar @ Q, jnp.transpose(M_bar, axes=(0, 2, 1))))                 # dim = l x nx x ny


    vmap_func = jax.vmap(jax.scipy.linalg.solve, in_axes=(None, 0))
    TranL = vmap_func(S, jnp.transpose(temp, axes=(0, 2, 1)))
    L = jnp.transpose(TranL, axes=(0, 2, 1))                                                 # dim = l x nx x ny

    m = m_ + jnp.einsum('ijk,k->ij', L, y -  C_bar @ m_k_1 - Du_bar)                         # dim = l x nx
    tempT = jnp.transpose(temp, axes=(0, 2, 1))                                              # dim = l x ny x nx
    P = P_ - jnp.einsum('ijk,ikl->ijl', L, tempT)                                            # dim = l x nx x nx

    return MVNStandard(m, P)


