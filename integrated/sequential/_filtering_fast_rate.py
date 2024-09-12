import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)
from integrated._base import MVNStandard
from integrated._utils import none_or_concat

# This code provides the filtering means and covariances for the fast rate states,
# but it does provide the filtering cross covariances between the states.


def filtering(y: jnp.ndarray,
              fast_params,
              xl_k_1):

    x_predict_interval = _integrated_predict(fast_params, xl_k_1)
    x_update_interval = _integrated_update(fast_params, x_predict_interval, xl_k_1, y)
    xs = none_or_concat(x_update_interval, xl_k_1, 1)

    return xs


def _integrated_predict(fast_params, x):
    m, P = x
    Abar, _, _, Bbar_u, Qbar, _, _, _, _, _, _ = fast_params

    m = Abar @ m + Bbar_u                                      # dim = (l - 1) x nx
    P = Abar @ P @ jnp.transpose(Abar, axes=(0, 2, 1)) + Qbar  # dim = (l - 1) x nx x nx
    return MVNStandard(m, P)


def _integrated_update(fast_params, x_predict_interval, xl_k_1, y):
    m_, P_ = x_predict_interval
    m_k_1, P_k_1 = xl_k_1
    Abar, Bbar, Gbar, Bbar_u, Qbar, C_bar, M_bar, D_bar, Rx, Q, Du_bar = fast_params

    S = C_bar @ P_k_1 @ C_bar.T + Rx                                                      # dim = ny x ny
    temp = (Abar @ P_k_1 @ C_bar.T
            + jnp.einsum('ijkl,jlm->ikm',
                         Gbar @ Q, jnp.transpose(M_bar, axes=(0, 2, 1))))                 # dim = (l - 1) x nx x ny


    vmap_func = jax.vmap(jax.scipy.linalg.solve, in_axes=(None, 0))
    TranL = vmap_func(S, jnp.transpose(temp, axes=(0, 2, 1)))
    L = jnp.transpose(TranL, axes=(0, 2, 1))                                                 # dim = (l - 1) x nx x ny

    m = m_ + jnp.einsum('ijk,k->ij', L, y -  C_bar @ m_k_1 - Du_bar)                         # dim = (l - 1) x nx
    tempT = jnp.transpose(temp, axes=(0, 2, 1))                                              # dim = (l - 1) x ny x nx
    P = P_ - jnp.einsum('ijk,ikl->ijl', L, tempT)                                            # dim = (l - 1) x nx x nx

    return MVNStandard(m, P)


