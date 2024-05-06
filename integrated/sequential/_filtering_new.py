import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)
from integrated._base import MVNStandard


def filtering(observations: jnp.ndarray,
              x0: MVNStandard,
              all_params):

    def body(carry, y):
        xl_k_1 = carry
        x_predict_interval = _integrated_predict(all_params, xl_k_1)
        x_update_interval = _integrated_update(all_params, x_predict_interval, y)
        xl_k = MVNStandard(x_update_interval.mean[-1], x_update_interval.cov[-1])
        return xl_k, x_update_interval

    _, xs = jax.lax.scan(body, x0, observations)
    return xs


def _integrated_predict(all_params, x):
    m, P = x
    Abar, _, _, Bbar_u, Qbar, _, _ = all_params

    m = Abar @ m + Bbar_u                                      # dim = l x nx
    P = Abar @ P @ jnp.transpose(Abar, axes=(0, 2, 1)) + Qbar  # dim = l x nx x nx
    return MVNStandard(m, P)


def _integrated_update(all_params, x_predict_interval, y):
    m_, P_ = x_predict_interval
    Abar, Bbar, Gbar, Bbar_u, Qbar, H, R = all_params

    y_diff= y - jnp.einsum('ijk,ik->j', H, m_)
    S = H[0] @ jnp.sum(P_, axis=0) @ H[0].T + R
    temp = P_ @ H[0].T
    KT = jnp.linalg.solve(S, jnp.transpose(temp, axes=(0, 2, 1)))
    K = jnp.transpose(KT, axes=(0, 2, 1))

    m = m_ + jnp.einsum('ijk,k->ij', K, y_diff)
    P = P_ - K @ H @ P_

    return MVNStandard(m, P)


