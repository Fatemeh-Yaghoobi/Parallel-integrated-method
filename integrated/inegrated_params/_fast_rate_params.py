import jax
import jax.numpy as jnp
from jax.lax import associative_scan

jax.config.update("jax_enable_x64", True)

from integrated.inegrated_params._slow_rate_params import _slow_rate_integrated_params



def _fast_params(transition_model, observation_model, l: int):
    A, B, u, Q = transition_model
    u_tensor = jnp.stack([u] * (l-1))                                             # [u, u, ..., u],               dim: (l-1) x nu x 1
    I_A_tensor = jnp.stack([jnp.eye(*A.shape)] + [A] * (l - 2))                   # [I, A, A, A,.., A],           dim: (l-1) x nx x nx
    IA_associative = associative_scan(lambda a, b: a @ b, I_A_tensor)             # [I, A, A^2, ..., A^(l-2)],    dim: (l-1) x nx x nx

    Abar = A @ IA_associative                                                     # [A, A^2, ..., A^(l-1)],       dim: (l-1) x nx x nx
    Gbar = jnp.zeros((l-1, l-1, *A.shape))
    Bbar = jnp.zeros((l-1, l-1, A.shape[0], u.shape[0]))
    for i in range(l-1):
        for j in range(l-1):
            if i >= j:
                Gbar = Gbar.at[i, j, :, :].set(IA_associative[i - j, :, :])          # dim: (l-1) x (l-1) x nx x nx
                Bbar = Bbar.at[i, j, :, :].set(IA_associative[i - j, :, :] @ B)      # dim: (l-1) x (l-1) x nx x nu
    Bbar_u= jnp.einsum('ijkl,jlm->ikm', Bbar, u_tensor).reshape(l - 1, -1)           # dim: (l-1) x nx
    GbarT = jnp.transpose(Gbar, axes=(0, 1, 3, 2))
    Qbar = jnp.einsum('ijkl,ijlm->ikm', Gbar @ Q, GbarT)                             # dim: (l-1) x nx x nx
    _, _, _, _, _, _, C_bar, M_bar, D_bar, Rx, Q, Du_bar = _slow_rate_integrated_params(transition_model, observation_model, l)
    M_bar_l_1 = M_bar[:-1]
    return Abar, Bbar, Gbar, Bbar_u, Qbar, C_bar, M_bar_l_1, D_bar, Rx, Q, Du_bar

