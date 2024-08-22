import jax
import jax.numpy as jnp
from jax.lax import associative_scan

jax.config.update("jax_enable_x64", True)


def _fast_rate_params(transition_model, l: int):
    '''
    Outputs are parameters in Equations 10

    '''
    A, B, u, Q = transition_model
    u_tensor = jnp.stack([u] * (l-1))                                                # [u, u, ..., u],               dim: (l-1) x nu x 1
    I_A_tensor = jnp.stack([jnp.eye(*A.shape)] + [A] * (l - 2))                      # [I, A, A, A,.., A],           dim: (l-1) x nx x nx
    IA_associative = associative_scan(lambda a, b: a @ b, I_A_tensor)                # [I, A, A^2, ..., A^(l-2)],    dim: (l-1) x nx x nx

    Abar = A @ IA_associative                                                        # [A, A^2, ..., A^(l-1)],       dim: (l-1) x nx x nx
    Gbar = jnp.zeros((l-1, l-1, *A.shape))
    Bbar = jnp.zeros((l-1, l-1, A.shape[0], u.shape[0]))
    for i in range(l-1):
        for j in range(l-1):
            if i >= j:
                Gbar = Gbar.at[i, j, :, :].set(IA_associative[i - j, :, :])          # dim: (l-1) x (l-1) x nx x nx
                Bbar = Bbar.at[i, j, :, :].set(IA_associative[i - j, :, :] @ B)      # dim: (l-1) x (l-1) x nx x nu
    Bbar_u= jnp.einsum('ijkl,jlm->ikm', Bbar, u_tensor).reshape(l-1, -1)             # dim: (l-1) x nx
    GbarT = jnp.transpose(Gbar, axes=(0, 1, 3, 2))
    Qbar = jnp.einsum('ijkl,ijlm->ikm', Gbar @ Q, GbarT)                             # dim: (l-1) x nx x nx
    return Abar, Bbar, Gbar, Bbar_u, Qbar

