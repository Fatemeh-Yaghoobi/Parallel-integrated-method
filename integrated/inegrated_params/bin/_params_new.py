import jax
import jax.numpy as jnp
from jax.lax import associative_scan

jax.config.update("jax_enable_x64", True)

from integrated._base import LinearTran, LinearObs



def _fast_and_slow_params(transition_model, observation_model, l: int):
    A, B, u, Q = transition_model
    u_tensor = jnp.stack([u] * l)                                                    # [u, u, ..., u],               dim: l x nu x 1
    I_A_tensor = jnp.stack([jnp.eye(*A.shape)] + [A] * (l - 1))                      # [I, A, A, A,.., A],           dim: l x nx x nx
    IA_associative = associative_scan(lambda a, b: a @ b, I_A_tensor)                # [I, A, A^2, ..., A^(l-1)],    dim: l x nx x nx

    Abar = A @ IA_associative                                                        # [A, A^2, ..., A^l],           dim: l x nx x nx
    Gbar = jnp.zeros((l, l, *A.shape))
    Bbar = jnp.zeros((l, l, A.shape[0], u.shape[0]))
    for i in range(l):
        for j in range(l):
            if i >= j:
                Gbar = Gbar.at[i, j, :, :].set(IA_associative[i - j, :, :])          # dim: l x l x nx x nx
                Bbar = Bbar.at[i, j, :, :].set(IA_associative[i - j, :, :] @ B)      # dim: l x l x nx x nu
    Bbar_u= jnp.einsum('ijkl,jlm->ikm', Bbar, u_tensor).reshape(l, -1)               # dim: l x nx
    GbarT = jnp.transpose(Gbar, axes=(0, 1, 3, 2))
    Qbar = jnp.einsum('ijkl,ijlm->ikm', Gbar @ Q, GbarT)                             # dim: l x nx x nx

    C, R = observation_model
    H = jnp.stack([C] * l) / l                                                       # [C, C, ..., C]/l,      dim: l x ny x nx
    return Abar, Bbar, Gbar, Bbar_u, Qbar, H, R

