import jax
import jax.numpy as jnp
from jax.lax import associative_scan

jax.config.update("jax_enable_x64", True)

from integrated.inegrated_params._slow_rate_params import _slow_rate_integrated_params



def _full_filtering_params(transition_model, observation_model, l: int):
    A, B, u, Q = transition_model
    nx = A.shape[0]
    nu = B.shape[1]
    u_tensor = jnp.stack([u] * l)                                             # [u, u, ..., u],               dim: l x nu x 1
    I_A_tensor = jnp.stack([jnp.eye(*A.shape)] + [A] * (l - 1))                   # [I, A, A, A,.., A],           dim: l x nx x nx
    IA_associative = associative_scan(lambda a, b: a @ b, I_A_tensor)             # [I, A, A^2, ..., A^(l-1)],    dim: l x nx x nx

    Abar = A @ IA_associative                                                     # [A, A^2, ..., A^l],       dim: l x nx x nx
    Abar = Abar.reshape(l * nx, nx)                                          # [A, A^2, ..., A^(l-1)],   dim: (l-1) x nx x nx

    Gbar = jnp.zeros((l * nx, l * nx))
    Bbar = jnp.zeros((l * nx, l * nu))
    for i in range(l):
        for j in range(l):
            if i >= j:
                Gbar = Gbar.at[i * nx: (i + 1) * nx, j * nx: (j + 1) * nx].set(IA_associative[i - j, :, :])      # dim: lnx X lnx
                Bbar = Bbar.at[i * nx: (i + 1) * nx, j * nu: (j + 1) * nu].set(IA_associative[i - j, :, :] @ B)  # dim: lnx X lnu
    uhat = u_tensor.reshape(l * nu, 1)                                          # dim: lnu x 1
    Bbar_u= (Bbar @ uhat).reshape(-1,)                                          # dim: lnx,
    Qtilda = jax.scipy.linalg.block_diag(*([Q] * l))
    Qbar = Gbar @ Qtilda @ Gbar.T                            # dim: lnx x lnx
    C, R = observation_model
    for i in range(l):
        if i == 0:
            H = C / l
        else:
            H = jnp.concatenate([H, C / l], axis=1)
    return Abar, Gbar, Bbar, Bbar_u, Qbar, H, R

