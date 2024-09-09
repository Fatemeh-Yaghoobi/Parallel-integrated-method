import jax.numpy as jnp
from jax.lax import associative_scan
import jax

jax.config.update('jax_platform_name', 'cpu')
jax.config.update("jax_enable_x64", True)

from integrated._base import MVNStandard

def _full_transition_params(transition_model, l: int):
    A, B, u, Q = transition_model
    nx = A.shape[0]
    nu = B.shape[1]
    A_I_tensor = jnp.stack([A] * (l - 1) + [jnp.eye(*A.shape)])                      # [A, A, A,.., A, I],           dim: l x nx x nx
    AI_associative = associative_scan(lambda a, b: a @ b, A_I_tensor, reverse=True)  # [A^(l-1), ..., A^2, A, I],    dim: l x nx x nx
    Bbar = AI_associative @ B
    Gbar = AI_associative
    Bbar_matrix = jnp.transpose((jnp.transpose(Bbar, (0, 2, 1))).reshape(l * nu, nx))   # dim: nx x (l * nu)
    Gbar_matrix = jnp.transpose((jnp.transpose(Gbar, (0, 2, 1))).reshape(l * nx, nx))   # dim: nx x (l * nx)
    Ahat = jnp.zeros((l * nx, l * nx))
    Bhat = jnp.zeros((l * nx, (2 * l - 1) * nu))
    Ghat = jnp.zeros((l * nx, (2 * l - 1) * nx))
    Qtilda = jax.scipy.linalg.block_diag(*( [Q] * (2*l - 1) ))
    uhat = jnp.array(jnp.stack([u] * (2 * l - 1))).reshape(-1, nu)
    for i in range(l):
        Ahat = Ahat.at[i * nx: (i + 1) * nx, i * nx: (i + 1) * nx].set(jax.numpy.linalg.matrix_power(A, l))
        Bhat = Bhat.at[i * nx: (i + 1) * nx, i * nu: (l + i) * nu].set(Bbar_matrix)
        Ghat = Ghat.at[i * nx: (i + 1) * nx, i * nx: (l + i) * nx].set(Gbar_matrix)
    Bhat_u = (Bhat @ uhat).reshape(-1,)
    Qhat = Ghat @ Qtilda @ Ghat.T
    return Ahat, Bhat_u, Qhat
