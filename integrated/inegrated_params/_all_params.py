import jax
import jax.numpy as jnp
import numpy as np
from jax.lax import associative_scan

jax.config.update("jax_enable_x64", True)

from integrated._base import LinearTran, LinearObs
from integrated.inegrated_params._slow_rate_params import _slow_rate_integrated_params



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
    AbarT = jnp.transpose(Abar, axes=(0, 2, 1))
    _, _, _, _, _, _, C_bar, M_bar, D_bar, Rx, Q, Du_bar = _slow_rate_integrated_params(transition_model, observation_model, l)
    # a = jnp.array([1, 2, 3, 4])
    # m_k_1 = jnp.stack([a] * l)
    # Cm = jnp.einsum('ij,lj->li', C_bar, m_k_1)  # dim = l x ny
    # np.testing.assert_allclose(Cm[1], C_bar @ m_k_1[1])

    # S = C_bar @ Q @ C_bar.T + Rx
    # temp =jnp.einsum('ijkl,jlm->ikm', Gbar @ Q, jnp.transpose(M_bar, axes=(0, 2, 1)))
    # print(temp.shape)
    # out1 = Gbar[-1] @ Q
    # out2 = A @ A @ Q @ M_bar[0].T + A@Q @ M_bar[1].T + Q @ M_bar[2].T
    # np.testing.assert_allclose(temp[2], out2)
    # invT = jax.scipy.linalg.solve(S, jnp.transpose(temp, axes=(0, 2, 1)))
    # vmap_func = jax.vmap(jax.scipy.linalg.solve, in_axes=(None, 0))
    # invT = vmap_func(S, jnp.transpose(temp, axes=(0, 2, 1)))
    # inv = jnp.transpose(invT, axes=(0, 2, 1))
    return Abar, Bbar, Gbar, Bbar_u, Qbar, C_bar, M_bar, D_bar, Rx, Q, Du_bar


A = jnp.array([[0.8499, 0.0350, 0.0240, 0.0431],
               [1.2081, 0.0738, 0.0763, 0.4087],
               [0.7331, 0.0674, 0.0878, 0.8767],
               [0.0172, 0.0047, 0.0114, 0.9123]])
B = jnp.array([[0, 0, 0, 1]]).T
u = jnp.array([[1]])
Q = 0.5 * jnp.eye(4)
transition_model = LinearTran(A, B, u, Q)
C = jnp.array([[1, 0, 0, 0],
               [0, 0, 0, 1]])
cov = jnp.eye(2)
observation_model = LinearObs(C, cov)
l = 3
Abar, Bbar, Gbar, Bbar_u, Qbar , C_bar, M_bar, D_bar, Rx, Q, Du_bar = _fast_and_slow_params(transition_model, observation_model, l)

