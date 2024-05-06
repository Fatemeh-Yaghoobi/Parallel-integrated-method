import jax
import jax.numpy as jnp
import jax.scipy.linalg as jlinalg
import numpy as np
from jax.experimental.host_callback import id_print

jax.config.update("jax_enable_x64", True)

def filtering_operator(elem1, elem2):
    """
    Associative operator described in https://ieeexplore.ieee.org/document/9013038 Lemma 8.

    Parameters
    ----------
    elem1: tuple of array
        A_i, b_i, C_i, eta_i, J_i
    elem2: tuple of array
        A_j, b_j, C_j, eta_j, J_j

    Returns
    -------
    elem12: tuple of array
        ...
    """
    A1, b1, C1, eta1, J1 = elem1                # A1: l x nx x nx, b1: l x nx, C1: l x nx x nx
    A2, b2, C2, eta2, J2 = elem2
    dim = b1.shape[1]

    I_dim = jnp.eye(dim)
    IpCJ = I_dim + C1 @ J2  # dim = l x nx x nx
    IpJC = I_dim + J2 @ C1   # dim = l x nx x nx

    vmap_solve = jax.vmap(jax.scipy.linalg.solve, in_axes=(0, 0))

    AIpCJ_invT = vmap_solve(jnp.transpose(IpCJ, axes=(0, 2, 1)), jnp.transpose(A2, axes=(0, 2, 1)))
    AIpCJ_inv = jnp.transpose(AIpCJ_invT, axes=(0, 2, 1))                  # dim = l x nx x nx

    AIpJC_invT = vmap_solve(jnp.transpose(IpJC, axes=(0, 2, 1)), A1)       # dim = l x nx x nx
    AIpJC_inv = jnp.transpose(AIpJC_invT, axes=(0, 2, 1))                  # dim = l x nx x nx

    A = AIpCJ_inv @ A1                                                  # dim = l x nx x nx
    b = jnp.einsum('ijk,ik->ij', AIpCJ_inv, b1 + jnp.einsum('ijk,ik->ij', C1, eta2)) + b2
    C = AIpCJ_inv @ (C1 @ jnp.transpose(A2, axes=(0, 2, 1))) + C2       # dim = l x nx x nx

    temp = jnp.einsum('ijk,ik->ij', J2, b1)                                 # dim = l x nx
    eta = jnp.einsum('ijk,ik->ij', AIpJC_inv, eta2 - temp) + eta1           # dim = l x nx
    J = AIpJC_inv @ (J2 @ A1) + J1                                          # dim = l x nx x nx
    return A, b, C, eta, J











# def filtering_operator(elem1, elem2):
#     """
#     Associative operator described in https://ieeexplore.ieee.org/document/9013038 Lemma 8.
#
#     Parameters
#     ----------
#     elem1: tuple of array
#         A_i, b_i, C_i, eta_i, J_i
#     elem2: tuple of array
#         A_j, b_j, C_j, eta_j, J_j
#
#     Returns
#     -------
#     elem12: tuple of array
#         ...
#     """
#     A1, b1, C1, eta1, J1 = elem1                # A1: l x nx x nx, b1: l x nx, C1: l x nx x nx
#     A2, b2, C2, eta2, J2 = elem2
#     dim = b1.shape[1]
#
#     I_dim = jnp.eye(dim)
#     IpCJ = I_dim + jnp.einsum('ijk,km->ijm', C1, J2)   # dim = l x nx x nx
#     IpJC = I_dim + jnp.einsum('ij,kjm->kim', J2, C1)   # dim = l x nx x nx
#
#     vmap_solve = jax.vmap(jax.scipy.linalg.solve, in_axes=(0, 0))
#
#     AIpCJ_invT = vmap_solve(jnp.transpose(IpCJ, axes=(0, 2, 1)), jnp.transpose(A2, axes=(0, 2, 1)))
#     AIpCJ_inv = jnp.transpose(AIpCJ_invT, axes=(0, 2, 1))                  # dim = l x nx x nx
#
#     AIpJC_invT = vmap_solve(jnp.transpose(IpJC, axes=(0, 2, 1)), A1)       # dim = l x nx x nx
#     AIpJC_inv = jnp.transpose(AIpJC_invT, axes=(0, 2, 1))                  # dim = l x nx x nx
#
#     A = AIpCJ_inv @ A1                                                  # dim = l x nx x nx
#     b = jnp.einsum('ijk,ik->ij', AIpCJ_inv, b1 + jnp.einsum('ijk,k->ij', C1, eta2)) + b2
#     C = AIpCJ_inv @ (C1 @ jnp.transpose(A2, axes=(0, 2, 1))) + C2       # dim = l x nx x nx
#
#     temp = jnp.einsum('ij,kj->ki', J2, b1)                              # dim = l x nx
#     eta = jnp.einsum('ijk,ik->ij', AIpJC_inv, eta2 - temp) + eta1       # dim = l x nx
#     J = AIpJC_inv @ (J2 @ A1) + J1                                      # dim = l x nx x nx
#     return A, b, C, eta[-1], J[-1]