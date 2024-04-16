from typing import Optional

import jax
import jax.numpy as jnp
from jax.experimental.host_callback import id_print
from jax.lax import scan, associative_scan
from jax.scipy.linalg import cho_solve

from integrated._base import MVNStandard, LinearIntegrated, LinearIntegratedObs
from integrated._utils import none_or_shift, none_or_concat

def integrated_parameters(A, l):
    def body(_, j):
        i = l - j
        A_vec = A ** (i - 1)
        Bi_vec = A ** j
        return _, (A_vec, Bi_vec)
    _, (A_vec, Bi_vec) = scan(body, None, jnp.arange(l))
    A_bar = (1 / l) * jnp.sum(A @ A_vec, axis=0)
    B_bar = (1 / l) * associative_scan(jnp.add, A_vec, reverse=True)
    return A_vec, Bi_vec


A = jnp.array([[2]])
l = 2
c = jnp.array([1, 2])
Q = jnp.eye(1)
h = MVNStandard(jnp.array([3]), jnp.eye(1))

def fast_state(A, c, Q, h, i):
    def fast_body(_, k):
        y = A**k @ Q @ (A**k).T
        return _, y
    _, ys = jax.lax.scan(fast_body, None, jnp.arange(i))
    Pxi = jnp.sum(ys, axis=0)
    mxi = A**i @ h.mean @ (A**i).T + c[i-1]
    return MVNStandard(mxi, Pxi)

# print(Q + A @ Q @ A.T, A**2@h.mean@(A**2).T + c[1])
# i = 2
# res = fast_state(A, c, Q, h, i)
# print(res)
#
# vmap_func = jax.vmap(fast_state, in_axes=(None, None, None, None, 0))
# i = jnp.array([1, 2])
# res_vmap = vmap_func(A, c, Q, h, i)
# print(res_vmap)


import jax.numpy as jnp
from jax import lax


def lower_block_triangular_matrix(A, n):
    def body(carry, _):
        carry = A @ carry
        return carry, carry
    _, lower_blocks = jax.lax.scan(body, A, jnp.arange(n))
    return lower_blocks


# Example usage
A = jnp.array([[1, 2, 3],
               [4, 5, 6],
               [7, 8, 9]])

n = 4  # Size of the resulting matrix
result_matrix = lower_block_triangular_matrix(A, n)
# print(result_matrix)

func = lambda j: jnp.linalg.matrix_power(A, j)
result = jax.vmap(func)(jnp.arange(1, 1))
print(result)
