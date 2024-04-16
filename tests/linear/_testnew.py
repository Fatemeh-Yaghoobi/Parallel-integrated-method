import jax.numpy as jnp
from jax import lax, jit


@jit
def lower_block_triangular_matrix(A, n):
    """
    Create a lower block triangular matrix with powers of A.

    Args:
    - A: Input matrix A
    - n: Size of the resulting matrix

    Returns:
    - Lower block triangular matrix
    """
    I = jnp.eye(A.shape[0])
    lower_blocks = [jnp.linalg.matrix_power(A, i) for i in range(1, n)]
    # lower_triangular_matrix = lax.pad(
    #     jnp.vstack([I] + lower_blocks),
    #     ((0, 0), (0, 0)),
    #     constant_values=0
    # )
    return lower_blocks


# Example usage
A = jnp.array([[1, 2, 3],
               [4, 5, 6],
               [7, 8, 9]])

n = 4  # Size of the resulting matrix
result_matrix = lower_block_triangular_matrix(A, n)
print(result_matrix)