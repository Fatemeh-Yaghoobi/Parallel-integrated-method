import jax
import jax.numpy as jnp
from jax import lax, jit
from jax.lax import scan, associative_scan


A = jnp.array([[1, 2, 3],
               [4, 5, 6],
               [7, 8, 9]])

l = 3
nx = A.shape[0]
A_tensor = jnp.stack([jnp.eye(*A.shape)] + [A]*(l-1)) # [I, A, A, A,.., A]  (l times)
zeros_matrix = jnp.zeros((A.shape[0]* (l), A.shape[0]* (l))) # nl by nl times zeros column
res = associative_scan(lambda a, b: a@b, A_tensor, axis=0)  # I, A, A^2, ..., A^(l-1)



def full_matrix(mytensor, l, nx):
    output = zeros_matrix
    for j in range(l):
        for i in range(l-j):
            output = jax.lax.dynamic_update_slice(output, res[j], ((i+j)*nx, i*nx))

    return output

print(full_matrix(A_tensor, l, nx))


