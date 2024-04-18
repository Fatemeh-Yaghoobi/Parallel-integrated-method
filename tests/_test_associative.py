import jax.numpy as jnp
from jax import lax, jit
from jax.lax import scan, associative_scan


A = jnp.array([[1, 2, 3],
               [4, 5, 6],
               [7, 8, 9]])

l = 4
A_tensor = jnp.array([A]*l)
res = associative_scan(lambda a, b: a@b, A_tensor, axis=0)

