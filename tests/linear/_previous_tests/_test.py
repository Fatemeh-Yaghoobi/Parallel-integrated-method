from typing import Optional

import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental.host_callback import id_print
from jax.lax import scan, associative_scan
from jax.scipy.linalg import cho_solve
from jax.scipy.linalg import solve
jax.config.update("jax_enable_x64", True)





l = 20
nx = 4
D = jnp.zeros((l, nx, nx)) + 2
D = D.at[0].set(jnp.eye(nx))
D = D.at[0, 0, :].set(jnp.array([1, 2, 3, 4]))
J = jnp.eye(nx) * 3
J = J.at[0, :].set(jnp.array([1, 2, 3, 4]))
J2 = jnp.stack([J] * l)

DJ = jnp.einsum('ijk,km->ijm', D, J) + jnp.eye(nx)
DJ2 = jnp.einsum('ijk,ikm->ijm', D, J2) + jnp.eye(nx)
np.testing.assert_array_equal(DJ, DJ2)
np.testing.assert_array_equal(DJ[0], D[0]@J + jnp.eye(nx))
np.testing.assert_array_equal(DJ[1], D[1]@J + jnp.eye(nx))
####
JD = jnp.einsum('ij,kjm->kim', J, D) + jnp.eye(nx)
J2D = jnp.einsum('kij,kjm->kim', J2, D) + jnp.eye(nx)
np.testing.assert_array_equal(JD, J2D)
np.testing.assert_array_equal(JD[0], J@D[0] + jnp.eye(nx))
np.testing.assert_array_equal(JD[1], J@D[1] + jnp.eye(nx))
#### solve
vmap_solve = jax.vmap(solve, in_axes=(0, 0))
F = D + 3 # l x nx x nx
F_DJinvT = vmap_solve(jnp.transpose(DJ, axes=(0, 2, 1)), jnp.transpose(F, axes=(0, 2, 1)))
F_DJinv = jnp.transpose(F_DJinvT, axes=(0, 2, 1))
i = 10
np.testing.assert_allclose(F_DJinv[i], solve(DJ[i].T, F[i].T).T, rtol=1e-10, atol=1e-10)

res = F_DJinv @ D
np.testing.assert_allclose(res[i], F_DJinv[i] @ D[i], rtol=1e-10, atol=1e-10)
####
eta = jnp.array([1, 2, 3, 4])
D_eta = jnp.einsum('ijk,k->ij', D, eta)
np.testing.assert_allclose(D_eta[i], D[i] @ eta, rtol=1e-15, atol=1e-15)
###
di = jnp.array([[1, 2, 3, 4], [2, 4, 6, 8]])
print(di.shape)
print(J.shape)
temp = jnp.einsum('ij,kj->ki', J, di)
np.testing.assert_allclose(temp[0], J @ di[0], rtol=1e-15, atol=1e-15)
np.testing.assert_allclose(temp[1], J @ di[1], rtol=1e-15, atol=1e-15)
print(temp.shape)