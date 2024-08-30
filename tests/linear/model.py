import jax
import jax.numpy as jnp
from chex import dataclass
from jax.lax import scan

from integrated._base import MVNStandard, LinearObs, LinearTran


@dataclass
class DistillationSSM:
    l: int
    interval: int
    nx: int
    ny: int
    Q: float
    R: float
    prior_x: MVNStandard
    seed: int

    def TranParams(self):
        A = jnp.array([[0.8499, 0.0350, 0.0240, 0.0431],
                       [1.2081, 0.0738, 0.0763, 0.4087],
                       [0.7331, 0.0674, 0.0878, 0.8767],
                       [0.0172, 0.0047, 0.0114, 0.9123]])
        B = jnp.array([[0, 0, 0, 1]]).T
        u = jnp.array([[1]])
        cov = self.Q * jnp.eye(self.nx)
        return LinearTran(A, B, u, cov)

    def ObsParams(self):
        C = jnp.array([[1, 0, 0, 0],
                       [0, 0, 0, 1]])
        cov = jnp.eye(self.ny) * self.R
        return LinearObs(C, cov)

    def TranModel(self, x):
        A = jnp.array([[0.8499, 0.0350, 0.0240, 0.0431],
                      [1.2081, 0.0738, 0.0763, 0.4087],
                      [0.7331, 0.0674, 0.0878, 0.8767],
                      [0.0172, 0.0047, 0.0114, 0.9123]])
        return A @ x

    def ObsModel(self, h, v):
        C = jnp.array([[1, 0, 0, 0],
                      [0, 0, 0, 1]])
        return C @ h + v

    def get_data(self):
        key = jax.random.PRNGKey(self.seed)
        T = self.interval * self.l
        p_gaussian_key, m_gaussian_key = jax.random.split(key, 2)
        x0 = self.prior_x.mean
        chol_Q = jnp.sqrt(self.Q)
        process_noises = jax.random.normal(p_gaussian_key, shape=(T, self.nx)) * chol_Q.T

        chol_R = jnp.sqrt(self.R)
        measurement_noises = jax.random.normal(m_gaussian_key, shape=(self.interval, self.ny)) * chol_R.T

        def body_x(x, inputs):
            p_noise = inputs
            x = self.TranModel(x) + p_noise
            return x, x

        _, true_states = scan(body_x, x0, process_noises)
        true_states = jnp.insert(true_states, 0, x0, 0)

        def body_h(_, k):
            start_index = (k * self.l + 1,0)
            slice_size = (self.l, self.nx)
            x_slice = jax.lax.dynamic_slice(true_states, start_index, slice_size)
            h = 1/self.l * jnp.sum(x_slice, axis=0)
            return _, h

        _, h = scan(body_h, None, jnp.arange(self.interval))

        vmap_func = jax.vmap(self.ObsModel, in_axes=(0, 0))
        y = vmap_func(h, measurement_noises)
        return true_states, h, y
