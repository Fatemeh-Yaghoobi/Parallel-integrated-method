import jax
import jax.numpy as jnp
import numpy
from jax.lax import associative_scan, dynamic_slice

from integrated._base import LinearTran, LinearObs


def _slow_rate_integrated_params(transition_model, l: int):
    A, B, u, Q = transition_model

    A_I_tensor = jnp.stack([A] * (l - 1) + [jnp.eye(*A.shape)])                     # [A, A, A,.., A, I],           dim: l x nx x nx
    AI_associative = associative_scan(lambda a, b: a @ b, A_I_tensor, reverse=True) # [A^(l-1), ..., A^2, A, I],    dim: l x nx x nx
    A_AI_associative = A @ AI_associative                                           # [A^l, ..., A^2, A],           dim: l x nx x nx
    A_bar = (1 / l) * jnp.sum(A_AI_associative, axis=0)                             # 1/l (A^l + ... + A^2 + A)     dim: nx x nx

    G_bar = (1 / l) * associative_scan(jnp.add, AI_associative, reverse=True)       #1/l[(I+ ... + A^(l-1)), ..., I] dim: l x nx x nx
    B_bar = G_bar @ B                                                               #     " @ B                      dim: l x nx x nu
    u_bar = jnp.array(jnp.stack([u] * l))                                           # [u, u, ..., u]                 dim: l x nu x 1
    Q_bar = jnp.sum(G_bar @ Q @ jnp.transpose(G_bar, axes=(0, 2, 1)), axis=0)       # dim: nx x nx
    Bu_bar = jnp.einsum('ikl,ilm->km', B_bar, u_bar).reshape(-1,)                   # dim: nx,
    return A_bar, G_bar, B_bar, u_bar, Bu_bar, Q_bar


def _fast_rate_integrated_params(transition_model, i: int, l: int):
    A, B, u, Q = transition_model
    A_bar, G_bar, _, _, Bu_bar, Q_bar = _slow_rate_integrated_params(transition_model, l)
    # C, R = observation_model

    ubar_f = jnp.array(jnp.stack([u] * i))                                     # [u u ... u],                  dim: i x nu x 1
    A_I_tensor = jnp.stack([A] * (i - 1) + [jnp.eye(*A.shape)])                # [A,.., A, I],                 dim: i x nx x nx
    Gbar_i_f = associative_scan(lambda a, b: a @ b, A_I_tensor, reverse=True)  # [A^(i-1), ..., A^2, A, I],    dim: i x nx x nx
    Bbar_i_f = Gbar_i_f @ B                                                    # [A^(i-1)B, ..., A^2B, AB, B], dim: i x nx x nu

    temp = dynamic_slice(Gbar_i_f, [0, 0, 0], [1, A.shape[0], A.shape[1]]).reshape(*A.shape)  # A^(i-1)
    At_i = A @ temp                                                                           # A @ A^(i-1) = A^i,  dim: nx x nx
    Bu_i_bar = jnp.einsum('ikl,ilm->km', Bbar_i_f, ubar_f).reshape(-1, )                      # dim: nx,
    c_i_f = Bu_i_bar - At_i @ Bu_bar                                                          # dim: nx,

    G_bar_i = G_bar[:i]                                                                      # dim: i x nx x nx
    return ubar_f, Gbar_i_f, Bbar_i_f, At_i, c_i_f, G_bar_i

def _test_fast_rate_integrated_params(transition_model, i, l):
    A, B, u, Q = transition_model
    I = jnp.eye(*A.shape)
    ubar_f, Gbar_i_f, Bbar_i_f, At_i, c_i_f, G_bar_i = _fast_rate_integrated_params(transition_model, i, l)
    # numpy.testing.assert_array_equal(ubar_f.shape, (i, B.shape[1], 1))
    # numpy.testing.assert_allclose(Gbar_i_f, jnp.array([A, I]), rtol=1e-06, atol=0)
    # numpy.testing.assert_allclose(Bbar_i_f, jnp.array([A @ B, B]), rtol=1e-06, atol=0)


def _test_slow_rate_integrated_params(transition_model, l):
    A, B, u, Q = transition_model
    I = jnp.eye(*A.shape)
    A_bar, G_bar, B_bar, u_bar, Bu_bar, Q_bar = _slow_rate_integrated_params(transition_model, l)
    numpy.testing.assert_allclose(A_bar, 1/l * (A + A @ A + A @ A @ A), rtol=1e-06, atol=0)
    numpy.testing.assert_allclose(G_bar[0], 1/l * (I + A + A @ A), rtol=1e-06, atol=0)
    numpy.testing.assert_allclose(G_bar[-1], 1/l * I, rtol=1e-06, atol=0)

    numpy.testing.assert_allclose(B_bar[0], 1/l * (B + A @ B + A @ A @ B), rtol=1e-06, atol=0)
    numpy.testing.assert_allclose(B_bar[-1], 1/l * B, rtol=1e-06, atol=0)
    numpy.testing.assert_allclose(jnp.einsum('ikl,ilm->km', B_bar, u_bar).reshape(-1,),
                                  (B_bar[0] @ u[0] + B_bar[1] @ u[1] + B_bar[2] @ u[2]), rtol=1e-06, atol=0)
    numpy.testing.assert_allclose(Q_bar,
                                  G_bar[0] @ Q @ G_bar[0].T + G_bar[1] @ Q @ G_bar[1].T + G_bar[2] @ Q @ G_bar[2].T,
                                  rtol=1e-06, atol=0)

A = jnp.array([[0.8499, 0.0350, 0.0240, 0.0431],
                   [1.2081, 0.0738, 0.0763, 0.4087],
                   [0.7331, 0.0674, 0.0878, 0.8767],
                   [0.0172, 0.0047, 0.0114, 0.9123]])
B = jnp.array([[0, 0, 0, 1]]).T
u = jnp.array([[1]])
Q = jnp.eye(4)
transition_model = LinearTran(A, B, u, Q)

_test_slow_rate_integrated_params(transition_model, l=3)
_test_fast_rate_integrated_params(transition_model, i=2, l=3)