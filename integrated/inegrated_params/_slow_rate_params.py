import jax.numpy as jnp
from jax.lax import associative_scan


def _slow_rate_integrated_params_init(transition_model, l: int):
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
    return A_bar, G_bar, B_bar, u_bar, Bu_bar, Q_bar, AI_associative


def _slow_rate_integrated_params(transition_model, observation_model, l: int):
    '''
    Outputs are parameters in Equations 4 and 5 of the derivations.tex + some additional parameters like 'Q_bar', 'Rx' in Equations 6 and 7
    Bu_bar: B_bar x u_bar
    Du_bar: D_bar x u_bar
    '''
    _A_bar, _G_bar, _B_bar, u_bar, _, _, AI_associative = _slow_rate_integrated_params_init(transition_model, l)
    C, R = observation_model
    A, B, u, Q = transition_model
    A_bar = jnp.linalg.matrix_power(A, l)
    G_bar = AI_associative
    B_bar = AI_associative @ B
    Q_bar = jnp.sum(G_bar @ Q @ jnp.transpose(G_bar, axes=(0, 2, 1)), axis=0)
    C_bar = C @ _A_bar
    M_bar = C @ _G_bar
    D_bar = C @ _B_bar
    Bu_bar = jnp.einsum('ikl,ilm->km', B_bar, u_bar).reshape(-1, )
    Du_bar = jnp.einsum('ikl,ilm->km', D_bar, u_bar).reshape(-1, )
    Rx = jnp.sum(M_bar @ Q @ jnp.transpose(M_bar, axes=(0, 2, 1)), axis=0) + R
    return A_bar, G_bar, B_bar, u_bar, Bu_bar, Q_bar, C_bar, M_bar, D_bar, Rx, Q, Du_bar




# def _test_slow_rate_integrated_params(transition_model, l):
#     A, B, u, Q = transition_model
#     I = jnp.eye(*A.shape)
#     A_bar, G_bar, B_bar, u_bar, Bu_bar, Q_bar, _ = _slow_rate_integrated_params_init(transition_model, l)  # tests for l=3
#     numpy.testing.assert_allclose(A_bar, 1/l * (A + A @ A + A @ A @ A), rtol=1e-06, atol=0)
#     numpy.testing.assert_allclose(G_bar[0], 1/l * (I + A + A @ A), rtol=1e-06, atol=0)
#     numpy.testing.assert_allclose(G_bar[-1], 1/l * I, rtol=1e-06, atol=0)
#
#     numpy.testing.assert_allclose(B_bar[0], 1/l * (B + A @ B + A @ A @ B), rtol=1e-06, atol=0)
#     numpy.testing.assert_allclose(B_bar[-1], 1/l * B, rtol=1e-06, atol=0)
#     numpy.testing.assert_allclose(jnp.einsum('ikl,ilm->km', B_bar, u_bar).reshape(-1,),
#                                   (B_bar[0] @ u[0] + B_bar[1] @ u[1] + B_bar[2] @ u[2]), rtol=1e-06, atol=0)
#     numpy.testing.assert_allclose(Q_bar,
#                                   G_bar[0] @ Q @ G_bar[0].T + G_bar[1] @ Q @ G_bar[1].T + G_bar[2] @ Q @ G_bar[2].T,
#                                   rtol=1e-06, atol=0)

