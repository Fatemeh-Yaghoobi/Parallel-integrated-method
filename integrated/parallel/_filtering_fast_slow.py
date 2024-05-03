import jax
import jax.numpy as jnp
jax.config.update("jax_enable_x64", True)
from integrated._base import MVNStandard
from integrated._utils import none_or_concat
from integrated.parallel._operator_fast_slow import filtering_operator


def filtering(observations: jnp.ndarray,
              x0: MVNStandard,
              all_params):
    m0, P0 = x0
    associative_params = _standard_associative_params(all_params, x0, observations)
    _, filtered_means, filtered_cov, _, _ = jax.lax.associative_scan(jax.vmap(filtering_operator), associative_params)

    # filtered_means = none_or_concat(filtered_means, m0, position=1)
    # filtered_cov = none_or_concat(filtered_cov, P0, position=1)

    res = jax.vmap(MVNStandard)(filtered_means, filtered_cov)

    return res


def _standard_associative_params(all_params, init_x, observations):
    T = observations.shape[0]

    def make_params(sl_params, x0, ys, i):
        predicate = i == 0

        def _first(_):
            return _standard_associative_params_first(sl_params, x0, ys)

        def _generic(_):
            return _standard_associative_params_generic(sl_params, ys)

        return jax.lax.cond(predicate, _first, _generic, None)

    vmapped_fn = jax.vmap(make_params, in_axes=[None, None, 0, 0])
    return vmapped_fn(all_params, init_x, observations, jnp.arange(T))


def _standard_associative_params_generic(all_params, y):
    Abar, Bbar, Gbar, Bbar_u, Qbar, C_bar, M_bar, D_bar, Rx, Q, Du_bar = all_params
    dim_l = Bbar.shape[0]
    temp = jnp.einsum('ijkl,jlm->ikm', Gbar @ Q, jnp.transpose(M_bar, axes=(0, 2, 1)))
    vmap_func = jax.vmap(jax.scipy.linalg.solve, in_axes=(None, 0))
    TranK = vmap_func(Rx, jnp.transpose(temp, axes=(0, 2, 1)))
    K = jnp.transpose(TranK, axes=(0, 2, 1))

    F = Abar - jnp.einsum('ijk,kl->ijl', K, C_bar)                                        # dim = l x nx x nx
    d = jnp.einsum('ijk,k->ij', K, y - Du_bar) + Bbar_u                                   # dim = l x nx
    tempT = jnp.transpose(temp, axes=(0, 2, 1))
    D = Qbar - jnp.einsum('ijk,ikl->ijl', K, tempT)                                       # dim = l x nx x nx

    tempt2 = jax.scipy.linalg.solve(Rx, C_bar).T
    eta = tempt2 @ (y - Du_bar)
    J = tempt2 @ C_bar
    return F, d, D, jnp.stack([eta] * dim_l), jnp.stack([J] * dim_l)


def _standard_associative_params_first(all_params, x0, y):
    Abar, Bbar, Gbar, Bbar_u, Qbar, C_bar, M_bar, D_bar, Rx, Q, Du_bar = all_params
    m0, P0 = x0
    dim_l = Bbar.shape[0]
    m_ = Abar @ m0 + Bbar_u                                      # dim = l x nx
    P_ = Abar @ P0 @ jnp.transpose(Abar, axes=(0, 2, 1)) + Qbar  # dim = l x nx x nx


    S = C_bar @ P0 @ C_bar.T + Rx  # dim = ny x ny
    temp = (Abar @ P0 @ C_bar.T
            + jnp.einsum('ijkl,jlm->ikm', Gbar @ Q, jnp.transpose(M_bar, axes=(0, 2, 1))))  # dim = l x nx x ny

    vmap_func = jax.vmap(jax.scipy.linalg.solve, in_axes=(None, 0))
    TranL = vmap_func(S, jnp.transpose(temp, axes=(0, 2, 1)))
    L = jnp.transpose(TranL, axes=(0, 2, 1))  # dim = l x nx x ny
    tempT = jnp.transpose(temp, axes=(0, 2, 1))  # dim = l x ny x nx


    F = jnp.zeros_like(Abar)
    d = m_ + jnp.einsum('ijk,k->ij', L, y - C_bar @ m0 - Du_bar)  # dim = l x nx
    D = P_ - jnp.einsum('ijk,ikl->ijl', L, tempT)                 # dim = l x nx x nx

    tempt2 = jax.scipy.linalg.solve(Rx, C_bar).T

    eta = tempt2 @ (y - Du_bar)
    J = tempt2 @ C_bar
    return F, d, D, jnp.stack([eta] * dim_l), jnp.stack([J] * dim_l)
