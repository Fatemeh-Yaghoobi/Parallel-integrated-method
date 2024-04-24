import jax
import jax.numpy as jnp


def none_or_concat(x, y, position=1):
    if x is None or y is None:
        return None
    if position == 1:
        return jax.tree_map(lambda a, b: jnp.concatenate([a[None, ...], b]), y, x)
    else:
        return jax.tree_map(lambda a, b: jnp.concatenate([b, a[None, ...]]), y, x)

