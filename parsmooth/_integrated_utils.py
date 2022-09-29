import jax
import jax.numpy as jnp
from jax import jit
from jax.lax import cond, scan


def A_bar(A, l):
    def body(carry, i):
        A = carry
        return A, jnp.linalg.matrix_power(A, i)
    _, A_stack = jax.lax.scan(body, A, jnp.arange(1, l+1))
    return 1/l * jnp.sum(A_stack, axis=0)


A = jnp.array([[0.8499, 0.0350, 0.0240, 0.0431],
              [1.2081, 0.0738, 0.0763, 0.4087],
              [0.7331, 0.0674, 0.0878, 0.8767],
              [0.0172, 0.0047, 0.0114, 0.9123]])

l=4


@jit
def matrix_power_while_inner(val, F):
    i, cur_val = val

    return i - 1, F @ cur_val


@jit
def matrix_power_while(F, n):
    cond_fun = lambda val: val[0] >= 0
    init_val = (n - 1, jnp.eye(F.shape[0]))
    body_fun = lambda val: matrix_power_while_inner(val, F)

    res = while_loop(cond_fun, body_fun, init_val)

    return res[1]


n = 140


@jit
def scan_fun(carry, xs):
    # One step of the iteration
    n, z, result = carry
    new_n, bit = divmod(n, 2)

    new_result = cond(bit, lambda x: z @ x, lambda x: x, result)

    # No more computation necessary if n = 0
    # Is there a better way to early break rather than just returning something empty?
    new_z = cond(new_n, lambda z: z @ z, lambda _: jnp.empty(z.shape), z)

    return (new_n, new_z, new_result), None


@jit
def matrix_power_scan(F, n, upper_limit=32):
    # TODO: I think we can avoid setting the third carry element to eye and save one matrix multiply
    init_carry = n, F, jnp.eye(F.shape[0])

    result = cond(n == 1, lambda _: F, lambda _: scan(scan_fun, init_carry, None, length=upper_limit)[0][2],
                  F)

    return result

@jit
def scan_fun(carry, xs):
    # One step of the iteration
    n, z, result = carry

    new_result = cond(bit, lambda x: z @ x, lambda x: x, result)

    # No more computation necessary if n = 0
    # Is there a better way to early break rather than just returning something empty?
    new_z = cond(new_n, lambda z: z @ z, lambda _: jnp.empty(z.shape), z)

    return (new_n, new_z, new_result), None


@jit
def matrix_power_scan(F, l):
    # TODO: I think we can avoid setting the third carry element to eye and save one matrix multiply
    init_carry = l, F, jnp.eye(F.shape[0])

    result = cond(l == 1, lambda _: F, lambda _: scan(scan_fun, init_carry, None, length=l)[0][2], F)

    return result