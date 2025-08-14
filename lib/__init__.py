from jax._src.random import orthogonal
import functools, jax, jax.numpy as jnp
from jax import random, jvp, vmap, jit
from jax.flatten_util import ravel_pytree
from typing import Callable, Any


def ravel(params):
    return ravel_pytree(params)


def make_coord_step(loss_fn):
    @functools.partial(jit, static_argnums=(3, 4))
    def step(p_flat, rng, batch, lr, n_grads):
        rng, rk = random.split(rng)
        ks = random.randint(rk, (n_grads,), 0, p_flat.size)
        tang = jax.nn.one_hot(ks, p_flat.size, dtype=p_flat.dtype)
        _, f_lin = jax.linearize(lambda p: loss_fn(p, batch), p_flat)
        gks = vmap(f_lin)(tang)
        p_flat = p_flat.at[ks].add(-lr * gks)
        return p_flat, rng
    return step


def orthogonalize_vectors(x: jnp.ndarray) -> jnp.ndarray:
    n_dirs = x.shape[0]
    vshape = x.shape[1:]
    mat = x.reshape(n_dirs, -1).T
    orig_dtype = x.dtype
    if orig_dtype in (jnp.float16, jnp.bfloat16):
        mat = mat.astype(jnp.float32)
    q, _ = jnp.linalg.qr(mat, mode='reduced')
    k = q.shape[1]
    if k != n_dirs:
        q = jnp.pad(q, ((0, 0), (0, n_dirs - k)))
    out = q.T.reshape((n_dirs,) + vshape)
    return out.astype(orig_dtype)

def forward_grad_random_proj(fun: Callable, rng: jax.random.PRNGKey, ortho = True, n_dirs: int = 4) -> Callable:
    def forward_grad_func(*primals: Any):
        keys = jax.tree.unflatten(
            jax.tree.structure(primals),
            jax.random.split(rng, len(jax.tree.flatten(primals)[0])),
        )
        tangents = jax.tree.map(
            lambda x, key: jax.random.normal(key, (n_dirs, *x.shape)), primals, keys
        )
        if ortho:
            tangents = jax.tree.map(
                lambda x: orthogonalize_vectors(x), tangents
            )
        else:
            tangents = jax.tree.map(
                lambda x: x / jnp.linalg.norm(x, axis=tuple(range(1, len(x.shape))), keepdims=True), tangents
            )
        loss, f_lin = jax.linearize(fun, *primals)
        vmapped = vmap(f_lin, in_axes=(0))
        jvp_ = vmapped(*tangents)
        return (
            loss,
            jax.tree.map(lambda tangent: (jvp_.reshape(-1, *((1,) * len(tangent.shape[1:]))) * tangent).mean(axis=0).clip(-1, 1), tangents)[0],
        )

    return forward_grad_func

