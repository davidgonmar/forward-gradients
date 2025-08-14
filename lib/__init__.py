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

        def proj(e):
            _, g = jvp(lambda p: loss_fn(p, batch), (p_flat,), (e,))
            return g

        gks = vmap(proj)(tang)
        p_flat = p_flat.at[ks].add(-lr * gks)
        return p_flat, rng

    return step


def forward_grad_random_proj(fun: Callable, rng: jax.random.PRNGKey):
    def forward_grad_func(*primals: Any):
        keys = jax.tree.unflatten(
            jax.tree.structure(primals),
            jax.random.split(rng, len(jax.tree.flatten(primals)[0])),
        )
        tangents = jax.tree.map(
            lambda x, key: jax.random.normal(key, x.shape), primals, keys
        )
        loss, jvp = jax.jvp(fun, primals=primals, tangents=tangents)
        return (
            loss,
            jax.tree.map(lambda tangent: jnp.clip(jvp, -5, 5) * tangent, tangents)[0],
        )

    return forward_grad_func
