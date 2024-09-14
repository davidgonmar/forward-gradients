import jax
import jax.numpy as jnp
from jax import random
from flax import linen as nn
import optax
import tensorflow_datasets as tfds
from flax.training import train_state
import jax.nn as jnn
import tensorflow as tf
from typing import Any, Callable, Sequence, Optional
import math


def forward_grad(fun: Callable, rng: jax.random.PRNGKey):
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


class MLP(nn.Module):
    @nn.compact
    def __call__(self, x):
        x = x.reshape((x.shape[0], -1))
        x = nn.Dense(1024)(x)
        x = jnn.relu(x)
        x = nn.Dense(1024)(x)
        x = jnn.relu(x)
        x = nn.Dense(10)(x)
        return x


def create_train_state(rng, model, initial_learning_rate):
    params = model.init(rng, jnp.ones([1, 28, 28]))["params"]

    learning_rate_schedule = optax.exponential_decay(
        init_value=initial_learning_rate,
        transition_steps=-10000,
        decay_rate=math.e,
        transition_begin=0,
        staircase=False,
    )
    optimizer = optax.sgd(learning_rate=learning_rate_schedule)
    return train_state.TrainState.create(
        apply_fn=model.apply, params=params, tx=optimizer
    )


def compute_loss(params, batch, apply_fn):
    inputs, targets = batch
    logits = apply_fn({"params": params}, inputs)
    one_hot_targets = jnn.one_hot(targets, 10)
    loss = optax.softmax_cross_entropy(logits, one_hot_targets).mean()
    return loss


def compute_metrics(logits, labels):
    predictions = jnp.argmax(logits, axis=-1)
    accuracy = jnp.mean(predictions == labels)
    return accuracy


@jax.jit
def train_step(state: train_state.TrainState, batch, rng):
    def loss_fn(params):
        return compute_loss(params, batch, state.apply_fn)

    loss, grads = forward_grad(loss_fn, rng)(state.params)
    new_state = state.apply_gradients(grads=grads)
    return loss, new_state


@jax.jit
def eval_step(state, batch):
    inputs, targets = batch
    logits = state.apply_fn({"params": state.params}, inputs)
    return compute_metrics(logits, targets)


def prepare_dataloader():
    ds_builder = tfds.builder("mnist")
    ds_builder.download_and_prepare()

    def preprocess_fn(sample):
        image = sample["image"]
        image = tf.cast(image, tf.float32) / 255.0
        label = sample["label"]
        return image, label

    train_ds = ds_builder.as_dataset(split="train", shuffle_files=True)
    train_ds = train_ds.map(preprocess_fn).shuffle(1024).batch(64)

    test_ds = ds_builder.as_dataset(split="test", shuffle_files=False)
    test_ds = test_ds.map(preprocess_fn).batch(1024)

    return train_ds, test_ds


def train_model(epochs, learning_rate):
    rng = random.PRNGKey(0)
    model = MLP()
    state = create_train_state(rng, model, learning_rate)

    train_ds, test_ds = prepare_dataloader()

    for epoch in range(epochs):
        for batch in tfds.as_numpy(train_ds):
            rng, _ = random.split(rng)
            loss, state = train_step(state, batch, rng)

        test_accuracy = 0
        for batch in tfds.as_numpy(test_ds):
            test_accuracy += eval_step(state, batch)
        test_accuracy /= len(test_ds)

        print(f"Epoch {epoch + 1}, Test accuracy: {test_accuracy * 100:.2f}%")


train_model(epochs=100, learning_rate=2e-5)
