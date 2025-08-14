import time
import math
from typing import Iterator, Tuple

import jax
import jax.numpy as jnp
from jax import random
from flax import linen as nn
import tensorflow as tf
import tensorflow_datasets as tfds
import lib
import functools

BATCH_SIZE = 16
EPOCHS = 100000
LEARNING_RATE = 1
N_GRADS = 64
DTYPE = jnp.float32
NUM_CLASSES = 10
SEED = 0


def _preprocess(x, y):
    x = tf.cast(x, tf.float32) / 255.0
    y = tf.cast(y, tf.int32)
    return x, y


def make_dataset(
    split: str, batch_size: int, training: bool
) -> Iterator[Tuple[jnp.ndarray, jnp.ndarray]]:
    ds = tfds.load("cifar10", split=split, as_supervised=True)
    ds = ds.map(_preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    if training:
        ds = ds.shuffle(10_000, seed=SEED, reshuffle_each_iteration=True)
        ds = ds.repeat()
    ds = ds.batch(batch_size, drop_remainder=True).prefetch(tf.data.AUTOTUNE)
    for x, y in tfds.as_numpy(ds):
        yield jnp.asarray(x, DTYPE), jnp.asarray(y, jnp.int32)


class CNN(nn.Module):
    @nn.compact
    def __call__(self, x, training: bool):
        x = nn.Conv(32, (3, 3), padding="SAME", dtype=DTYPE)(x)
        x = nn.BatchNorm(use_running_average=not training)(x)
        x = nn.relu(x)
        x = nn.Conv(64, (3, 3), padding="SAME", dtype=DTYPE)(x)
        x = nn.BatchNorm(use_running_average=not training)(x)
        x = nn.relu(x)
        x = nn.Conv(128, (3, 3), padding="SAME", dtype=DTYPE)(x)
        x = nn.BatchNorm(use_running_average=not training)(x)
        x = nn.relu(x)
        x = x.mean(axis=(1, 2))
        return nn.Dense(NUM_CLASSES, dtype=DTYPE)(x)


def main():
    rng = random.PRNGKey(SEED)
    train_iter = make_dataset("train", BATCH_SIZE, training=True)
    model = CNN()
    variables = model.init(rng, jnp.ones((1, 32, 32, 3), DTYPE), training=True)
    params = variables["params"]
    batch_stats = variables["batch_stats"]
    params_flat, unravel = lib.ravel(params)

    @jax.jit
    def loss_and_acc_train(pflat, bstats, batch):
        x, y = batch
        variables = {"params": unravel(pflat), "batch_stats": bstats}
        logits, updates = model.apply(
            variables, x, training=True, mutable=["batch_stats"]
        )
        new_bstats = updates["batch_stats"]
        lprobs = jax.nn.log_softmax(logits.astype(jnp.float32))
        n = y.size
        loss = -jnp.mean(lprobs[jnp.arange(n), y])
        acc = jnp.mean((jnp.argmax(lprobs, axis=-1) == y).astype(jnp.float32))
        return loss, acc, new_bstats

    @jax.jit
    def loss_and_acc_eval(pflat, bstats, batch):
        x, y = batch
        variables = {"params": unravel(pflat), "batch_stats": bstats}
        logits = model.apply(variables, x, training=False, mutable=False)
        lprobs = jax.nn.log_softmax(logits.astype(jnp.float32))
        n = y.size
        loss = -jnp.mean(lprobs[jnp.arange(n), y])
        acc = jnp.mean((jnp.argmax(lprobs, axis=-1) == y).astype(jnp.float32))
        return loss, acc

    def loss_for_update(p, b):
        l, _ = loss_and_acc_eval(p, batch_stats, b)
        return l

    coord_step = lib.make_coord_step(loss_for_update)

    @jax.jit
    def train_step(pflat, bstats, rng, batch):
        loss, acc, new_bstats = loss_and_acc_train(pflat, bstats, batch)
        pflat, rng = coord_step(pflat, rng, batch, LEARNING_RATE, N_GRADS)
        return pflat, new_bstats, rng, loss, acc

    test_x, test_y = [], []
    for x, y in make_dataset("test", BATCH_SIZE, training=False):
        test_x.append(x)
        test_y.append(y)
    test_x = jnp.stack(test_x)
    test_y = jnp.stack(test_y)
    TEST_STEPS = int(test_x.shape[0])

    @jax.jit
    def evaluate(pflat, bstats, data_x, data_y):
        def body(i, carry):
            tl, tc = carry
            batch = (data_x[i], data_y[i])
            loss, acc = loss_and_acc_eval(pflat, bstats, batch)
            tl = tl + loss * BATCH_SIZE
            tc = tc + acc * BATCH_SIZE
            return tl, tc

        init = (jnp.array(0.0, dtype=jnp.float32), jnp.array(0.0, dtype=jnp.float32))
        total_loss, total_correct = jax.lax.fori_loop(0, TEST_STEPS, body, init)
        denom = float(TEST_STEPS * BATCH_SIZE)
        return total_loss / denom, total_correct / denom

    steps_per_epoch = math.ceil(50_000 / BATCH_SIZE)

    for epoch in range(1, EPOCHS + 1):
        t0 = time.time()
        running_loss = 0.0
        running_correct = 0.0
        count = 0
        for step in range(1, steps_per_epoch + 1):
            batch = next(train_iter)
            params_flat, batch_stats, rng, loss, acc = train_step(
                params_flat, batch_stats, rng, batch
            )
            bs = batch[0].shape[0]
            running_loss += float(loss) * bs
            running_correct += float(acc) * bs
            count += bs
            if step == 100:
                break
        avg_train_loss = running_loss / count
        avg_train_acc = running_correct / count
        test_loss, test_acc = evaluate(params_flat, batch_stats, test_x, test_y)
        print(
            f"{epoch:5d} | {avg_train_loss:10.4f} | {avg_train_acc:9.3f} | {test_loss:9.4f} | {test_acc:8.3f} | {time.time()-t0:7.1f}"
        )
    print("\nTraining complete.")


if __name__ == "__main__":
    main()
