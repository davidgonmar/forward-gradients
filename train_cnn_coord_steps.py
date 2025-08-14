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

BATCH_SIZE = 32
EPOCHS = 100000
LEARNING_RATE = 1
N_GRADS = 128
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
    ds = ds.map(
        lambda x, y: (tf.cast(x, tf.float32) / 255.0, tf.cast(y, tf.int32)),
        num_parallel_calls=tf.data.AUTOTUNE,
    )
    if training:
        ds = ds.shuffle(10_000, seed=SEED, reshuffle_each_iteration=True)
        ds = ds.repeat()
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    for x, y in tfds.as_numpy(ds):
        yield jnp.asarray(x, DTYPE), jnp.asarray(y, jnp.int32)


class CNN(nn.Module):
    @nn.compact
    def __call__(self, x):
        x = nn.Conv(32, (3, 3), padding="SAME", dtype=DTYPE)(x)
        x = nn.relu(x)
        x = x.mean(axis=(1, 2))
        return nn.Dense(NUM_CLASSES, dtype=DTYPE)(x)


def loss_and_acc(params_flat, unravel, model, batch):
    x, y = batch
    logits = model.apply({"params": unravel(params_flat)}, x)
    lprobs = jax.nn.log_softmax(logits.astype(jnp.float32))
    n = y.size
    loss = -jnp.mean(lprobs[jnp.arange(n), y])
    acc = jnp.mean((jnp.argmax(lprobs, axis=-1) == y).astype(jnp.float32))
    return loss, acc


def evaluate(model, params_flat, unravel, steps: int) -> Tuple[float, float]:
    test_iter = make_dataset("test", BATCH_SIZE, training=False)
    total_loss = 0.0
    total_correct = 0.0
    total = 0
    for _ in range(steps):
        x, y = next(test_iter)
        loss, acc = loss_and_acc(params_flat, unravel, model, (x, y))
        bs = x.shape[0]
        total_loss += float(loss) * bs
        total_correct += float(acc) * bs
        total += bs
    return total_loss / total, total_correct / total


def main():
    rng = random.PRNGKey(SEED)
    train_iter = make_dataset("train", BATCH_SIZE, training=True)
    model = CNN()
    params = model.init(rng, jnp.ones((1, 32, 32, 3), DTYPE))["params"]
    params_flat, unravel = lib.ravel(params)
    coord_step = lib.make_coord_step(lambda p, b: loss_and_acc(p, unravel, model, b)[0])
    steps_per_epoch = math.ceil(50_000 / BATCH_SIZE)
    test_steps = math.ceil(10_000 / BATCH_SIZE)
    for epoch in range(1, EPOCHS + 1):
        t0 = time.time()
        running_loss = 0.0
        running_correct = 0.0
        count = 0
        for step in range(1, steps_per_epoch + 1):
            batch = next(train_iter)
            loss, acc = loss_and_acc(params_flat, unravel, model, batch)
            bs = batch[0].shape[0]
            running_loss += float(loss) * bs
            running_correct += float(acc) * bs
            count += bs
            params_flat, rng = coord_step(
                params_flat, rng, batch, LEARNING_RATE, N_GRADS
            )
            if step == 5:
                break
        avg_train_loss = running_loss / count
        avg_train_acc = running_correct / count
        test_loss, test_acc = evaluate(model, params_flat, unravel, test_steps)
        print(
            f"{epoch:5d} | {avg_train_loss:10.4f} | {avg_train_acc:9.3f} | {test_loss:9.4f} | {test_acc:8.3f} | {time.time()-t0:7.1f}"
        )
    print("\nTraining complete.")


if __name__ == "__main__":
    main()
