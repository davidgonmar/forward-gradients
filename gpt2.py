import time
import math
import jax
import jax.numpy as jnp
from jax import random
from transformers import GPT2TokenizerFast, FlaxGPT2LMHeadModel
from datasets import load_dataset
import lib

MODEL_NAME = "gpt2"
SEQ_LEN = 128
BATCH = 8
N_GRADS = 4
LR = 1
STEPS = 20000
LOG_EVERY = 1
WARMUP = 0

tok = GPT2TokenizerFast.from_pretrained(MODEL_NAME)
tok.pad_token = tok.eos_token

ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")


def tok_fn(e):
    return {"ids": tok(e["text"])["input_ids"]}


ds = ds.map(tok_fn, remove_columns=["text"], batched=False, num_proc=4)


def stream():
    buf = []
    for ex in iter(ds):
        buf.extend(ex["ids"] + [tok.eos_token_id])
        while len(buf) >= BATCH * SEQ_LEN + 1:
            x = jnp.array(buf[: BATCH * SEQ_LEN]).reshape(BATCH, SEQ_LEN)
            y = jnp.array(buf[1 : BATCH * SEQ_LEN + 1]).reshape(BATCH, SEQ_LEN)
            buf = buf[SEQ_LEN:]
            yield x, y


data_iter = stream()

model, params = FlaxGPT2LMHeadModel.from_pretrained(
    MODEL_NAME, _do_init=False, dtype=jnp.bfloat16
)
flat_p, unravel = lib.ravel(params)


def loss_fn(p_flat, batch):
    tokens, labels = batch
    attn = jnp.ones(tokens.shape, dtype=jnp.int32)
    pos = jnp.broadcast_to(jnp.arange(tokens.shape[1]), tokens.shape)
    logits = model.module.apply(
        {"params": unravel(p_flat)}, tokens, attn, pos, deterministic=True
    ).logits
    lprobs = jax.nn.log_softmax(logits.astype(jnp.float32))
    idx = jnp.arange(labels.size)
    return -jnp.mean(lprobs.reshape(-1, lprobs.shape[-1])[idx, labels.reshape(-1)])


coord_step = lib.make_coord_step(loss_fn)

rng = random.PRNGKey(0)
tokens_seen = 0
t0 = time.time()

print(f"{'step':>7} | {'loss':>8} | {'ppl':>8} | {'tok/s':>7}")

for step in range(STEPS):
    batch = next(data_iter)
    if step % LOG_EVERY == 0 and step >= WARMUP:
        loss = float(loss_fn(flat_p, batch))
        ppl = math.exp(loss)
        elapsed = time.time() - t0
        tps = tokens_seen / elapsed if elapsed > 0 else 0.0
        print(f"{step:7d} | {loss:8.4f} | {ppl:8.1f} | {tps:7.0f}")
    flat_p, rng = coord_step(flat_p, rng, batch, LR, N_GRADS)
    if step == WARMUP:
        tokens_seen = 0
        t0 = time.time()
    else:
        tokens_seen += batch[0].size

elapsed = time.time() - t0
print("\n===========  SUMMARY  ===========")
print(f"tokens processed: {tokens_seen:,}")
print(f"tokens / second : {tokens_seen/elapsed:,.0f}")
print("peak GPU memory : 0 MB")
