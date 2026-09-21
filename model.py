# Tiny GPT from scratch in pure NumPy, with a hand-written backward pass.

import os
import sys
import time

import numpy as np


def build_vocab(text):
    # Sorted unique characters in the corpus.
    return sorted(set(text))


def clean_corpus(text):
    # Lowercase and keep only letters, period, comma, and whitespace, so the
    # model never has to learn numbers or punctuation it cannot produce.
    allowed = set("abcdefghijklmnopqrstuvwxyz., \n")
    return "".join(c for c in text.lower() if c in allowed)


def build_maps(vocab):
    # Character <-> integer maps.
    return {c: i for i, c in enumerate(vocab)}, {i: c for i, c in enumerate(vocab)}


def encode(text, stoi):
    return [stoi[c] for c in text]


def decode(ids, itos):
    return "".join(itos[i] for i in ids)


def encode_corpus(text, stoi):
    return np.array([stoi[c] for c in text], dtype=np.int64)


def split_train_val(data, train_frac=0.9):
    split = int(len(data) * train_frac)
    return data[:split], data[split:]


def get_batch(data, block_size, batch_size, rng):
    # Random (input, target) subsequences shifted by one token.
    offsets = rng.integers(0, len(data) - block_size, size=batch_size)
    x = np.vstack([data[o:o + block_size] for o in offsets])
    y = np.vstack([data[o + 1:o + 1 + block_size] for o in offsets])
    return x, y


def softmax(x):
    shifted = x - np.max(x, axis=-1, keepdims=True)
    exp = np.exp(shifted)
    return exp / np.sum(exp, axis=-1, keepdims=True)


def create_token_embedding(vocab_size, d_model, scale=0.02, rng=None):
    rng = rng or np.random.default_rng()
    return rng.normal(0.0, scale, size=(vocab_size, d_model))


def create_positional_embedding(block_size, d_model, scale=0.02, rng=None):
    # Centered normal init; plain uniform(0, scale) biases every position positive.
    rng = rng or np.random.default_rng()
    return rng.normal(0.0, scale, size=(block_size, d_model))


def stack_blocks(n_layers, d_model, n_heads, d_ff, rng=None):
    # One transformer block = pre-LN attention + pre-LN FFN.
    # Output projections (Wo, w2) are residual-path layers; GPT-2-style they
    # carry an extra 1/sqrt(2*n_layers) scale so many stacked blocks stay stable.
    rng = rng or np.random.default_rng(0)
    residual = 1.0 / np.sqrt(2 * n_layers)
    blocks = []
    for _ in range(n_layers):
        blocks.append({
            "ln1": {"gamma": np.ones(d_model), "beta": np.zeros(d_model)},
            "ln2": {"gamma": np.ones(d_model), "beta": np.zeros(d_model)},
            "attn": {
                "n_heads": n_heads,
                "Wq": rng.normal(0.0, 0.02, size=(d_model, d_model)),
                "Wk": rng.normal(0.0, 0.02, size=(d_model, d_model)),
                "Wv": rng.normal(0.0, 0.02, size=(d_model, d_model)),
                "Wo": rng.normal(0.0, 0.02, size=(d_model, d_model)) * residual,
                "bo": np.zeros(d_model),
            },
            "ffn": {
                "w1": rng.normal(0.0, 0.02, size=(d_model, d_ff)),
                "b1": np.zeros(d_ff),
                "w2": rng.normal(0.0, 0.02, size=(d_ff, d_model)) * residual,
                "b2": np.zeros(d_model),
            },
        })
    return blocks


def layernorm_cached(x, gamma, beta, eps=1e-5):
    mean = np.mean(x, axis=-1, keepdims=True)
    var = np.mean((x - mean) ** 2, axis=-1, keepdims=True)
    std = np.sqrt(var + eps)
    x_hat = (x - mean) / std
    cache = {"x_hat": x_hat, "std": std, "gamma": gamma}
    return x_hat * gamma + beta, cache


def layernorm_backward(dy, cache):
    axes = tuple(range(dy.ndim - 1))
    dgamma = np.sum(dy * cache["x_hat"], axis=axes)
    dbeta = np.sum(dy, axis=axes)
    dx_hat = dy * cache["gamma"]
    dx = (dx_hat
          - np.mean(dx_hat, axis=-1, keepdims=True)
          - cache["x_hat"] * np.mean(dx_hat * cache["x_hat"], axis=-1, keepdims=True)) / cache["std"]
    return dx, dgamma, dbeta


def attention_cached(x, attn):
    n_heads = attn["n_heads"]
    B, T, d_model = x.shape
    d_head = d_model // n_heads

    def project(w):
        return (x @ w).reshape(B, T, n_heads, d_head).transpose(0, 2, 1, 3)

    q, k, v = project(attn["Wq"]), project(attn["Wk"]), project(attn["Wv"])
    scores = q @ k.swapaxes(-1, -2) / np.sqrt(d_head)

    mask = np.tril(np.ones((T, T), dtype=bool))
    scores = np.where(mask, scores, -np.inf)
    probs = softmax(scores)

    merged = (probs @ v).transpose(0, 2, 1, 3).reshape(B, T, d_model)
    cache = {
        "q": q, "k": k, "v": v, "probs": probs, "mask": mask,
        "merged": merged, "x": x, "d_head": d_head,
    }
    return merged @ attn["Wo"] + attn["bo"], cache


def attention_backward(dy, cache, attn):
    B, T, d_model = dy.shape
    d_head = cache["d_head"]
    n_heads = attn["n_heads"]
    q, k, v = cache["q"], cache["k"], cache["v"]
    probs, mask, merged, x = cache["probs"], cache["mask"], cache["merged"], cache["x"]

    d_merged = dy @ attn["Wo"].T
    dWo = merged.reshape(-1, d_model).T @ dy.reshape(-1, d_model)
    dbo = np.sum(dy, axis=(0, 1))

    d_heads = d_merged.reshape(B, T, n_heads, d_head).transpose(0, 2, 1, 3)
    d_probs = d_heads @ v.swapaxes(-1, -2)
    d_v = probs.swapaxes(-1, -2) @ d_heads

    d_scores = probs * (d_probs - np.sum(probs * d_probs, axis=-1, keepdims=True))
    d_scores = np.where(mask, d_scores, 0.0) / np.sqrt(d_head)

    d_q = d_scores @ k
    d_k = d_scores.swapaxes(-1, -2) @ q

    def unproject(head_grad, w):
        flat = head_grad.transpose(0, 2, 1, 3).reshape(B * T, d_model)
        return (flat @ w.T).reshape(B, T, d_model), x.reshape(B * T, d_model).T @ flat

    dx_q, dWq = unproject(d_q, attn["Wq"])
    dx_k, dWk = unproject(d_k, attn["Wk"])
    dx_v, dWv = unproject(d_v, attn["Wv"])

    grads = {"Wq": dWq, "Wk": dWk, "Wv": dWv, "Wo": dWo, "bo": dbo}
    return dx_q + dx_k + dx_v, grads


def ffn_cached(x, ffn):
    h = x @ ffn["w1"] + ffn["b1"]
    a = np.maximum(0, h)
    cache = {"x": x, "h": h, "a": a}
    return a @ ffn["w2"] + ffn["b2"], cache


def ffn_backward(dy, cache, ffn):
    x, h, a = cache["x"], cache["h"], cache["a"]
    dW2 = a.reshape(-1, a.shape[-1]).T @ dy.reshape(-1, dy.shape[-1])
    db2 = np.sum(dy, axis=(0, 1))
    dh = (dy @ ffn["w2"].T) * (h > 0)
    dx = dh @ ffn["w1"].T
    dW1 = x.reshape(-1, x.shape[-1]).T @ dh.reshape(-1, dh.shape[-1])
    db1 = np.sum(dh, axis=(0, 1))
    grads = {"w1": dW1, "b1": db1, "w2": dW2, "b2": db2}
    return dx, grads


def block_forward_cached(x, block):
    ln1_out, ln1_cache = layernorm_cached(x, **block["ln1"])
    attn_out, attn_cache = attention_cached(ln1_out, block["attn"])
    x = x + attn_out
    ln2_out, ln2_cache = layernorm_cached(x, **block["ln2"])
    ffn_out, ffn_cache = ffn_cached(ln2_out, block["ffn"])
    x = x + ffn_out
    cache = {"ln1": ln1_cache, "attn": attn_cache, "ln2": ln2_cache, "ffn": ffn_cache}
    return x, cache


def block_backward(dy, cache, block):
    dx, ffn_grads = ffn_backward(dy, cache["ffn"], block["ffn"])
    dx_ln2, dgamma2, dbeta2 = layernorm_backward(dx, cache["ln2"])
    d_x1 = dy + dx_ln2

    dx, attn_grads = attention_backward(d_x1, cache["attn"], block["attn"])
    dx_ln1, dgamma1, dbeta1 = layernorm_backward(dx, cache["ln1"])
    d_x = d_x1 + dx_ln1

    grads = {
        "ln1": {"gamma": dgamma1, "beta": dbeta1},
        "ln2": {"gamma": dgamma2, "beta": dbeta2},
        "attn": attn_grads,
        "ffn": ffn_grads,
    }
    return d_x, grads


def forward_cached(ids, model):
    B, T = ids.shape
    x = model["tok_emb"][ids] + model["pos_emb"][:T]
    block_caches = []
    for block in model["blocks"]:
        x, cache = block_forward_cached(x, block)
        block_caches.append(cache)
    x, ln_f_cache = layernorm_cached(x, **model["ln_f"])
    cache = {
        "ids": ids,
        "T": T,
        "blocks": block_caches,
        "ln_f": ln_f_cache,
        "lm_head": {"x": x},
    }
    return x @ model["lm_head"]["w_lm"] + model["lm_head"]["b_lm"], cache


def forward(ids, model):
    logits, _ = forward_cached(ids, model)
    return logits


def cross_entropy_loss_and_grad(logits, targets):
    B, T, V = logits.shape
    flat = logits.reshape(-1, V)
    probs = softmax(flat)
    labels = targets.ravel()
    loss = -np.mean(np.log(probs[np.arange(B * T), labels] + 1e-12))
    dlogits = probs.copy()
    dlogits[np.arange(B * T), labels] -= 1.0
    dlogits = dlogits.reshape(logits.shape) / (B * T)
    return loss, dlogits


def backward(dlogits, cache, model):
    dy = dlogits
    x_lm = cache["lm_head"]["x"]
    dW_lm = x_lm.reshape(-1, x_lm.shape[-1]).T @ dy.reshape(-1, dy.shape[-1])
    db_lm = np.sum(dy, axis=(0, 1))
    dx = dy @ model["lm_head"]["w_lm"].T

    dx_ln, dgamma_f, dbeta_f = layernorm_backward(dx, cache["ln_f"])

    block_grads = []
    for block_cache, block in zip(reversed(cache["blocks"]), reversed(model["blocks"])):
        dx_ln, grads = block_backward(dx_ln, block_cache, block)
        block_grads.insert(0, grads)

    d_tok = np.zeros_like(model["tok_emb"])
    np.add.at(d_tok, cache["ids"], dx_ln)
    d_pos = np.zeros_like(model["pos_emb"])
    d_pos[:cache["T"]] = np.sum(dx_ln, axis=0)

    return {
        "tok_emb": d_tok,
        "pos_emb": d_pos,
        "blocks": block_grads,
        "ln_f": {"gamma": dgamma_f, "beta": dbeta_f},
        "lm_head": {"w_lm": dW_lm, "b_lm": db_lm},
    }


def initialize_moments(model):
    if isinstance(model, dict):
        return {k: initialize_moments(v) for k, v in model.items()}
    if isinstance(model, list):
        return [initialize_moments(v) for v in model]
    if isinstance(model, np.ndarray):
        return np.zeros_like(model)
    return None


def _is_finite(tree):
    if isinstance(tree, np.ndarray):
        return bool(np.all(np.isfinite(tree)))
    if isinstance(tree, dict):
        return all(_is_finite(v) for v in tree.values())
    if isinstance(tree, list):
        return all(_is_finite(v) for v in tree)
    return True


def _tree_norm_sq(tree):
    if isinstance(tree, np.ndarray):
        return float(np.sum(tree * tree))
    if isinstance(tree, dict):
        return sum(_tree_norm_sq(v) for v in tree.values())
    if isinstance(tree, list):
        return sum(_tree_norm_sq(v) for v in tree)
    return 0.0


def _scale_tree(tree, factor):
    if isinstance(tree, np.ndarray):
        tree[...] = tree * factor
    elif isinstance(tree, dict):
        for v in tree.values():
            _scale_tree(v, factor)
    elif isinstance(tree, list):
        for v in tree:
            _scale_tree(v, factor)


def clip_grads(grads, max_norm):
    # Scale the whole gradient tree down to max_norm when its global L2 norm
    # exceeds it. Keeps any single step from blowing up into a NaN/divergence.
    norm = np.sqrt(_tree_norm_sq(grads))
    if norm > max_norm:
        _scale_tree(grads, max_norm / (norm + 1e-8))


def adamw_update(model, grads, m, v, t, lr, weight_decay, beta1=0.9, beta2=0.999, eps=1e-8):
    if isinstance(model, dict):
        for k in grads:
            adamw_update(model[k], grads[k], m[k], v[k], t, lr, weight_decay, beta1, beta2, eps)
    elif isinstance(model, list):
        for param, grad, m_, v_ in zip(model, grads, m, v):
            adamw_update(param, grad, m_, v_, t, lr, weight_decay, beta1, beta2, eps)
    elif isinstance(model, np.ndarray):
        m[...] = beta1 * m + (1 - beta1) * grads
        v[...] = beta2 * v + (1 - beta2) * grads * grads
        m_hat = m / (1 - beta1 ** t)
        v_hat = v / (1 - beta2 ** t)
        # Decoupled weight decay: shrink the weight directly, not via the
        # gradient, after the Adam step.
        model[...] = model - lr * (m_hat / (np.sqrt(v_hat) + eps) + weight_decay * model)


def lr_for_step(step, n_steps, lr, warmup_steps, min_lr):
    if step < warmup_steps:
        return lr * (step + 1) / warmup_steps
    progress = (step - warmup_steps) / max(1, n_steps - warmup_steps)
    return min_lr + 0.5 * (lr - min_lr) * (1 + np.cos(np.pi * progress))


def _flatten_into(flat, prefix, tree):
    if isinstance(tree, dict):
        for k, v in tree.items():
            _flatten_into(flat, f"{prefix}/{k}", v)
    elif isinstance(tree, list):
        for i, v in enumerate(tree):
            _flatten_into(flat, f"{prefix}/{i}", v)
    elif isinstance(tree, np.ndarray):
        flat[prefix] = tree
    elif isinstance(tree, (bool, int, float, np.integer, np.floating, str)):
        flat[prefix] = np.array(tree)
    elif tree is None:
        flat[prefix] = np.array(None, dtype=object)
    else:
        raise TypeError(f"cannot flatten {type(tree).__name__} at {prefix}")


def _rebuild_from(npy, prefix):
    # Leaf value stored under its own key; 0-d arrays come back as scalars.
    if prefix in npy.files:
        arr = npy[prefix]
        return arr.item() if arr.ndim == 0 else arr
    keys = [k for k in npy.files if k.startswith(prefix + "/")]
    children = {}
    for k in keys:
        first = k[len(prefix) + 1:].split("/", 1)[0]
        if first not in children:
            children[first] = _rebuild_from(npy, f"{prefix}/{first}")
    if children and all(k.isdigit() for k in children):
        idxs = sorted(int(k) for k in children)
        if idxs == list(range(len(idxs))):
            return [children[str(i)] for i in idxs]
    return children


def save_checkpoint(path, model, m, v, step, rng, n_steps, block_size, vocab_size):
    flat = {}
    _flatten_into(flat, "model", model)
    _flatten_into(flat, "m", m)
    _flatten_into(flat, "v", v)
    _flatten_into(flat, "rng", rng.bit_generator.state)
    flat["step"] = np.array(step)
    flat["n_steps"] = np.array(n_steps)
    flat["block_size"] = np.array(block_size)
    flat["vocab_size"] = np.array(vocab_size)
    tmp = os.path.splitext(path)[0] + ".tmp.npz"
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez(tmp, **flat)
    # Atomic on POSIX: the write either fully lands under the real name or not.
    os.replace(tmp, path)


def load_checkpoint(path):
    state = np.load(path, allow_pickle=True)
    checkpoint = {
        "model": _rebuild_from(state, "model"),
        "m": _rebuild_from(state, "m"),
        "v": _rebuild_from(state, "v"),
        "step": int(state["step"].item()),
        "n_steps": int(state["n_steps"].item()),
        "block_size": int(state["block_size"].item()),
        "vocab_size": int(state["vocab_size"].item()),
    }
    rng = np.random.Generator(np.random.PCG64())
    rng.bit_generator.state = _rebuild_from(state, "rng")
    checkpoint["rng"] = rng
    return checkpoint


def train(model, train_ids, block_size, batch_size=16, lr=3e-4, n_steps=3000,
          warmup_steps=200, min_lr=3e-5, weight_decay=0.1, grad_clip=1.0,
          log_every=250, eval_every=500, progress_every=25, val_ids=None,
          n_eval_steps=2, checkpoint_path=None, save_every=100,
          time_limit=None, rng=None):
    rng = rng or np.random.default_rng(1337)
    m, v = initialize_moments(model), initialize_moments(model)
    start_step = 0
    vocab_size = model["lm_head"]["b_lm"].size
    if checkpoint_path and os.path.exists(checkpoint_path):
        restored = load_checkpoint(checkpoint_path)
        start_step = restored["step"]
        model.clear()
        model.update(restored["model"])
        m, v = restored["m"], restored["v"]
        rng = restored["rng"]
        print(f"[resume] continuing from step {start_step}/{n_steps}")
    elif checkpoint_path:
        # New run: record the initial state so a resume exists even for step 0.
        save_checkpoint(checkpoint_path, model, m, v, 0, rng,
                        n_steps, block_size, vocab_size)

    start = time.time()
    bar_w = 20
    for step in range(start_step, n_steps):
        xb, yb = get_batch(train_ids, block_size, batch_size, rng)
        logits, cache = forward_cached(xb, model)
        loss, dlogits = cross_entropy_loss_and_grad(logits, yb)
        if not np.isfinite(loss):
            print(f"[error] non-finite loss ({loss}) at step {step}; aborting.", file=sys.stderr)
            sys.exit(1)
        grads = backward(dlogits, cache, model)
        if not _is_finite(grads):
            print(f"[error] non-finite gradients at step {step}; aborting.", file=sys.stderr)
            sys.exit(1)
        clip_grads(grads, grad_clip)
        step_lr = lr_for_step(step, n_steps, lr, warmup_steps, min_lr)
        adamw_update(model, grads, m, v, step + 1, step_lr, weight_decay)

        done_step = step + 1
        if checkpoint_path and (done_step % save_every == 0 or done_step == n_steps):
            save_checkpoint(checkpoint_path, model, m, v, done_step, rng,
                            n_steps, block_size, vocab_size)

        if step % log_every == 0 or done_step == n_steps:
            if progress_every:
                sys.stdout.write("\n")
            line = f"[step {step:5d}] train loss {loss:.4f}"
            if val_ids is not None and (step % eval_every == 0 or done_step == n_steps):
                val = validation_loss(model, val_ids, block_size, batch_size // 4 or 1, n_eval_steps)
                line += f"   val {val:.4f}"
            print(line)
        elif progress_every and step % progress_every == 0:
            done_offset = step - start_step + 1
            frac = done_step / n_steps
            elapsed = time.time() - start
            if done_offset > 0:
                eta = (n_steps - done_step) * elapsed / done_offset
            else:
                eta = 0.0
            bar = "#" * int(frac * bar_w) + "-" * (bar_w - int(frac * bar_w))
            sys.stdout.write(
                f"\r[{bar}] {done_step}/{n_steps} ({100 * frac:4.1f}%) "
                f"loss {loss:.4f} lr {step_lr:.2e} "
                f"{int(elapsed // 60)}:{int(elapsed % 60):02d} ETA {int(eta // 60)}:{int(eta % 60):02d}"
            )
            sys.stdout.flush()

        if time_limit is not None and done_step < n_steps:
            # Pause once the wall-clock budget is spent; the checkpoint already
            # covers every step up to here. A later run will resume from step.
            remaining = time_limit - (time.time() - start)
            if remaining <= 0:
                if checkpoint_path:
                    save_checkpoint(checkpoint_path, model, m, v, done_step, rng,
                                    n_steps, block_size, vocab_size)
                sys.stdout.write("\n")
                print(f"[pause] time budget spent at step {done_step}/{n_steps}; "
                      f"re-run to continue.")
                return model

    if progress_every:
        sys.stdout.write("\n")
    elapsed = time.time() - start
    print(f"training finished in {int(elapsed // 60)}:{int(elapsed % 60):02d}")
    return model


def validation_loss(model, val_ids, block_size, batch_size=4, n_eval_batches=2):
    rng = np.random.default_rng(42)
    total = 0.0
    for _ in range(n_eval_batches):
        x, y = get_batch(val_ids, block_size, batch_size, rng)
        logits = forward(x, model)
        probs = softmax(logits.reshape(-1, logits.shape[-1]))
        correct = probs[np.arange(y.size), y.ravel()]
        total += -np.mean(np.log(correct + 1e-12))
    return total / n_eval_batches


def generate(model, prompt, n_new_tokens, block_size, temperature=1.0, top_k=0, rng=None):
    rng = rng or np.random.default_rng()
    ctx = prompt
    for _ in range(n_new_tokens):
        logits = forward(ctx[:, -block_size:], model)[:, -1, :] / temperature
        if top_k > 0:
            top_k = min(top_k, logits.shape[-1])
            threshold = np.partition(logits, -top_k, axis=-1)[:, -top_k, np.newaxis]
            logits = np.where(logits >= threshold, logits, -np.inf)
        probs = softmax(logits)
        token = int(rng.choice(probs.shape[-1], p=probs[0]))
        ctx = np.concatenate([ctx, np.array([[token]], dtype=ctx.dtype)], axis=1)
    return ctx