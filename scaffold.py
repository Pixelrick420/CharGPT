# Tiny GPT from scratch: download real data, train (pausable/resumable), and generate.

import argparse
import os
import shutil
import sys
import urllib.request

import numpy as np

import config
from model import (
    build_maps,
    build_vocab,
    clean_corpus,
    create_positional_embedding,
    create_token_embedding,
    decode,
    encode,
    encode_corpus,
    generate,
    load_checkpoint,
    save_checkpoint,
    split_train_val,
    stack_blocks,
    train,
    validation_loss,
)

CHECKPOINT_PATH = config.CHECKPOINT_PATH


def fetch_corpus(path=config.DATA_PATH, url=config.CORPUS_URL):
    if os.path.exists(path):
        print(f"[data] cached at {path}")
        return open(path, encoding="utf-8").read()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    print(f"[data] downloading {url}")
    urllib.request.urlretrieve(url, path)
    return open(path, encoding="utf-8").read()


def build_model(vocab_size, block_size, rng=None):
    rng = rng or np.random.default_rng(0)
    return {
        "tok_emb": create_token_embedding(vocab_size, config.D_MODEL, rng=rng),
        "pos_emb": create_positional_embedding(block_size, config.D_MODEL, rng=rng),
        "blocks": stack_blocks(config.N_LAYERS, config.D_MODEL, config.N_HEADS, config.D_FF, rng=rng),
        "ln_f": {"gamma": np.ones(config.D_MODEL), "beta": np.zeros(config.D_MODEL)},
        "lm_head": {
            "w_lm": rng.normal(0.0, 0.02, size=(config.D_MODEL, vocab_size)),
            "b_lm": np.zeros(vocab_size),
        },
    }


def widen_positional(model, m, v, new_block_size):
    # Extend the learned positional embeddings to a wider context by tiling the
    # trained rows, mirroring both the weights and their Adam moments so the new
    # positions behave like their trained twins until fine-tuning separates them.
    old_pos = model["pos_emb"]
    b, d = old_pos.shape
    if b >= new_block_size:
        return model, m, v

    def grow(x):
        grown = np.zeros((new_block_size, d), dtype=x.dtype)
        for i in range((new_block_size + b - 1) // b):
            lo, hi = i * b, min((i + 1) * b, new_block_size)
            grown[lo:hi] = x[:hi - lo]
        return grown

    model["pos_emb"] = grow(old_pos)
    m["pos_emb"] = grow(m["pos_emb"])
    v["pos_emb"] = grow(v["pos_emb"])
    print(f"[improve] widened pos_emb {b} -> {new_block_size} by tiling laid rows")
    return model, m, v


def prepare_improve(baseline_path, improve_path, new_block_size, n_steps):
    # Seed the improved checkpoint from the baseline exactly once. Step is reset
    # to 0 so the LR schedule warm-restarts (the weights and Adam moments carry
    # over intact); the baseline file is only ever read, never written.
    if os.path.exists(improve_path):
        print(f"[improve] using existing {improve_path}")
        return improve_path
    if not os.path.exists(baseline_path):
        print("[error] no baseline checkpoint found; run training first.",
              file=sys.stderr)
        sys.exit(1)
    print(f"[improve] seeding {improve_path} from baseline {baseline_path}")
    shutil.copyfile(baseline_path, improve_path)
    restored = load_checkpoint(improve_path)
    model, m, v = restored["model"], restored["m"], restored["v"]
    widen_positional(model, m, v, new_block_size)
    save_checkpoint(improve_path, model, m, v, 0, restored["rng"],
                    n_steps, new_block_size, restored["vocab_size"])
    print(f"[improve] seeded warm-restart checkpoint (step 0/{n_steps}, "
          f"block_size {new_block_size})")
    return improve_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the tiny GPT, pausably.")
    parser.add_argument("--minutes", type=float, default=None,
                        help="wall-clock budget per run in minutes; hit it and the "
                             "run pauses with a checkpoint saved, ready to resume later.")
    parser.add_argument("--steps", type=int, default=config.N_STEPS,
                        help="total steps across all runs (default: config.N_STEPS)")
    parser.add_argument("--improve", action="store_true",
                        help="continue from the trained baseline with a warm-restarted "
                             "LR schedule and wider context; baseline is never modified.")
    parser.add_argument("--prompt", type=str, default="ROMEO:",
                        help="generation prompt (default: ROMEO:)")
    args = parser.parse_args()

    rng = np.random.default_rng(0)

    text = fetch_corpus()
    text = clean_corpus(text)
    vocab = build_vocab(text)
    stoi, itos = build_maps(vocab)
    data = encode_corpus(text, stoi)
    train_ids, val_ids = split_train_val(data, 0.9)
    print(f"vocab_size={len(vocab)} train={len(train_ids):,} val={len(val_ids):,}")

    if args.improve:
        checkpoint_path = prepare_improve(
            CHECKPOINT_PATH, config.IMPROVE_CHECKPOINT_PATH,
            config.IMPROVE_BLOCK_SIZE, config.IMPROVE_N_STEPS)
        block_size = config.IMPROVE_BLOCK_SIZE
        n_steps = config.IMPROVE_N_STEPS
    else:
        checkpoint_path = CHECKPOINT_PATH
        block_size = config.BLOCK_SIZE
        n_steps = args.steps

    resumed = os.path.exists(checkpoint_path)
    model = build_model(len(vocab), block_size)
    if not resumed:
        before = validation_loss(model, val_ids, block_size, 4, config.N_EVAL_STEPS)
        print(f"val_loss before training ~ {before:.4f}")

    train(model, train_ids, block_size,
          batch_size=config.BATCH_SIZE, lr=config.LR, n_steps=n_steps,
          warmup_steps=config.WARMUP_STEPS, min_lr=config.MIN_LR,
          weight_decay=config.WEIGHT_DECAY, grad_clip=config.GRAD_CLIP,
          val_ids=val_ids, eval_every=config.EVAL_EVERY, log_every=config.LOG_EVERY,
          progress_every=config.PROGRESS_EVERY, n_eval_steps=config.N_EVAL_STEPS,
          checkpoint_path=checkpoint_path, save_every=config.CHECKPOINT_EVERY,
          time_limit=args.minutes * 60 if args.minutes else None, rng=rng)

    prompt = np.array([encode(clean_corpus(args.prompt), stoi)])
    generated = generate(model, prompt, 300, block_size,
                         temperature=0.8, top_k=config.TOP_K, top_p=config.TOP_P, rng=rng)
    print("\n----- generated -----")
    print(decode(generated[0], itos))