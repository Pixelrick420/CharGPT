# Tiny GPT from scratch: download real data, train (pausable/resumable), and generate.

import argparse
import os
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
    split_train_val,
    stack_blocks,
    train,
    validation_loss,
)

CHECKPOINT_PATH = "data/checkpoint.npz"


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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the tiny GPT, pausably.")
    parser.add_argument("--minutes", type=float, default=None,
                        help="wall-clock budget per run in minutes; hit it and the "
                             "run pauses with a checkpoint saved, ready to resume later.")
    parser.add_argument("--steps", type=int, default=config.N_STEPS,
                        help="total steps across all runs (default: config.N_STEPS)")
    args = parser.parse_args()

    rng = np.random.default_rng(0)

    text = fetch_corpus()
    text = clean_corpus(text)
    vocab = build_vocab(text)
    stoi, itos = build_maps(vocab)
    data = encode_corpus(text, stoi)
    train_ids, val_ids = split_train_val(data, 0.9)
    print(f"vocab_size={len(vocab)} train={len(train_ids):,} val={len(val_ids):,}")

    resumed = os.path.exists(CHECKPOINT_PATH)
    model = build_model(len(vocab), config.BLOCK_SIZE)
    if not resumed:
        before = validation_loss(model, val_ids, config.BLOCK_SIZE, 4, config.N_EVAL_STEPS)
        print(f"val_loss before training ~ {before:.4f}")

    train(model, train_ids, config.BLOCK_SIZE,
          batch_size=config.BATCH_SIZE, lr=config.LR, n_steps=args.steps,
          warmup_steps=config.WARMUP_STEPS, min_lr=config.MIN_LR,
          weight_decay=config.WEIGHT_DECAY, grad_clip=config.GRAD_CLIP,
          val_ids=val_ids, eval_every=config.EVAL_EVERY, log_every=config.LOG_EVERY,
          progress_every=config.PROGRESS_EVERY, n_eval_steps=config.N_EVAL_STEPS,
          checkpoint_path=CHECKPOINT_PATH, save_every=config.CHECKPOINT_EVERY,
          time_limit=args.minutes * 60 if args.minutes else None, rng=rng)

    prompt = np.array([encode(clean_corpus("ROMEO:"), stoi)])
    generated = generate(model, prompt, 300, config.BLOCK_SIZE,
                         temperature=0.8, top_k=config.TOP_K, rng=rng)
    print("\n----- generated -----")
    print(decode(generated[0], itos))