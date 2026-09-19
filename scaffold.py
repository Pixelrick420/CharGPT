"""Tiny GPT from scratch: download real data, train, and generate."""

import os
import urllib.request

import numpy as np

from model import (
    build_maps,
    build_vocab,
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

D_MODEL = 16
N_HEADS = 2
D_FF = 32
N_LAYERS = 2
BLOCK_SIZE = 32
BATCH_SIZE = 16
N_STEPS = 6000
LR = 3e-4

CORPUS_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
DATA_PATH = "data/tinyshakespeare.txt"


def fetch_corpus(path=DATA_PATH, url=CORPUS_URL):
    if os.path.exists(path):
        print(f"[data] cached at {path}")
        return open(path, encoding="utf-8").read()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    print(f"[data] downloading {url}")
    urllib.request.urlretrieve(url, path)
    return open(path, encoding="utf-8").read()


def build_model(vocab_size, block_size):
    return {
        "tok_emb": create_token_embedding(vocab_size, D_MODEL),
        "pos_emb": create_positional_embedding(block_size, D_MODEL),
        "blocks": stack_blocks(N_LAYERS, D_MODEL, N_HEADS, D_FF),
        "ln_f": {"gamma": np.ones(D_MODEL), "beta": np.zeros(D_MODEL)},
        "lm_head": {
            "w_lm": np.random.randn(D_MODEL, vocab_size) * 0.02,
            "b_lm": np.zeros(vocab_size),
        },
    }


if __name__ == "__main__":
    np.random.seed(0)
    rng = np.random.default_rng(0)

    text = fetch_corpus()
    vocab = build_vocab(text)
    stoi, itos = build_maps(vocab)
    data = encode_corpus(text, stoi)
    train_ids, val_ids = split_train_val(data, 0.9)
    print(f"vocab_size={len(vocab)} train={len(train_ids):,} val={len(val_ids):,}")

    model = build_model(len(vocab), BLOCK_SIZE)
    before = validation_loss(model, val_ids, BLOCK_SIZE, 4, 2)
    print(f"val_loss before training ~ {before:.4f}")

    train(model, train_ids, BLOCK_SIZE,
          batch_size=BATCH_SIZE, lr=LR, n_steps=N_STEPS, log_every=250, rng=rng)
    after = validation_loss(model, val_ids, BLOCK_SIZE, 4, 2)
    print(f"val_loss after training  ~ {after:.4f}")

    prompt = np.array([encode("ROMEO:", stoi)])
    generated = generate(model, prompt, 300, BLOCK_SIZE,
                         temperature=0.8, top_k=50, rng=rng)
    print("\n----- generated -----")
    print(decode(generated[0], itos))