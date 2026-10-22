"""Reference solutions for notebook 03. Try the exercises yourself first.

Paste each block into the matching exercise cell. Needs the notebook's imports
(Counter, numpy as np, torch).
"""

from collections import Counter

import numpy as np  # noqa: F401
import torch  # noqa: F401

# -------- E1: BPE from scratch --------
def get_pair_counts(ids):
    return Counter(zip(ids, ids[1:]))


def merge(ids, pair, new_id):
    out, i = [], 0
    while i < len(ids):
        if i < len(ids) - 1 and ids[i] == pair[0] and ids[i + 1] == pair[1]:
            out.append(new_id)
            i += 2
        else:
            out.append(ids[i])
            i += 1
    return out


def learn_merges(text, n_merges):
    ids = list(text.encode("utf-8"))
    merges = []
    for k in range(n_merges):
        counts = get_pair_counts(ids)
        if not counts:
            break
        pair = counts.most_common(1)[0][0]
        ids = merge(ids, pair, 256 + k)
        merges.append(pair)
    return merges


def bpe_encode(text, merges):
    ids = list(text.encode("utf-8"))
    for k, pair in enumerate(merges):
        ids = merge(ids, pair, 256 + k)
    return ids


def bpe_decode(ids, merges):
    vocab = {i: bytes([i]) for i in range(256)}
    for k, (a, b) in enumerate(merges):
        vocab[256 + k] = vocab[a] + vocab[b]
    return b"".join(vocab[i] for i in ids).decode("utf-8", errors="replace")

# -------- E2: x / y shift --------
def make_xy(ids, context_size):
    if len(ids) < context_size + 1:
        raise ValueError(f"need at least {context_size + 1} ids, got {len(ids)}")
    return ids[:context_size], ids[1 : context_size + 1]

# -------- E3: Sliding windows --------
def sliding_windows(ids, max_length, stride):
    return [
        (ids[i : i + max_length], ids[i + 1 : i + max_length + 1])
        for i in range(0, len(ids) - max_length, stride)
    ]

# -------- E4: Memmap batch sampler --------
def sample_batch(data, batch_size, context_length, rng):
    # largest valid start leaves room for context_length + 1 tokens
    starts = rng.integers(0, len(data) - context_length, size=batch_size)
    x = np.stack([data[s : s + context_length] for s in starts]).astype(np.int64)  # (B, T)
    y = np.stack([data[s + 1 : s + context_length + 1] for s in starts]).astype(np.int64)  # (B, T)
    return torch.from_numpy(x), torch.from_numpy(y)

# -------- E5: Embedding lookup two ways --------
def lookup(W, ids):
    return W[ids]  # (...,) -> (..., C)


def lookup_onehot(W, ids):
    one_hot = torch.nn.functional.one_hot(ids, num_classes=W.shape[0]).to(W.dtype)  # (..., V)
    return one_hot @ W  # (..., V) @ (V, C) -> (..., C)

# -------- E6: Token + position embeddings --------
def input_embeddings(tok_W, pos_W, ids):
    _, T = ids.shape
    return tok_W[ids] + pos_W[:T]  # (B, T, C) + (T, C): positions broadcast over the batch
