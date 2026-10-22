# Learning log

## Phase 0: scaffold, config, environment

**Concept.** A `src/` layout puts the package in `src/tinygpt/`, so `import tinygpt` only works
once it is installed (`uv sync` does an editable install). Tests therefore exercise the installed
package, and a missing file in the packaging shows up immediately instead of hiding behind the
current directory. Configs are frozen dataclasses loaded from TOML: a run reads its settings once
and cannot change them mid-flight; unknown keys raise, so a typo like `drop_out` fails loudly.

**Shapes preview.** `ModelConfig` checks `d_model % n_head == 0` because attention later reshapes
`(B, T, C) -> (B, n_head, T, C // n_head)`; with `small`, `C=256`, `n_head=8`, head dim 32.

**Predict-first answer.** Tied embedding matrix = 4096 x 256 = 1,048,576 parameters, about 18% of
the ~5.8M total (blocks: ~12 x 256^2 x 6 = 4.7M; positions: 256 x 256 = 65k). With GPT-2's 50257
vocab it would be 12.9M, which is why we trained our own 4096 BPE.

**Link to your notebook.** The notebook has no config: vocab size and context length are loose
variables. Here they live in one place and every run saves its resolved config.

<details><summary>Check questions</summary>

1. *Why does `src/` layout catch packaging bugs?* Tests import the installed package, not the
   working directory.
2. *Why frozen dataclasses?* Immutability: no hidden mutation during a run, hashable, safe to share.
3. *Why must `d_model` divide by `n_head`?* Heads split the channel dimension evenly.
</details>

**Interview angle.** "I kept configuration declarative (TOML into frozen, validated dataclasses) so
every run is reproducible from one saved file." Mention strict typing and `make check` as the single
gate used locally and in CI.

## Phase 1: tokenizer (byte-level BPE)

**Concept.** Text must become integers. Your notebook split on words and built a dictionary, so
unseen words needed `<|unk|>`. Byte-level BPE starts from the 256 byte values and *learns* merges:
count adjacent pairs in the training text, fuse the most frequent into a new token, repeat until the
vocab reaches 4096 (256 bytes + 1 special + 3839 merges). Encoding replays the merges in the order
they were learned. Because every string is bytes, nothing is ever unknown.

**Shapes.** `encode: str -> list[int]` (length L, each id in `[0, vocab)`); later the pipeline
turns these into `(B, T)` int tensors. Decoding concatenates bytes and decodes UTF-8, which is why
a single id may print as `�`: it holds a partial multi-byte character.

**Worked example (tiny vocab 512, fixture stories).** `"the"` -> `[84, 257]` = `t` + `he`
(not one token: only 255 merges to spend and "the" is usually preceded by a space); `" the"` ->
one token `[261]`; `"Pip"` -> `Pi` + `p`; `"🦊"` -> 4 byte tokens (the emoji is 4 UTF-8 bytes
and never appeared in training, so no merges cover it). With vocab 4096 on real TinyStories, common
words become single tokens.

**Design choices.** `add_prefix_space=False` keeps `" the"` and `"the"` distinct, like GPT-2.
Train the tokenizer on the training split only. `Tokenizer` is a `Protocol`: model and trainer
depend on the interface, so the tiktoken adapter slots in (used in one comparison test).

<details><summary>Check questions</summary>

1. *Why can byte-level BPE never produce an unknown token?* The base alphabet is all 256 bytes,
   so any input decomposes into known symbols.
2. *Why does `" the"` get its own token while `"the"` may not?* Pre-tokenization keeps the leading
   space attached to the word, and `" the"` is far more frequent mid-sentence.
3. *What does a bigger vocab trade off?* Fewer tokens per text (cheaper attention) versus a larger
   embedding/output matrix (vocab x d_model) and sparser statistics per token.
</details>

**Interview angle.** "I trained a 4096-token byte-level BPE rather than reusing GPT-2's 50k vocab,
because at d_model=256 the embedding table alone would have been over twice the rest of the model.
Byte-level means no unknown tokens, and I verified lossless round trips on unicode and emoji."

## Phase 2: data pipeline (clean, encode, memmap)

**Concept.** Training should never touch text. We turn the corpus into numbers once, then training just reads
them. `make data` runs `tinygpt prepare-data`: stream stories, clean them, train BPE on the training split only,
encode each story plus `<|endoftext|>`, and write one flat `uint16` file per split (`train.bin`, `val.bin`) and a
`meta.json`.

**Shapes.** `train.bin` is a flat `(N,)` array of uint16 (N = 60,838,396, about 2 bytes per token). A training batch picks
`B` random starts `s` and slices `x = tokens[s : s+T]`, `y = tokens[s+1 : s+T+1]`, giving `x, y` of shape `(B, T)`,
cast to int64 for the embedding lookup. `y` is `x` shifted left by one, so at every position the target is the
next token. One window therefore gives `T` training examples at once.

**Versus the notebook.** The notebook's `GPTDatasetV1` builds a Python list holding a tensor for every window, so memory grows
as `num_windows x context_length`. The memmap maps the file and the OS pages in only what we touch: RAM stays tiny and
random starts replace `stride`. We keep `SlidingWindowDataset` for deterministic evaluation and tests.

**Design choices.** Validation is held out, and the tokenizer never sees it. Cleaning is light (whitespace, control characters) so
the model still sees natural text. `meta.json` stores the tokenizer SHA256 so a checkpoint can be checked against its tokenizer.
Measured: 202.8 tokens per story, 13 of 300,000 stories dropped as empty.

<details><summary>Check questions</summary>

1. *Why is uint16 enough, and what breaks past 65,535?* ids are below 4096, so 2 bytes hold them. A vocab of 65,536 or more would
   overflow, and `encode_split` raises instead of silently wrapping.
2. *Why are `x` and `y` offset by one token?* The model is trained to predict the next token at every position, so the target
   at position `i` is the input at `i+1`.
3. *Why train the tokenizer on the training split only?* If it saw validation text, merges would be tuned to it and validation
   loss would be slightly optimistic. Held-out data should be untouched.
</details>

**Interview angle.** "I tokenised 300k stories once into a 122 MB uint16 memmap (60.8M tokens), so batches are random slices with
almost no RAM, instead of materialising every window as a tensor. The tokenizer is fit on train only, and `meta.json` records its
hash so checkpoints can be matched to the right vocabulary."

## Phase 3: the model (attention, MLP, GPT)

**Concept.** A GPT turns token ids into next-token scores. Each token id becomes a vector, a position vector is added,
and the sum passes through a stack of identical blocks. Each block lets tokens look at earlier tokens (attention)
and then transforms each token on its own (MLP). A final LayerNorm and a linear head give one score per vocabulary entry.

**Shapes, small config (B=batch, T=256, C=256, H=8 heads, head_dim=32, V=4096).**
```
idx                (B, T)            int64 token ids
wte(idx) + wpe(pos) (B, T, C)        token vector + position vector
-- per block (x6) --
 ln1(x)            (B, T, C)
 qkv(x)            (B, T, 3C)        -> split into q, k, v, each (B, T, C)
 q, k, v           (B, H, T, 32)     view + transpose: heads become a batch dimension
 q @ k^T / sqrt(32)(B, H, T, T)      score of every query position against every key position
 mask + softmax    (B, H, T, T)      future positions get -inf, so weight 0; rows sum to 1
 att @ v           (B, H, T, 32)
 concat heads+proj (B, T, C)         added back onto x (residual)
 mlp(ln2(x))       (B, T, C)         C -> 4C -> C, added back onto x (residual)
-- end --
ln_f, lm_head      (B, T, V)         logits; loss = cross_entropy over B*T rows
```

**The causal mask.** `scores[t, s]` is how much position t cares about position s. Setting it to `-inf` for s > t makes the
softmax weight exactly 0, so token t cannot read the future. That is what lets one forward pass train all `T` next-token
predictions at once without cheating. Test: change a later token and earlier logits stay identical.

**Why divide by sqrt(head_dim).** A dot product of two 32-dim vectors with unit-variance entries has variance 32. Without scaling, the
scores get large, softmax saturates to near one-hot, and gradients vanish. Dividing by sqrt(32) restores variance ~1.

**Pre-norm.** `x = x + attn(ln1(x))` keeps an un-normalised identity path from the output back to the input, so gradients reach early
layers directly. Post-norm puts LayerNorm on that path and is harder to train without careful warmup.

**Weight tying.** `lm_head.weight is wte.weight`: one (4096 x 256) matrix reads tokens in and scores tokens out. It saves
1,048,576 parameters (18% of the model) and ties the two meanings together. Counted once: **5,853,184 parameters**
(4,739,072 without embeddings). The test recomputes this by hand from the design.

**Init and initial loss.** All weights ~ N(0, 0.02), so the logits are tiny and the softmax is almost uniform over V tokens. Cross-entropy
of a uniform guess is `ln(V)`: ln(4096) = 8.318, and the measured untrained loss was 8.371 (tiny: 6.243 vs ln(512) = 6.238).
If your first loss is far from `ln(V)`, something is wrong, which is why this is a test. The residual output projections get an extra `1/sqrt(2*n_layer)` so variance does not grow with depth.

**Versus the notebook.** The notebook stopped at embeddings. Here the embedding layer is the front of a model, and the same `wte` doubles as the output layer.

<details><summary>Check questions</summary>

1. *Why is the untrained loss ln(V)?* Near-zero logits give a uniform distribution, so every token has probability 1/V and
   the cross-entropy is `-ln(1/V) = ln V`.
2. *What would happen without the causal mask?* Position t could read token t+1, which is its own training target.
   Loss would drop fast in training, but the model would be useless at generation time, when the future does not exist.
3. *Why are the naive and sdpa attention paths both kept?* The naive one is readable and serves as a reference; sdpa is faster.
   A test checks they agree to 1e-5, so the optimised path is proven correct.
</details>

**Interview angle.** "I implemented GPT-2 style attention twice: an explicit softmax(QK^T/sqrt(d))V with a triangular mask, and the
fused PyTorch kernel, and a test asserts they agree and that the future never leaks into the past. I also check that the parameter count
equals a hand-derived formula and that the untrained loss is ln(vocab), which catches wiring and init bugs before any training."
