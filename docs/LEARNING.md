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
