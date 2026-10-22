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
