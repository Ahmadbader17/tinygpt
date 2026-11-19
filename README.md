# Tiny GPT

**A GPT-style language model built from scratch in PyTorch, trained end to end on a laptop CPU, and shipped as a tested API, a Docker image and a published Hugging Face model.**

[![ci](https://github.com/Ahmadbader17/tinygpt/actions/workflows/ci.yml/badge.svg)](https://github.com/Ahmadbader17/tinygpt/actions/workflows/ci.yml)
· **Model:** [huggingface.co/Ahmadbader17/tiny-gpt](https://huggingface.co/Ahmadbader17/tiny-gpt)

**Tech stack:** Python · PyTorch · Transformer architecture · BPE tokenization · FastAPI · Pydantic · Docker · Hugging Face Hub · pytest · mypy · ruff · GitHub Actions · uv

---

## Aim
I wanted to understand how large language models actually work by building one myself, with no prebuilt model classes,
and to take it all the way from raw text to a deployed service, the way a real ML system is built.

## What I built
- **Transformer model from scratch:** causal multi-head self-attention, pre-norm blocks, MLP, weight tying (5.8M parameters).
- **Tokenizer and data pipeline:** byte-level BPE tokenizer (vocab 4,096) and memory-mapped training data.
- **Training loop:** AdamW, learning-rate warmup + cosine decay, gradient clipping, checkpoints with exact resume.
- **Inference:** text generation with temperature, top-k sampling and fixed seeds, available from a CLI.
- **Serving:** typed FastAPI service (`/v1/completions`) with input validation, a request queue and a demo web page.
- **Deployment:** non-root Docker image; model weights and a model card published on the Hugging Face Hub.
- **Engineering quality:** 119 automated tests, strict type checking, linting and CI on every push.

## Results
Trained on 77.8M tokens of the TinyStories dataset in **7.9 hours on a single laptop CPU (no GPU)**.

| Model | Validation loss (nats) | Validation perplexity (lower is better) |
|---|---|---|
| Uniform guess | 8.32 | 4096 |
| Unigram counts | 5.88 | 357 |
| Bigram counts | 3.65 | 38.4 |
| **Tiny GPT** | **1.688** | **5.41** |

Sample output (prompt in bold, not cherry-picked):

> **One day, a little girl named** Lucy went to play outside. She saw a big tree with a pretty flower. She thought it would be fun to draw with it.

![loss curve](docs/img/loss_curve.png)

Full numbers, more samples and failure cases are in [`docs/RESULTS.md`](docs/RESULTS.md).

## Try the published model
```bash
git clone https://github.com/Ahmadbader17/tinygpt && cd tinygpt && uv sync
uv run tinygpt generate --hub Ahmadbader17/tiny-gpt --prompt "Once upon a time, a little fox" --seed 42
```
Or run the API with Docker:
```bash
docker build -t tiny-gpt .
docker run -p 7860:7860 -e TINYGPT_HF_REPO=Ahmadbader17/tiny-gpt tiny-gpt
# open http://localhost:7860 (demo) or http://localhost:7860/docs (API)
```

## Train it yourself
Needs Python 3.12 and [uv](https://docs.astral.sh/uv/). Everything runs on CPU; a GPU is used if present.
```bash
make setup     # install the locked environment and pre-commit hooks
make check     # lint, format check, mypy --strict, pytest (about 30 s)
make data      # stream 300k + 5k stories, train the tokenizer, write uint16 .bin files (a few minutes)
make train     # configs/small.toml into runs/small; Ctrl-C saves, re-run with --resume to continue
make generate PROMPT="Once upon a time"
make export && make serve    # bundle runs/small for serving, then http://localhost:8000
```
`make train` runs 7,500 steps (about 6 to 7 hours at about 2,500 tokens/s with 12 threads, peak memory about 3.5 GB).
For a quick end-to-end check, `uv run tinygpt train --config configs/small.toml --max-steps 100 --run-dir runs/quick`
takes about 6 minutes and reaches a validation loss of 4.50.

To reproduce the reported run exactly (seed 1337, 9,500 steps), then the baselines and the final number:
```bash
uv run tinygpt train --config configs/small.toml --max-steps 9500 --run-dir runs/overnight_long
uv run tinygpt baselines --config configs/small.toml
uv run tinygpt export --run runs/overnight_long   # prints val loss over 200 fixed batches
```
`uv run tinygpt bench --config configs/small.toml` measures throughput per thread count; on a hybrid Intel CPU,
12 threads trained 4 to 5 times faster than all 22, so set `num_threads` from it on your machine.

## How it works
```text
text -> byte-level BPE -> token ids (B, T)
     -> token embedding + learned position embedding         (B, T, 256)
     -> 6 x [ x + attention(LayerNorm(x)); x + MLP(LayerNorm(x)) ]
          attention: 8 heads of 32 dims, causal mask, softmax(QK^T / sqrt(32)) V
          MLP: 256 -> 1024 -> 256 with GELU
     -> LayerNorm -> output head tied to the token embedding  (B, T, 4096) logits
loss = cross-entropy of every position against the next token
```

| Module | Responsibility |
|---|---|
| `config.py` | frozen dataclasses loaded from TOML; every hyperparameter lives in `configs/*.toml` |
| `data/` | stream TinyStories, clean, encode to a flat `uint16` file, sample random `(x, y)` windows from a memmap |
| `tokenizer/` | byte-level BPE (Hugging Face `tokenizers`) behind a small protocol |
| `model/` | attention (a readable reference path plus PyTorch's fused kernel, tested equal), block, GPT |
| `training/` | LR schedule, one training step, eval, atomic checkpoints with exact resume, throughput bench |
| `inference/` | sampling loop and `CompletionService`, the one entry point shared by the CLI and the API |
| `serving/` | FastAPI app: validation (422), loading (503), busy (429) |

Key hyperparameters (`configs/small.toml`):

| | |
|---|---|
| Model | 6 layers, d_model 256, 8 heads, context 256, vocab 4096, dropout 0.1, 5,853,184 parameters |
| Batch | 32 x 256 = 8,192 tokens per step |
| Optimiser | AdamW, betas (0.9, 0.95), weight decay 0.1 on matrices only, gradient clip 1.0 |
| Schedule | 200 warmup steps to 1e-3, cosine decay to 1e-4 |
| Seed | 1337 (same config and seed give the same tokens on CPU) |

More: [`docs/architecture.md`](docs/architecture.md) (diagrams, file formats), [`DESIGN.md`](DESIGN.md) and
[`docs/DECISIONS.md`](docs/DECISIONS.md) (every decision and why), [`docs/LEARNING.md`](docs/LEARNING.md) (each stage
with tensor shapes), [`docs/AUDIT.md`](docs/AUDIT.md) (the latest code audit).

## Key takeaways
- A small model trained on modest hardware can learn fluent, grammatical English when the data matches its size.
- **Measure, don't guess:** benchmarking showed 12 CPU threads trained 4 to 5 times faster than all 22, which changed how the run was planned.
- Tests that check the maths (causality, attention equivalence, expected starting loss, exact resume) caught bugs early and made the long training run safe to trust.
- Building the full path, from data to a deployed API, taught as much as building the model itself.

## Limitations
- It writes simple children's stories; it is not a chat assistant and cannot follow instructions.
- Longer stories lose the plot or repeat themselves; the context window is 256 tokens.
- Generation has no KV cache, so each new token re-runs the whole context (about 0.5 s per 100 tokens on the laptop CPU).
- One training run, one seed: the numbers above have no error bars.
- Not evaluated for bias or safety.

## Weight Space Programming study
A preregistered replication of "Sparks of Weight-Space Programming" on this model (reading, repairing and editing the
weights by hand) lives on branch [`exp/weight_space_programming`](https://github.com/Ahmadbader17/tinygpt/tree/exp/weight_space_programming),
with its report in `docs/WSP_REPORT.md` there. It is kept off the main line so this repo stays the model itself.

## Licence and credits
Code and weights: MIT. Training data: [TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories) by Ronen Eldan and Yuanzhi Li (CDLA-Sharing-1.0).
