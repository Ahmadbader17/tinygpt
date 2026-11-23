# Tiny GPT

**A GPT-style language model built from scratch in PyTorch, trained end to end on a laptop CPU, and delivered as a tested API, a Docker image and a published Hugging Face model.**

[![ci](https://github.com/Ahmadbader17/tinygpt/actions/workflows/ci.yml/badge.svg)](https://github.com/Ahmadbader17/tinygpt/actions/workflows/ci.yml)
· **Model:** [huggingface.co/Ahmadbader17/tiny-gpt](https://huggingface.co/Ahmadbader17/tiny-gpt)

**Tech stack:** Python · PyTorch · Transformer architecture · BPE tokenization · FastAPI · Pydantic · Docker · Hugging Face Hub · pytest · mypy · ruff · GitHub Actions · uv

---

## Purpose
The project builds a language model without prebuilt model classes and takes it from raw text to a deployed service, following the structure of a production ML system. Its aim is a working understanding of how large language models operate.

## Scope
- **Transformer model:** causal multi-head self-attention, pre-norm blocks, MLP and weight tying (5.8M parameters).
- **Tokenizer and data pipeline:** byte-level BPE tokenizer (vocabulary 4,096) and memory-mapped training data.
- **Training loop:** AdamW, learning-rate warmup with cosine decay, gradient clipping, and checkpoints with exact resume.
- **Inference:** text generation with temperature, top-k sampling and fixed seeds, available through a CLI.
- **Serving:** typed FastAPI service (`/v1/completions`) with input validation, a request queue and a demo web page.
- **Deployment:** non-root Docker image; model weights and a model card published on the Hugging Face Hub.
- **Engineering quality:** 120 automated tests, type checking, linting and CI on every push.

## Results
The model was trained on 77.8M tokens of the TinyStories dataset in **7.9 hours on a single laptop CPU (no GPU)**.

| Model | Validation perplexity (lower is better) |
|---|---|
| Bigram baseline | 38.4 |
| **Tiny GPT** | **5.4** |

Sample output (prompt in bold; not cherry-picked):

> **One day, a little girl named** Lucy went to play outside. She saw a big tree with a pretty flower. She thought it would be fun to draw with it.

![loss curve](docs/img/loss_curve.png)

Full results, further samples and failure cases are in [`docs/RESULTS.md`](docs/RESULTS.md).

## Key findings
- A small model trained on modest hardware can produce fluent, grammatical English when the data matches its size.
- Benchmarking showed that 12 CPU threads trained 4–5× faster than all 22, which determined the run plan.
- Tests of the underlying mathematics (causality, attention equivalence, expected initial loss, exact resume) caught bugs early and made the long training run trustworthy.
- Building the full path from data to a deployed API was as instructive as building the model.

## Quick start
```bash
git clone https://github.com/Ahmadbader17/tinygpt && cd tinygpt && uv sync
uv run tinygpt generate --hub Ahmadbader17/tiny-gpt --prompt "Once upon a time, a little fox" --seed 42
```
To run the API with Docker:
```bash
docker build -t tiny-gpt .
docker run -p 7860:7860 -e TINYGPT_HF_REPO=Ahmadbader17/tiny-gpt tiny-gpt
# open http://localhost:7860 (demo) or http://localhost:7860/docs (API)
```

## Limitations
- The model writes simple children's stories. It is not a chat assistant and cannot follow instructions.
- Longer stories lose the plot or repeat themselves; the context window is 256 tokens.
- The model has not been evaluated for bias or safety.

## Further reading
- [`docs/architecture.md`](docs/architecture.md): how the components fit together
- [`DESIGN.md`](DESIGN.md) and [`docs/DECISIONS.md`](docs/DECISIONS.md): design and the decisions made during implementation
- [`docs/LEARNING.md`](docs/LEARNING.md): each stage explained with tensor shapes

## Licence and credits
Code and weights: MIT. Training data: [TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories) by Ronen Eldan and Yuanzhi Li (CDLA-Sharing-1.0).
