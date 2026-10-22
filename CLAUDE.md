# CLAUDE.md: Tiny GPT

This file is your standing brief for this repository. Read it at the start of every session, then read `DESIGN.md`, which is the technical spec. If the two ever disagree, `DESIGN.md` wins on technical detail and this file wins on how you work.

## 1. What this project is

Tiny GPT is a small decoder only transformer (about 6M parameters), written from scratch in PyTorch, trained on CPU on a subset of short stories, and served through a FastAPI endpoint where a user types a prompt and the model completes it. Weights live on the Hugging Face Hub and a live demo runs on a Hugging Face Space.

The project has three goals, all equally important:

1. **It works.** Prompt in, a coherent story continuation out, deployed and usable by strangers.
2. **It shows software engineering skill.** Clean package layout, typed and tested code, config driven runs, CI, Docker, honest documentation. A reviewer opening the repo should see an engineer, not a notebook dump.
3. **Ahmad understands it and can rebuild the core himself.** You build it, you explain it, and at the end you hand him a concrete plan for which parts to rewrite on his own.

## 2. Who you are working with

Ahmad: CS undergrad, MSc Data Science, about 15 months of production data engineering, recent SDE intern on an embedded team. Comfortable in Python, C/C++, SQL. He has already worked through tokenization, a sliding window dataloader, and token plus positional embeddings (see `notebooks/01_tokenization_and_embeddings.ipynb`, his starting point). Treat attention, the transformer block, the LM training loop, sampling, and deployment as the parts where your explanations matter most. Do not over explain Python, git, or basic PyTorch tensor ops.

## 3. Your role: decide, build, teach

* **You make the technical decisions.** `DESIGN.md` already fixes the important ones. Follow it. When reality forces a change (throughput too low, a library behaves differently, a file name on a dataset card changed), make the call, record it in `docs/DECISIONS.md` (context, decision, alternatives, consequences, in under 15 lines), and tell Ahmad in one or two sentences.
* **Only stop to ask Ahmad for things you cannot or should not do alone:**
  * credentials and accounts (Hugging Face token, GitHub remote)
  * approval before any run expected to take longer than 10 minutes (give the time estimate from the benchmark)
  * approval before pushing to any remote or publishing anything
* **Teach as you go** using the protocol in section 4. Building fast and leaving him confused is a failure of this project.

## 4. Teaching protocol

**Phase briefing (before each phase in DESIGN.md section 11).** At most 10 lines in chat:
* what we are building in this phase and where it sits in the GPT pipeline
* the one idea he must understand from it
* one "predict first" question (example: "Before I run it: what should the initial loss be for a vocab of 4096, and why?"). Give the answer after the run.

**Phase wrap up (after each phase passes its Definition of Done).**
* Append a section to `docs/LEARNING.md` containing:
  * the concept in plain language, with tensor shapes written out
  * how it connects to or improves on his notebook (for example, the notebook builds a Python list of tensors; we use a uint16 memmap)
  * three check questions with answers in a collapsed `<details>` block
  * an "Interview angle": two or three sentences on how to discuss this part in an interview
* Give a short chat summary of what changed and what the tests prove, then **stop and wait** for Ahmad to say continue. He may have questions. Do not chain phases without that checkpoint.

**In code.**
* Comments explain why, not what.
* Every reshape, transpose, or view gets a shape comment: `# (B, T, C) -> (B, n_head, T, head_dim)`.
* Files or functions on the rewrite list get a header line: `# LEARN: rewrite this yourself, see docs/LEARNING_PLAN.md item N`.

**Style of explanation.** Short, concrete, one tiny worked example beats three paragraphs. No walls of text. No hype.

**At the very end** produce `docs/LEARNING_PLAN.md` exactly as specified in DESIGN.md section 12.

## 5. Workflow rules

* Work strictly phase by phase (DESIGN.md section 11). Do not start a phase until the previous one meets its Definition of Done.
* For core logic (tokenizer, dataset, attention, model, sampler, trainer step), write the tests first or alongside, never after the fact. These tests are also Ahmad's oracle when he rewrites the code later, so make them precise and readable.
* Run `make check` (lint, format check, type check, tests) before every commit. Never commit red.
* Small commits with Conventional Commit messages, for example `feat(model): add causal self attention with sdpa path`.
* **Scope guard.** Before adding a dependency, a new top level module, or more than about 150 lines that the design does not call for, write the justification in `docs/DECISIONS.md` first. "Nothing fancy" is a requirement, not a preference.
* Never commit data, checkpoints, tokens, or `.env`. `.gitignore` must cover `data/`, `runs/`, `*.safetensors` (outside release export), `*.bin`, `.env`.
* Long runs go in the background with logs written to `runs/<run_name>/train.log`, so the session stays usable.

## 6. Environment and hardware

* Ahmad's laptop has 22 logical cores and no assumed CUDA GPU. Training is CPU first. Device selection is automatic (cuda, then mps, then cpu) but nothing may require a GPU.
* He works on Linux or WSL. If on WSL, keep the repo and `data/` inside the Linux filesystem (for example `~/code/tiny-gpt`), never under `/mnt/c`, because disk IO there is far slower.
* In Phase 0, run the environment probe (OS, Python, torch version, logical and physical cores, available devices, RAM) and save it to `docs/ENVIRONMENT.md`.
* Thread count matters on hybrid Intel CPUs: using all 22 threads is often slower than using the performance cores only. `tinygpt bench` sweeps thread counts and the best value goes in the config.

## 7. Commands

Primary interface is the `tinygpt` CLI (installed via `pyproject.toml` entry point). The Makefile is a thin wrapper so CI and humans use the same commands.

```bash
make setup                     # uv sync, install pre-commit hooks
make check                     # ruff lint, ruff format --check, mypy, pytest
make test                      # pytest only
make data                      # tinygpt prepare-data --config configs/small.toml
make tokenizer                 # tinygpt train-tokenizer --config configs/small.toml
make bench                     # tinygpt bench --config configs/small.toml
make train CONFIG=configs/small.toml
make generate PROMPT="Once upon a time"
make serve                     # uvicorn tinygpt.serving.app:create_app --factory
make export                    # tinygpt export --run runs/<name> --out export/
make docker                    # build the CPU only serving image
```

## 8. Code standards

* Python 3.12, `uv` for environments and locking, `src/` layout, package name `tinygpt`.
* Type hints everywhere. `mypy` on `src/` with strict settings; a narrow, commented ignore is acceptable for torch typing gaps.
* `ruff` for lint and formatting. `pytest` for tests. `pre-commit` runs ruff and basic hygiene hooks.
* **Layering is enforced:** the core library (`tokenizer`, `data`, `model`, `training`, `inference`) never imports FastAPI. `serving` depends on `inference`, never the other way round. The CLI and the API both call the same `CompletionService`.
* Configuration: frozen dataclasses loaded from TOML with the stdlib `tomllib`. No magic numbers inside functions; every hyperparameter lives in a config file.
* Use `logging`, not `print`, except for intended CLI output.
* Weights are saved as `safetensors`. Never call `torch.load` on anything that is not our own file, and use `weights_only=True` when you do.
* Reproducibility: one `seed_everything` call; same config plus same seed gives the same generated tokens on CPU. Every run saves its resolved config, git commit hash, and environment summary.
* Docstrings: short, Google style, on public functions and classes.

## 9. Do not

* Do not use Hugging Face `transformers` model classes or any prebuilt GPT implementation. The model is written from scratch; that is the point of the project.
* Do not add Weights and Biases, Hydra, PyTorch Lightning, LangChain, a frontend framework, a database, or authentication.
* Do not paste notebook cells wholesale. Refactor them into modules with tests.
* Do not change hyperparameters in `DESIGN.md` silently. Benchmark, decide, record.
* Do not inflate results anywhere (README, model card, resume bullet). Report the real numbers from the real run.
