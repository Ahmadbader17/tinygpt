# Decisions

## 0001: Notebook source and repo location
- **Context:** `DESIGN.md` expects `notebooks/01_tokenization_and_embeddings.ipynb`. The file supplied was `Byte_Pair_Encoding.ipynb` in the Desktop `tinyGPT` folder. The Desktop folder is on `/mnt/c` (slow IO under WSL).
- **Decision:** copy the notebook unchanged to the expected name; develop in `~/code/tiny-gpt`. The Desktop folder keeps only the briefing docs and the original.
- **Alternatives:** work under `/mnt/c` (rejected, CLAUDE.md section 6).
- **Consequences:** notebooks are excluded from ruff so the original stays byte-identical.

## 0002: Tooling without sudo
- **Context:** `uv` and `make` were not installed and sudo needs a password.
- **Decision:** install `uv` with its installer and extract the Ubuntu `make` .deb into `~/.local/bin`.
- **Consequences:** no system changes; CI uses `astral-sh/setup-uv` and system `make`.

## 0003: CPU-only torch via uv index
- **Decision:** `[tool.uv.sources] torch` points at the PyTorch CPU wheel index, so local, CI and Docker installs match and stay small.

## 0004: tiktoken as an optional dependency group
- **Context:** DESIGN D2 wants a tiktoken GPT-2 adapter for one comparison test; tiktoken is not in the D9 dependency list and fetches its vocab over the network.
- **Decision:** put it in a `compare` dependency group (not runtime deps). The test skips if the package or network is missing.
- **Alternatives:** make it a runtime dependency (rejected: the serving image never needs it).
- **Consequences:** CI runs `uv sync --group compare` so the comparison test runs there.

## 0005: notebook dependency group
- **Context:** Ahmad wants to step through each phase in VS Code notebooks.
- **Decision:** `notebook` group (ipykernel, nbclient, nbformat); not part of runtime or CI.
- **Consequences:** `make setup` installs it; walkthrough notebooks live in `notebooks/NN_*.ipynb` and are committed with outputs.

## 0006: real-run budget from the CPU benchmark
- **Context:** `tinygpt bench` on the Core Ultra 9 185H (22 threads, WSL2) measured training throughput of the small config (batch 32 x 256 tokens): 4 threads ~2.1k tok/s, 10 to 12 threads ~2.5k to 2.9k tok/s, 16 threads ~0.9k, 22 threads ~0.6k. Using all threads is 4 to 5 times slower, as the risk table in DESIGN.md predicted for hybrid CPUs.
- **Decision:** `num_threads = 12`. `max_steps = 7500` (61.4M tokens, one pass over the 60.8M-token training set). Planned for 2.5k tok/s sustained, so about 6.8 h; 2.8k gives 6.1 h. The run also gets `--max-hours` as a safety stop.
- **Why one epoch:** 61M tokens is about 10 tokens per parameter, and no data repeats, so validation loss tracks generalisation directly. More steps would mean a second epoch over the same stories.
- **Alternatives:** more steps (rejected: repeats data, and the cosine schedule must finish inside the sleep window); a smaller context or fewer layers (not needed, the budget fits).
- **Consequences:** if the machine sleeps or throttles hard, `--resume` continues from `checkpoints/last`; time-limited stops leave the learning rate not fully decayed.

## 0007: model bundle and `export` pulled forward from Phase 8
- **Context:** the API (Phase 7) needs a self-contained model directory; DESIGN puts `tinygpt export` in Phase 8.
- **Decision:** `export_bundle` and `tinygpt export` land now (about 30 lines): `model.safetensors`, `model_config.json`, `tokenizer.json`, `model_info.json` (steps, tokens, validation loss and perplexity, tokenizer hash). The server loads only this directory, so it never depends on `runs/` or `data/`. Phase 8 adds the model card and Hub upload on top.
- **Alternatives:** serve straight from a run directory plus `data/tokenizer` (rejected: the Docker image and the Space would need the training folders).
- **Consequences:** `TINYGPT_HF_REPO` (download from the Hub) is not in `Settings` yet; Phase 8 adds it with `huggingface_hub`.
- **Also:** `generate()` now uses `torch.inference_mode()` (DESIGN D7) and `CompletionService` takes an optional `queue_timeout` so a full server answers 429 instead of queueing forever.

## 0008: publishing layout, and what is not yet verified
- **Decision:** the model repo holds the export bundle plus `deploy/MODEL_CARD.md` as `README.md`; the Docker Space holds `Dockerfile`, `pyproject.toml`, `uv.lock`, `LICENSE`, `src/` and `deploy/space_README.md` as `README.md` (front matter: `sdk: docker`, `app_port: 7860`). The Space downloads weights at startup from `TINYGPT_HF_REPO`, which `tinygpt publish --upload` sets as a Space variable. `publish` is a dry run unless `--upload` is given and reads `HF_TOKEN` from the environment only.
- **Deviation:** the config file is `model_config.json`, not `config.json` as DESIGN D8 says; our `ModelConfig` is not a `transformers` config, so a generic name would mislead tools that look for one.
- **Not verified locally:** Docker is not installed in this WSL. The install and serve steps were replicated in a clean folder (`uv sync --frozen --no-dev` then `uvicorn`), but `docker build` itself has not run, and the Space has not been built.
- **Defaults for a small free host:** `TINYGPT_NUM_THREADS=2`, `TINYGPT_MAX_CONCURRENCY=1` (a free Space has 2 vCPUs). Space latency is unmeasured.
- **Image size:** `matplotlib` is a runtime dependency (plots only, DESIGN D9) so the image carries it; torch CPU dominates the size anyway (about 1 GB `.venv`).
- **Licence caveat:** code and weights are MIT; the data is CDLA-Sharing-1.0. The model card says so and points to the dataset card. Whether the data licence imposes anything on trained weights is a legal question to confirm, not something decided here.

## 0010: Weight Space Programming study as a separate research harness
- **Context:** Ahmad asked for a small replication of "Sparks of Weight-Space Programming" (Takehi et al., 2 Nov 2026) on the published checkpoint, on branch `exp/weight_space_programming`. DESIGN.md does not call for it, so the scope guard applies.
- **Decision:** harness code lives in a new package `src/tinygpt/wsp/` (typed, under mypy strict and pytest like the rest); the plan, edit scripts, journals and result files live in `experiments/wsp/`; the report goes to `docs/WSP_REPORT.md`. `mypy` now also checks `experiments/`. No new dependency.
- **Layering:** `tinygpt.wsp` imports `model`, `tokenizer` and `data`; nothing in the core library, the CLI or `serving` imports it. The original checkpoint and `main` are never modified; edited weights are written under `runs/wsp/` (gitignored) and are rebuilt by scripts.
- **Why a second sampler (`wsp/generation.py`):** the final evaluation generates about 70,000 stories. The repo's `generate` has no KV cache and stops a batch only when every row ends, so it would take many hours; the cached sampler does 1,500 stories in under 2 minutes. The evaluation sampler computes the same distribution (temperature, top-k) with a KV cache and common random numbers, and a test pins it to `GPT.forward` and to greedy `generate`.
- **Alternatives:** a separate repo (rejected: the study must run from a clean clone of this one); scripts only, no package (rejected: the metrics and the budgeted forward function need tests and types).
- **Consequences:** CLAUDE.md's teaching protocol (phase briefings, `docs/LEARNING.md` entries) is not applied to this study; it follows the phases and stop points of Ahmad's study brief instead. `experiments/wsp/PLAN.md` is the spec for it.
- **Editors' scripts are evidence (added 2026-11-12):** the analysis and edit scripts written by an editor (`experiments/wsp/analysis/`, `experiments/wsp/edits/exp*/`, `experiments/wsp/logs/`) are archived as they were run and are excluded from ruff and mypy. They stay under the static rule scan (`tests/test_wsp_edit_rules.py`), which is the check that matters for them. Harness code is linted and typed as before.
- **Moved off the main line (2026-11-19):** the study (harness, scripts, results, report) was removed from branch `simplify`; it lives unchanged on `exp/weight_space_programming` and tag `wsp-freeze`, which its reproduction steps already pin.
