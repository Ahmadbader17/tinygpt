# Decisions

## 0001: Notebook source and repo location
- **Context:** `DESIGN.md` expects `notebooks/01_tokenization_and_embeddings.ipynb`. The supplied file was `Byte_Pair_Encoding.ipynb` in the Desktop `tinyGPT` folder, which is on `/mnt/c` (slow IO under WSL).
- **Decision:** The notebook was copied unchanged to the expected name, and development takes place in `~/code/tiny-gpt`. The Desktop folder retains only the briefing documents and the original.
- **Alternatives:** Working under `/mnt/c` (rejected, `CLAUDE.md` section 6).
- **Consequences:** Notebooks are excluded from ruff so the original stays byte-identical.

## 0002: Tooling without sudo
- **Context:** `uv` and `make` were not installed, and sudo requires a password.
- **Decision:** `uv` was installed with its installer, and the Ubuntu `make` .deb was extracted into `~/.local/bin`.
- **Consequences:** No system changes. CI uses `astral-sh/setup-uv` and system `make`.

## 0003: CPU-only torch via uv index
- **Decision:** `[tool.uv.sources] torch` points at the PyTorch CPU wheel index, so local, CI and Docker installs match and stay small.

## 0004: tiktoken as an optional dependency group
- **Context:** DESIGN D2 specifies a tiktoken GPT-2 adapter for one comparison test. tiktoken is not in the D9 dependency list and fetches its vocabulary over the network.
- **Decision:** tiktoken is placed in a `compare` dependency group, outside the runtime dependencies. The test skips if the package or network is unavailable.
- **Alternatives:** A runtime dependency (rejected: the serving image never needs it).
- **Consequences:** CI runs `uv sync --group compare`, so the comparison test runs there.

## 0005: notebook dependency group
- **Context:** The owner wants to step through each phase in VS Code notebooks.
- **Decision:** A `notebook` group (ipykernel, nbclient, nbformat), excluded from runtime and CI.
- **Consequences:** `make setup` installs it. Walkthrough notebooks live in `notebooks/NN_*.ipynb` and are committed with outputs.

## 0006: Real-run budget from the CPU benchmark
- **Context:** `tinygpt bench` on the Core Ultra 9 185H (22 threads, WSL2) measured training throughput for the small config (batch 32 × 256 tokens): about 2.1k tok/s with 4 threads, 2.5k to 2.9k tok/s with 10 to 12 threads, about 0.9k with 16 and about 0.6k with 22. Using all threads is 4 to 5 times slower, as the `DESIGN.md` risk table predicted for hybrid CPUs.
- **Decision:** `num_threads = 12` and `max_steps = 7500` (61.4M tokens, one pass over the 60.8M-token training set). At a sustained 2.5k tok/s this takes about 6.8 h (6.1 h at 2.8k). The run also has a `--max-hours` safety stop.
- **Rationale for one epoch:** 61M tokens is about 10 tokens per parameter, and no data repeats, so validation loss tracks generalisation directly. Further steps would repeat the same stories.
- **Alternatives:** More steps (rejected: data repeats, and the cosine schedule must finish within the sleep window); a smaller context or fewer layers (unnecessary, the budget fits).
- **Consequences:** If the machine sleeps or throttles heavily, `--resume` continues from `checkpoints/last`. A time-limited stop leaves the learning rate not fully decayed.

## 0007: Model bundle and `export` moved forward from Phase 8
- **Context:** The API (Phase 7) needs a self-contained model directory; DESIGN places `tinygpt export` in Phase 8.
- **Decision:** `export_bundle` and `tinygpt export` (about 30 lines) were implemented early. The bundle holds `model.safetensors`, `model_config.json`, `tokenizer.json` and `model_info.json` (steps, tokens, validation loss and perplexity, tokenizer hash). The server loads only this directory, so it never depends on `runs/` or `data/`. Phase 8 adds the model card and Hub upload.
- **Alternatives:** Serving from a run directory plus `data/tokenizer` (rejected: the Docker image and the Space would then need the training folders).
- **Consequences:** `TINYGPT_HF_REPO` (download from the Hub) was not yet part of `Settings`; Phase 8 adds it with `huggingface_hub`.
- **Also:** `generate()` now uses `torch.inference_mode()` (DESIGN D7), and `CompletionService` accepts an optional `queue_timeout`, so a saturated server answers 429 instead of queueing indefinitely.

## 0008: Publishing layout and unverified items
- **Decision:** The model repository holds the export bundle plus `deploy/MODEL_CARD.md` as `README.md`. The Docker Space holds `Dockerfile`, `pyproject.toml`, `uv.lock`, `LICENSE`, `src/` and `deploy/space_README.md` as `README.md` (front matter: `sdk: docker`, `app_port: 7860`). The Space downloads weights at startup from `TINYGPT_HF_REPO`, which `tinygpt publish --upload` sets as a Space variable. `publish` is a dry run unless `--upload` is given, and reads `HF_TOKEN` from the environment only.
- **Deviation:** The configuration file is `model_config.json`, not `config.json` as DESIGN D8 states. The `ModelConfig` is not a `transformers` config, so a generic name would mislead tools that look for one.
- **Not verified locally:** Docker is not installed in this WSL environment. The install and serve steps were replicated in a clean folder (`uv sync --frozen --no-dev`, then `uvicorn`), but `docker build` itself has not run and the Space has not been built.
- **Defaults for a small free host:** `TINYGPT_NUM_THREADS=2` and `TINYGPT_MAX_CONCURRENCY=1` (a free Space has 2 vCPUs). Space latency is unmeasured.
- **Image size:** `matplotlib` is a runtime dependency (plots only, DESIGN D9), so the image includes it. CPU torch dominates the size regardless (about 1 GB `.venv`).
- **Licence caveat:** Code and weights are MIT; the data is CDLA-Sharing-1.0. The model card states both and points to the dataset card. Whether the data licence places obligations on trained weights is a legal question that remains to be confirmed.
