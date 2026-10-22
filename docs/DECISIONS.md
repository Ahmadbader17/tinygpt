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
