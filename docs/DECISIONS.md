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
