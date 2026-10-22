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
