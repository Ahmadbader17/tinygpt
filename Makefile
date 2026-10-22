CONFIG ?= configs/small.toml
PROMPT ?= Once upon a time
RUN ?= runs/latest

.PHONY: setup check test data tokenizer bench train generate serve export docker

setup:
	uv sync --group compare --group notebook
	uv run pre-commit install

check:
	uv run ruff check .
	uv run ruff format --check .
	uv run mypy
	uv run pytest

test:
	uv run pytest

data:
	uv run tinygpt prepare-data --config configs/small.toml

tokenizer:
	uv run tinygpt train-tokenizer --config configs/small.toml

bench:
	uv run tinygpt bench --config configs/small.toml

train:
	uv run tinygpt train --config $(CONFIG)

generate:
	uv run tinygpt generate --run $(RUN) --prompt "$(PROMPT)"

serve:
	uv run uvicorn tinygpt.serving.app:create_app --factory

export:
	uv run tinygpt export --run $(RUN) --out export/

docker:
	docker build -t tiny-gpt .
