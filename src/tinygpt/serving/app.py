"""App factory: build the app, load the model in the background during startup."""

from __future__ import annotations

import json
import logging
import threading
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from typing import Any

import torch
from fastapi import FastAPI

from tinygpt import __version__
from tinygpt.inference import CompletionService
from tinygpt.inference.service import INFO_FILE
from tinygpt.serving.routes import router
from tinygpt.serving.settings import Settings

log = logging.getLogger(__name__)

ServiceLoader = Callable[[Settings], CompletionService]


def fetch_bundle(settings: Settings) -> None:
    """Download the model bundle from the Hub into model_dir when a repo is configured."""
    if settings.hf_repo:
        from huggingface_hub import snapshot_download

        log.info("downloading %s from the Hugging Face Hub", settings.hf_repo)
        snapshot_download(repo_id=settings.hf_repo, local_dir=settings.model_dir)


def load_from_disk(settings: Settings) -> CompletionService:
    if settings.num_threads:
        torch.set_num_threads(settings.num_threads)
    fetch_bundle(settings)
    return CompletionService.from_bundle(
        settings.model_dir, settings.max_concurrency, settings.queue_timeout
    )


def create_app(settings: Settings | None = None, loader: ServiceLoader = load_from_disk) -> FastAPI:
    """`loader` is injectable so tests can serve a tiny random model without any files."""
    settings = settings or Settings.from_env()

    def load(app: FastAPI) -> None:
        try:
            app.state.service = loader(settings)
            info_path = settings.model_dir / INFO_FILE
            if info_path.exists():
                app.state.info = json.loads(info_path.read_text(encoding="utf-8"))
            log.info("model loaded")
        except Exception as exc:  # report through /v1/* as 503 instead of crashing the process
            log.exception("model failed to load")
            app.state.load_error = str(exc)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        # load in a thread so /health answers immediately while the weights are read
        thread = threading.Thread(target=load, args=(app,), daemon=True)
        thread.start()
        yield
        thread.join(timeout=0)

    app = FastAPI(title="Tiny GPT", version=__version__, lifespan=lifespan)
    state: Any = app.state
    state.service, state.load_error, state.info = None, None, {}
    state.settings = settings
    app.include_router(router)
    return app
