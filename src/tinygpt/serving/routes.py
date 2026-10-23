"""HTTP routes. They only translate between HTTP and CompletionService."""

from __future__ import annotations

import dataclasses
from pathlib import Path

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import FileResponse

from tinygpt import __version__
from tinygpt.inference import CompletionRequest, CompletionService, ServiceBusy
from tinygpt.serving.schemas import CompletionIn, CompletionOut, ModelInfo

router = APIRouter()
STATIC = Path(__file__).parent / "static"


def get_service(request: Request) -> CompletionService:
    """The loaded service, or 503 while the model loads (or if loading failed)."""
    service: CompletionService | None = request.app.state.service
    if service is None:
        error = request.app.state.load_error
        raise HTTPException(503, f"model failed to load: {error}" if error else "model is loading")
    return service


@router.get("/", include_in_schema=False)
def index() -> FileResponse:
    return FileResponse(STATIC / "index.html")


@router.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}  # liveness only: true even while the model is still loading


@router.get("/v1/model", response_model=ModelInfo)
def model_info(request: Request) -> ModelInfo:
    service = get_service(request)
    info = request.app.state.info
    return ModelInfo(
        name="tiny-gpt",
        version=__version__,
        parameters=service.model.num_parameters(),
        config=dataclasses.asdict(service.model.cfg),
        training=info,
    )


# a plain `def` on purpose: FastAPI runs it in a worker thread, so generation
# never blocks the event loop
@router.post("/v1/completions", response_model=CompletionOut)
def completions(body: CompletionIn, request: Request) -> CompletionOut:
    service = get_service(request)
    vocab = service.model.cfg.vocab_size
    if body.top_k is not None and body.top_k > vocab:
        raise HTTPException(422, f"top_k must be at most the vocabulary size ({vocab})")
    try:
        res = service.complete(
            CompletionRequest(
                body.prompt, body.max_new_tokens, body.temperature, body.top_k, body.seed
            )
        )
    except ServiceBusy:
        raise HTTPException(
            429, "the server is busy generating other requests; try again shortly"
        ) from None
    return CompletionOut(
        prompt=body.prompt,
        completion=res.text,
        tokens_generated=res.completion_tokens,
        stop_reason="eot" if res.finish_reason == "stop" else "length",
        seed=res.seed,
        latency_ms=round(res.latency_s * 1000),
    )
