# CPU-only serving image. Also what the Hugging Face Docker Space builds.
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy

# pinned uv binary; uv.lock pins CPU-only torch, so no CUDA wheels end up in the image
COPY --from=ghcr.io/astral-sh/uv:0.13.5 /uv /usr/local/bin/uv

# Spaces run containers as uid 1000; a non-root user is good practice anywhere
RUN useradd --create-home --uid 1000 user
USER user
WORKDIR /home/user/app

COPY --chown=user pyproject.toml uv.lock README.md LICENSE ./
COPY --chown=user src ./src
RUN uv sync --frozen --no-dev --no-cache  # no cache: it would be baked into the layer (~1 GB)

ENV PATH="/home/user/app/.venv/bin:$PATH" \
    TINYGPT_MODEL_DIR=/home/user/model \
    TINYGPT_NUM_THREADS=2 \
    TINYGPT_MAX_CONCURRENCY=1

EXPOSE 7860
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s \
  CMD python -c "import urllib.request as u; u.urlopen('http://localhost:7860/health')"

# weights come from TINYGPT_HF_REPO (Hub) or a mounted TINYGPT_MODEL_DIR
CMD ["uvicorn", "tinygpt.serving.app:create_app", "--factory", "--host", "0.0.0.0", "--port", "7860"]
