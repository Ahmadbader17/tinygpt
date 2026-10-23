import threading
import time
from collections.abc import Iterator
from pathlib import Path

import pytest
import torch
from fastapi.testclient import TestClient

from tinygpt.config import load_config
from tinygpt.inference import CompletionService
from tinygpt.model import GPT
from tinygpt.serving.app import create_app
from tinygpt.serving.settings import Settings
from tinygpt.tokenizer import BPETokenizer

CFG = load_config(Path(__file__).parent.parent / "configs" / "tiny.toml")
BODY = {"prompt": "Once upon a time", "max_new_tokens": 8, "seed": 1}


def make_service(tok: BPETokenizer, queue_timeout: float | None = 0.2) -> CompletionService:
    torch.manual_seed(0)
    return CompletionService(GPT(CFG.model), tok, max_concurrency=1, queue_timeout=queue_timeout)


@pytest.fixture
def client(tiny_tokenizer: BPETokenizer) -> Iterator[TestClient]:
    app = create_app(Settings(), loader=lambda _: make_service(tiny_tokenizer))
    with TestClient(app) as c:
        wait_until_loaded(app)
        yield c


def wait_until_loaded(app: object) -> None:
    deadline = time.time() + 5
    while app.state.service is None and time.time() < deadline:  # type: ignore[attr-defined]
        time.sleep(0.01)


def test_health(client: TestClient) -> None:
    assert client.get("/health").json() == {"status": "ok"}


def test_model_endpoint_reports_parameters_and_config(client: TestClient) -> None:
    body = client.get("/v1/model").json()
    assert body["parameters"] == GPT(CFG.model).num_parameters()
    assert body["config"]["vocab_size"] == CFG.model.vocab_size
    assert body["version"]


def test_completion_returns_contract_fields(client: TestClient) -> None:
    res = client.post("/v1/completions", json=BODY)
    assert res.status_code == 200
    body = res.json()
    assert body["prompt"] == BODY["prompt"]
    assert 0 <= body["tokens_generated"] <= 8
    assert body["stop_reason"] in ("eot", "length")
    assert body["seed"] == 1
    assert body["latency_ms"] >= 0


def test_same_seed_gives_same_completion(client: TestClient) -> None:
    a = client.post("/v1/completions", json=BODY).json()["completion"]
    b = client.post("/v1/completions", json=BODY).json()["completion"]
    assert a == b


def test_unseeded_response_seed_reproduces_it(client: TestClient) -> None:
    first = client.post("/v1/completions", json={**BODY, "seed": None}).json()
    again = client.post("/v1/completions", json={**BODY, "seed": first["seed"]}).json()
    assert again["completion"] == first["completion"]


@pytest.mark.parametrize(
    "bad",
    [
        {"prompt": ""},
        {"prompt": "x" * 501},
        {"max_new_tokens": 0},
        {"max_new_tokens": 257},
        {"temperature": -0.1},
        {"temperature": 2.1},
        {"top_k": 0},
        {"top_k": CFG.model.vocab_size + 1},  # checked against the loaded model
        {"seed": -1},
        {"prompt": 5},
    ],
)
def test_invalid_input_is_422(client: TestClient, bad: dict[str, object]) -> None:
    assert client.post("/v1/completions", json={**BODY, **bad}).status_code == 422


def test_missing_prompt_is_422(client: TestClient) -> None:
    assert client.post("/v1/completions", json={}).status_code == 422


def test_503_while_model_is_loading(tiny_tokenizer: BPETokenizer) -> None:
    release = threading.Event()

    def slow_loader(_: Settings) -> CompletionService:
        release.wait(5)
        return make_service(tiny_tokenizer)

    app = create_app(Settings(), loader=slow_loader)
    with TestClient(app) as c:
        assert c.get("/health").status_code == 200  # alive although not ready
        assert c.post("/v1/completions", json=BODY).status_code == 503
        release.set()
        wait_until_loaded(app)
        assert c.post("/v1/completions", json=BODY).status_code == 200


def test_503_with_reason_when_loading_fails() -> None:
    def broken(_: Settings) -> CompletionService:
        raise FileNotFoundError("no weights here")

    app = create_app(Settings(), loader=broken)
    with TestClient(app) as c:
        time.sleep(0.2)
        res = c.post("/v1/completions", json=BODY)
        assert res.status_code == 503
        assert "no weights here" in res.json()["detail"]
        assert c.get("/health").status_code == 200


def test_429_when_all_slots_are_busy(
    tiny_tokenizer: BPETokenizer, monkeypatch: pytest.MonkeyPatch
) -> None:
    entered, release = threading.Event(), threading.Event()
    import tinygpt.inference.service as service_module

    real_generate = service_module.generate

    def blocking_generate(*args, **kwargs):  # type: ignore[no-untyped-def]
        entered.set()
        release.wait(5)
        return real_generate(*args, **kwargs)

    monkeypatch.setattr(service_module, "generate", blocking_generate)
    app = create_app(Settings(), loader=lambda _: make_service(tiny_tokenizer, queue_timeout=0.1))
    with TestClient(app) as c:
        wait_until_loaded(app)
        first: list[int] = []
        t = threading.Thread(
            target=lambda: first.append(c.post("/v1/completions", json=BODY).status_code)
        )
        t.start()
        assert entered.wait(5)  # the only slot is now held by a running generation
        assert c.post("/v1/completions", json=BODY).status_code == 429
        release.set()
        t.join(5)
        assert first == [200]  # the request that held the slot still succeeds


def test_index_page_is_served(client: TestClient) -> None:
    res = client.get("/")
    assert res.status_code == 200
    assert "Tiny GPT" in res.text
    assert "/v1/completions" in res.text


def test_openapi_docs_are_available(client: TestClient) -> None:
    assert client.get("/docs").status_code == 200
    assert "/v1/completions" in client.get("/openapi.json").json()["paths"]


def test_settings_from_env() -> None:
    s = Settings.from_env(
        {"TINYGPT_MODEL_DIR": "/m", "TINYGPT_MAX_CONCURRENCY": "3", "TINYGPT_NUM_THREADS": "6"}
    )
    assert (s.model_dir, s.max_concurrency, s.num_threads) == (Path("/m"), 3, 6)
    assert Settings.from_env({}) == Settings()


def test_settings_reads_hf_repo() -> None:
    assert Settings.from_env({"TINYGPT_HF_REPO": "me/tiny-gpt"}).hf_repo == "me/tiny-gpt"
    assert Settings.from_env({"TINYGPT_HF_REPO": ""}).hf_repo is None


def test_fetch_bundle_downloads_only_when_a_repo_is_configured(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import huggingface_hub

    from tinygpt.serving.app import fetch_bundle

    calls: list[dict[str, object]] = []
    monkeypatch.setattr(huggingface_hub, "snapshot_download", lambda **kw: calls.append(kw))
    fetch_bundle(Settings(model_dir=tmp_path))
    assert calls == []  # local bundle: no network
    fetch_bundle(Settings(model_dir=tmp_path, hf_repo="me/tiny-gpt"))
    assert calls == [{"repo_id": "me/tiny-gpt", "local_dir": tmp_path}]
