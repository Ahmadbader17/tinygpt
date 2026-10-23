import threading
from pathlib import Path

import pytest
import torch

from tinygpt.config import load_config
from tinygpt.inference import CompletionRequest, CompletionService
from tinygpt.model import GPT
from tinygpt.tokenizer import BPETokenizer

CFG = load_config(Path(__file__).parent.parent / "configs" / "tiny.toml")


@pytest.fixture
def service(tiny_tokenizer: BPETokenizer) -> CompletionService:
    torch.manual_seed(0)
    return CompletionService(GPT(CFG.model), tiny_tokenizer)


def test_returns_completion_without_the_prompt(service: CompletionService) -> None:
    res = service.complete(CompletionRequest("Once upon a time", max_new_tokens=10, seed=1))
    assert res.prompt_tokens == len(service.tokenizer.encode("Once upon a time"))
    assert res.completion_tokens <= 10
    assert res.finish_reason in ("stop", "length")
    assert res.latency_s > 0


def test_same_seed_same_text_and_unseeded_reports_its_seed(service: CompletionService) -> None:
    req = CompletionRequest("The fox", max_new_tokens=15, seed=7)
    assert service.complete(req).text == service.complete(req).text
    first = service.complete(CompletionRequest("The fox", max_new_tokens=15))
    again = service.complete(CompletionRequest("The fox", max_new_tokens=15, seed=first.seed))
    assert again.text == first.text  # the returned seed reproduces the sample


def test_length_finish_reason_and_token_count(service: CompletionService) -> None:
    res = service.complete(CompletionRequest("The fox", max_new_tokens=5, temperature=0))
    if res.finish_reason == "length":
        assert res.completion_tokens == 5


def test_stop_token_ends_generation_and_is_not_in_text(
    service: CompletionService, tiny_tokenizer: BPETokenizer
) -> None:
    req = CompletionRequest("The fox", max_new_tokens=30, temperature=0)
    first = tiny_tokenizer.encode(service.complete(req).text)[0]  # greedy first new token

    class StopAtFirst:
        eot_id = first
        vocab_size = tiny_tokenizer.vocab_size

        def encode(self, text: str) -> list[int]:
            return tiny_tokenizer.encode(text)

        def decode(self, ids: list[int]) -> str:
            return tiny_tokenizer.decode(ids)

    stopper = CompletionService(service.model, StopAtFirst())  # type: ignore[arg-type]
    res = stopper.complete(req)
    assert res.finish_reason == "stop"
    assert res.completion_tokens == 0
    assert res.text == ""


def test_empty_prompt_starts_from_eot(service: CompletionService) -> None:
    res = service.complete(CompletionRequest("", max_new_tokens=5, seed=0))
    assert res.prompt_tokens == 1


def test_concurrent_requests_all_complete_with_same_results(service: CompletionService) -> None:
    capped = CompletionService(service.model, service.tokenizer, max_concurrency=1)
    req = CompletionRequest("A cat", max_new_tokens=8, seed=3)
    expected = capped.complete(req).text
    results: list[str] = []
    threads = [
        threading.Thread(target=lambda: results.append(capped.complete(req).text)) for _ in range(4)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert results == [expected] * 4


def test_rejects_bad_concurrency(tiny_tokenizer: BPETokenizer) -> None:
    with pytest.raises(ValueError):
        CompletionService(GPT(CFG.model), tiny_tokenizer, max_concurrency=0)


def test_from_run_loads_best_checkpoint(tmp_path: Path, tiny_tokenizer: BPETokenizer) -> None:
    torch.manual_seed(0)
    model = GPT(CFG.model)
    model.save_pretrained(tmp_path / "run" / "checkpoints" / "best")
    tiny_tokenizer.save(tmp_path / "tok.json")
    svc = CompletionService.from_run(tmp_path / "run", tmp_path / "tok.json")
    req = CompletionRequest("The fox", max_new_tokens=10, seed=2)
    assert svc.complete(req).text == CompletionService(model, tiny_tokenizer).complete(req).text
