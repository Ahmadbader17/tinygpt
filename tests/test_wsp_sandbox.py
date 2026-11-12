from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from tinygpt.config import load_config
from tinygpt.model import GPT
from tinygpt.tokenizer import BPETokenizer
from tinygpt.wsp.activations import CAPTURE_KINDS, forward_with_attention
from tinygpt.wsp.generation import DecodeConfig, generate_all
from tinygpt.wsp.sandbox import (
    MAX_READ_LENGTH,
    REPO_ROOT,
    BudgetExceeded,
    LedgerError,
    Sandbox,
    SealedAccessError,
    guard_path,
    init_budget,
)
from tinygpt.wsp.scoring import sequence_nll

CFG = load_config(Path(__file__).parent.parent / "configs" / "tiny.toml").model


@pytest.fixture
def checkpoint(tmp_path: Path, tiny_tokenizer: BPETokenizer) -> tuple[Path, Path]:
    """A tiny random checkpoint with its tokenizer, in a temporary folder."""
    torch.manual_seed(0)
    ckpt = tmp_path / "base"
    GPT(CFG).save_pretrained(ckpt)
    tok = tmp_path / "tokenizer.json"
    tiny_tokenizer.save(tok)
    (ckpt / "tokenizer.json").write_bytes(tok.read_bytes())
    return ckpt, tok


def test_budget_refuses_the_call_after_the_limit(
    tmp_path: Path, checkpoint: tuple[Path, Path]
) -> None:
    ckpt, tok = checkpoint
    sb = Sandbox(tmp_path / "w", role="editor", limit=2, tokenizer_path=tok)
    sb.read(ckpt, [[5, 9, 12]], purpose="first")
    sb.read(ckpt, [[5, 9, 12]], purpose="second")
    with pytest.raises(BudgetExceeded):
        sb.read(ckpt, [[5, 9, 12]], purpose="third, one too many")
    assert sb.used == 2
    assert sb.remaining == 0
    assert sb.ledger.records()[-1]["status"] == "refused"


def test_budget_survives_a_new_process(tmp_path: Path, checkpoint: tuple[Path, Path]) -> None:
    ckpt, tok = checkpoint
    sb = Sandbox(tmp_path / "w", role="editor", limit=1, tokenizer_path=tok)
    sb.read(ckpt, [[5, 9, 12]], purpose="only pass")
    again = Sandbox(tmp_path / "w", role="editor", limit=1, tokenizer_path=tok)
    assert again.used == 1
    with pytest.raises(BudgetExceeded):
        again.read(ckpt, [[5, 9, 12]], purpose="retry after restart")


def test_harness_role_is_logged_but_unlimited(
    tmp_path: Path, checkpoint: tuple[Path, Path]
) -> None:
    ckpt, tok = checkpoint
    sb = Sandbox(tmp_path / "w", role="harness", limit=None, tokenizer_path=tok)
    for _ in range(3):
        sb.read(ckpt, [[5, 9, 12]], purpose="harness check")
    assert sb.used == 3 and sb.remaining is None


def test_every_call_needs_a_purpose(tmp_path: Path, checkpoint: tuple[Path, Path]) -> None:
    ckpt, tok = checkpoint
    sb = Sandbox(tmp_path / "w", role="editor", limit=5, tokenizer_path=tok)
    with pytest.raises(ValueError, match="purpose"):
        sb.read(ckpt, [[5, 9, 12]], purpose="   ")
    assert sb.used == 0


def test_ledger_detects_an_edited_record(tmp_path: Path, checkpoint: tuple[Path, Path]) -> None:
    ckpt, tok = checkpoint
    sb = Sandbox(tmp_path / "w", role="editor", limit=5, tokenizer_path=tok)
    sb.read(ckpt, [[5, 9, 12]], purpose="honest purpose")
    sb.read(ckpt, [[5, 9, 12]], purpose="second")
    ledger = tmp_path / "w" / "ledger.jsonl"
    ledger.write_text(ledger.read_text().replace("honest purpose", "rewritten"))
    with pytest.raises(LedgerError):
        Sandbox(tmp_path / "w", role="editor", limit=5, tokenizer_path=tok)


def test_ledger_detects_a_deleted_record(tmp_path: Path, checkpoint: tuple[Path, Path]) -> None:
    ckpt, tok = checkpoint
    sb = Sandbox(tmp_path / "w", role="editor", limit=5, tokenizer_path=tok)
    for i in range(3):
        sb.read(ckpt, [[5, 9, 12]], purpose=f"call {i}")
    ledger = tmp_path / "w" / "ledger.jsonl"
    lines = ledger.read_text().splitlines()
    ledger.write_text("\n".join(lines[:1] + lines[2:]) + "\n")
    with pytest.raises(LedgerError):
        Sandbox(tmp_path / "w", role="editor", limit=5, tokenizer_path=tok)


def test_sealed_locations_are_refused() -> None:
    with pytest.raises(SealedAccessError):
        guard_path(REPO_ROOT / "data" / "sealed" / "stories.txt")
    with pytest.raises(SealedAccessError):
        guard_path(REPO_ROOT / "runs" / "wsp" / "sealed" / "faults.json")
    with pytest.raises(SealedAccessError):
        guard_path(Path("/tmp/x/TinyStoriesV2-GPT4-valid.txt"))
    assert guard_path(REPO_ROOT / "data" / "raw" / "val.txt").name == "val.txt"


def test_sandbox_cannot_be_opened_inside_sealed_data() -> None:
    with pytest.raises(SealedAccessError):
        Sandbox(REPO_ROOT / "data" / "sealed" / "editor", role="editor", limit=1)


def test_loaded_weights_do_not_track_gradients(
    tmp_path: Path, checkpoint: tuple[Path, Path]
) -> None:
    ckpt, tok = checkpoint
    sb = Sandbox(tmp_path / "w", role="editor", limit=3, tokenizer_path=tok)
    sb.read(ckpt, [[5, 9, 12]], purpose="grad check")
    model, _ = sb._load(ckpt)
    assert all(not p.requires_grad for p in model.parameters())


def test_read_matches_a_direct_computation(tmp_path: Path, checkpoint: tuple[Path, Path]) -> None:
    ckpt, tok = checkpoint
    seq = [5, 9, 12, 40, 41, 3, 7]
    sb = Sandbox(tmp_path / "w", role="editor", limit=3, tokenizer_path=tok)
    got = sb.read(ckpt, [seq], purpose="direct check")
    model = GPT.from_pretrained(ckpt).eval()
    logits, _ = model(torch.tensor([seq[:-1]]))
    expected = torch.nn.functional.cross_entropy(logits[0], torch.tensor(seq[1:]), reduction="sum")
    assert abs(got["nll"][0] - expected.item()) < 1e-4
    assert got["tokens"] == [len(seq) - 1]


def test_read_rejects_bad_input(tmp_path: Path, checkpoint: tuple[Path, Path]) -> None:
    ckpt, tok = checkpoint
    sb = Sandbox(tmp_path / "w", role="editor", limit=5, tokenizer_path=tok)
    with pytest.raises(ValueError, match="sequences"):
        sb.read(ckpt, [[1, 2]] * 65, purpose="too many rows")
    with pytest.raises(ValueError, match="vocabulary"):
        sb.read(ckpt, [[1, 512]], purpose="bad token")
    with pytest.raises(ValueError, match="tokens"):
        sb.read(ckpt, [[1] * (MAX_READ_LENGTH + 1)], purpose="too long")
    with pytest.raises(ValueError, match="unknown capture"):
        sb.read(ckpt, [[1, 2, 3]], purpose="bad kind", capture=("gradients",), layers=(0,))
    with pytest.raises(ValueError, match="equal length"):
        sb.read(ckpt, [[1, 2, 3], [1, 2]], purpose="ragged", capture=("logits",))
    assert sb.used == 0  # invalid input is rejected before a pass is spent


def test_attention_capture_is_causal_and_normalised(
    tmp_path: Path, checkpoint: tuple[Path, Path]
) -> None:
    ckpt, tok = checkpoint
    sb = Sandbox(tmp_path / "w", role="editor", limit=3, tokenizer_path=tok)
    seqs = [[5, 9, 12, 40, 41], [7, 7, 8, 8, 9]]
    got = sb.read(ckpt, seqs, purpose="attention check", attention_layers=(0, 1))
    with np.load(got["capture_file"]) as data:
        layer0 = data["attn0"]
    assert layer0.shape == (2, CFG.n_head, 5, 5)
    assert np.allclose(layer0.sum(axis=-1), 1.0, atol=1e-5)
    assert np.all(np.triu(layer0[0, 0], k=1) == 0)


def test_forward_with_attention_matches_the_model(checkpoint: tuple[Path, Path]) -> None:
    ckpt, _ = checkpoint
    model = GPT.from_pretrained(ckpt).eval()
    idx = torch.randint(0, CFG.vocab_size, (3, 12), generator=torch.Generator().manual_seed(2))
    with torch.inference_mode():
        ref, _ = model(idx)
        got, captured = forward_with_attention(model, idx, layers=(0, 1))
    assert torch.allclose(ref, got, atol=1e-5)
    assert set(captured) == {0, 1}


def test_generate_matches_generate_all_and_writes_every_row(
    tmp_path: Path, checkpoint: tuple[Path, Path]
) -> None:
    import json

    ckpt, tok = checkpoint
    sb = Sandbox(tmp_path / "w", role="editor", limit=3, tokenizer_path=tok)
    decode = DecodeConfig(max_new_tokens=8, temperature=0.8, top_k=40)
    prompts = [[5, 9, 12], [7, 7]]
    got = sb.generate(ckpt, prompts, seeds=(0, 1), decode=decode, purpose="generation check")
    model = GPT.from_pretrained(ckpt).eval()
    ref = generate_all(model, prompts, (0, 1), decode, sb.tokenizer.eot_id)
    rows = json.loads(Path(got["generations_file"]).read_text())
    assert len(rows) == 4
    for row in rows:
        assert row["tokens"] == ref[row["prompt"]][row["seed"]].tokens
    assert got["mean_length"] == np.mean([len(r["tokens"]) for r in rows])


def test_generate_enforces_row_and_length_limits(
    tmp_path: Path, checkpoint: tuple[Path, Path]
) -> None:
    ckpt, tok = checkpoint
    sb = Sandbox(tmp_path / "w", role="editor", limit=5, tokenizer_path=tok)
    with pytest.raises(ValueError, match="continuations"):
        sb.generate(
            ckpt,
            [[5, 9]] * 33,
            seeds=(0, 1),
            decode=DecodeConfig(max_new_tokens=4, temperature=0),
            purpose="too many rows",
        )
    with pytest.raises(ValueError, match="max_new_tokens"):
        sb.generate(
            ckpt, [[5, 9]], seeds=None, decode=DecodeConfig(max_new_tokens=201), purpose="too long"
        )
    assert sb.used == 0


def test_sequence_nll_rejects_out_of_range_lengths(checkpoint: tuple[Path, Path]) -> None:
    ckpt, _ = checkpoint
    model = GPT.from_pretrained(ckpt).eval()
    with pytest.raises(ValueError):
        sequence_nll(model, [[5]])


def test_captured_pieces_add_up_to_the_residual_stream(
    tmp_path: Path, checkpoint: tuple[Path, Path]
) -> None:
    ckpt, tok = checkpoint
    sb = Sandbox(tmp_path / "w", role="editor", limit=3, tokenizer_path=tok)
    seqs = [[5, 9, 12, 40, 41, 7], [7, 7, 8, 8, 9, 3]]
    got = sb.read(
        ckpt,
        seqs,
        purpose="capture check",
        capture=(*CAPTURE_KINDS, "logits", "token_loss"),
        layers=(0, 1),
    )
    model = GPT.from_pretrained(ckpt).eval()
    with np.load(got["capture_file"]) as data:
        a = {k: torch.from_numpy(data[k]) for k in data.files}
    bias = model.blocks[1].attn.proj.bias
    # the heads' separate writes, plus the bias, are the attention sublayer's output
    assert torch.allclose(a["head_out1"].sum(dim=1) + bias, a["attn_out1"], atol=1e-5)
    assert torch.allclose(a["resid_pre1"] + a["attn_out1"], a["resid_mid1"], atol=1e-5)
    assert torch.allclose(a["resid_mid1"] + a["mlp_out1"], a["resid_post1"], atol=1e-5)
    assert torch.allclose(a["resid_post0"], a["resid_pre1"])
    assert a["mlp_hidden0"].shape == (2, 6, 4 * CFG.d_model)
    with torch.inference_mode():
        ref, _ = model(torch.tensor(seqs))
    assert torch.allclose(a["logits"], ref, atol=1e-5)
    # per-token losses sum to the reported sequence loss
    summed = a["token_loss"].double().sum(dim=1)
    assert torch.allclose(summed, torch.tensor(got["nll"], dtype=torch.float64), atol=1e-4)


def test_editor_budget_comes_from_the_harness_record(
    tmp_path: Path, checkpoint: tuple[Path, Path]
) -> None:
    ckpt, tok = checkpoint
    work = tmp_path / "ws" / "work"
    genesis = init_budget(work, limit=2)
    # the editor asks for more; the harness record wins
    sb = Sandbox(work, role="someone-else", limit=1000, tokenizer_path=tok)
    assert sb.limit == 2 and sb.role == "editor"
    sb.read(ckpt, [[5, 9, 12]], purpose="one")
    sb.read(ckpt, [[5, 9, 12]], purpose="two")
    with pytest.raises(BudgetExceeded):
        Sandbox.open(work, tokenizer_path=tok).read(ckpt, [[5, 9, 12]], purpose="three")
    assert sb.ledger.records()[0]["hash"] == genesis
    with pytest.raises(LedgerError):
        init_budget(work, limit=50)  # a budget cannot be issued twice


def test_open_needs_a_harness_budget(tmp_path: Path, checkpoint: tuple[Path, Path]) -> None:
    _, tok = checkpoint
    with pytest.raises(LedgerError, match="no budget record"):
        Sandbox.open(tmp_path / "plain", tokenizer_path=tok)
