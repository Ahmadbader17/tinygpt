from pathlib import Path

import pytest

from tinygpt.config import Config, ModelConfig, load_config

CONFIGS = Path(__file__).parent.parent / "configs"


@pytest.mark.parametrize("name", ["tiny", "small"])
def test_shipped_configs_load(name: str) -> None:
    cfg = load_config(CONFIGS / f"{name}.toml")
    assert cfg.name == name
    assert cfg.model.d_model % cfg.model.n_head == 0


def test_small_matches_design() -> None:
    m = load_config(CONFIGS / "small.toml").model
    assert (m.vocab_size, m.context_length, m.d_model, m.n_layer, m.n_head) == (
        4096,
        256,
        256,
        6,
        8,
    )


def test_config_is_frozen() -> None:
    cfg = load_config(CONFIGS / "tiny.toml")
    with pytest.raises(AttributeError):
        cfg.model.d_model = 1  # type: ignore[misc]


def test_heads_must_divide_d_model() -> None:
    with pytest.raises(ValueError):
        ModelConfig(vocab_size=10, context_length=8, d_model=10, n_layer=1, n_head=3)


def test_unknown_key_rejected(tmp_path: Path) -> None:
    bad = tmp_path / "bad.toml"
    bad.write_text(
        'name = "x"\n[model]\nvocab_size=8\ncontext_length=8\nd_model=8\n'
        "n_layer=1\nn_head=2\ndrop_out=0.1\n"
    )
    with pytest.raises(ValueError, match="drop_out"):
        load_config(bad)


def test_type_is_config() -> None:
    assert isinstance(load_config(CONFIGS / "tiny.toml"), Config)
