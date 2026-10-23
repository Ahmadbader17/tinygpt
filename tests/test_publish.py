from pathlib import Path
from typing import Any

import pytest

from tinygpt.publish import BUNDLE_FILES, SPACE_FILES, plan_model_upload, plan_space_upload, upload

ROOT = Path(__file__).parent.parent


class FakeApi:
    """Records calls instead of talking to the Hub."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, Any]] = []

    def create_repo(self, repo_id: str, **kwargs: Any) -> None:
        self.calls.append(("create_repo", (repo_id, kwargs)))

    def create_commit(self, repo_id: str, operations: list[Any], **kwargs: Any) -> None:
        self.calls.append(
            ("create_commit", (repo_id, [op.path_in_repo for op in operations], kwargs))
        )

    def add_space_variable(self, repo_id: str, key: str, value: str) -> None:
        self.calls.append(("add_space_variable", (repo_id, key, value)))


@pytest.fixture
def bundle(tmp_path: Path) -> Path:
    for name in BUNDLE_FILES:
        (tmp_path / name).write_text("x")
    return tmp_path


def test_model_plan_contains_bundle_and_card_as_readme(bundle: Path) -> None:
    plan = plan_model_upload(bundle, ROOT / "deploy" / "MODEL_CARD.md", "me/tiny-gpt")
    assert set(plan.files) == {*BUNDLE_FILES, "README.md"}
    assert plan.files["README.md"].name == "MODEL_CARD.md"


def test_space_plan_has_dockerfile_source_and_space_readme() -> None:
    plan = plan_space_upload(ROOT, ROOT / "deploy" / "space_README.md", "me/demo")
    assert set(SPACE_FILES) <= set(plan.files)
    assert plan.files["README.md"].name == "space_README.md"
    assert "src/tinygpt/serving/static/index.html" in plan.files
    assert not any("__pycache__" in dest or dest.endswith(".pyc") for dest in plan.files)
    assert not any(dest.startswith(("data", "runs", "export", "tests")) for dest in plan.files)


def test_missing_bundle_file_is_an_error(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="model.safetensors"):
        plan_model_upload(tmp_path, ROOT / "deploy" / "MODEL_CARD.md", "me/x")


def test_upload_is_one_commit_and_sets_space_variable(bundle: Path) -> None:
    api = FakeApi()
    plan = plan_space_upload(ROOT, ROOT / "deploy" / "space_README.md", "me/demo")
    url = upload(plan, api, {"TINYGPT_HF_REPO": "me/tiny-gpt"})
    names = [name for name, _ in api.calls]
    assert names == ["create_repo", "create_commit", "add_space_variable"]
    assert api.calls[0][1][1]["space_sdk"] == "docker"
    assert url == "https://huggingface.co/spaces/me/demo"


def test_model_upload_does_not_request_docker_sdk(bundle: Path) -> None:
    api = FakeApi()
    plan = plan_model_upload(bundle, ROOT / "deploy" / "MODEL_CARD.md", "me/tiny-gpt")
    assert upload(plan, api) == "https://huggingface.co/me/tiny-gpt"
    assert "space_sdk" not in api.calls[0][1][1]


def test_space_readme_front_matter_matches_the_dockerfile() -> None:
    readme = (ROOT / "deploy" / "space_README.md").read_text()
    dockerfile = (ROOT / "Dockerfile").read_text()
    assert "sdk: docker" in readme and "app_port: 7860" in readme
    assert '--port", "7860"' in dockerfile and "EXPOSE 7860" in dockerfile
    assert "USER user" in dockerfile  # not root
