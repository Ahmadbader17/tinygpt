"""Plan and perform the upload of the model bundle (model repo) and the app (Docker Space).

Nothing is uploaded unless `upload=True`; the default only lists what would be sent.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

BUNDLE_FILES = ("model.safetensors", "model_config.json", "tokenizer.json", "model_info.json")
SPACE_FILES = ("Dockerfile", "pyproject.toml", "uv.lock", "LICENSE")


@dataclass(frozen=True)
class UploadPlan:
    repo_id: str
    repo_type: str  # "model" or "space"
    files: dict[str, Path]  # path inside the repo -> local file


def _walk(root: Path) -> Iterator[Path]:
    for path in sorted(root.rglob("*")):
        if path.is_file() and "__pycache__" not in path.parts and path.suffix != ".pyc":
            yield path


def plan_model_upload(bundle_dir: Path, model_card: Path, repo_id: str) -> UploadPlan:
    files = {name: bundle_dir / name for name in BUNDLE_FILES}
    files["README.md"] = model_card
    return _checked(UploadPlan(repo_id, "model", files))


def plan_space_upload(project_root: Path, space_readme: Path, repo_id: str) -> UploadPlan:
    files = {name: project_root / name for name in SPACE_FILES}
    files["README.md"] = space_readme  # the Space's README carries the Docker front matter
    src = project_root / "src"
    files.update({str(p.relative_to(project_root)): p for p in _walk(src)})
    return _checked(UploadPlan(repo_id, "space", files))


def _checked(plan: UploadPlan) -> UploadPlan:
    missing = [name for name, path in plan.files.items() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing files for {plan.repo_type} upload: {missing}")
    return plan


def upload(plan: UploadPlan, api: Any, variables: dict[str, str] | None = None) -> str:
    """One atomic commit per repo. `api` is a huggingface_hub.HfApi (injectable for tests)."""
    from huggingface_hub import CommitOperationAdd

    kwargs: dict[str, Any] = {"space_sdk": "docker"} if plan.repo_type == "space" else {}
    api.create_repo(plan.repo_id, repo_type=plan.repo_type, exist_ok=True, **kwargs)
    ops = [CommitOperationAdd(dest, str(src)) for dest, src in plan.files.items()]
    api.create_commit(
        plan.repo_id, ops, repo_type=plan.repo_type, commit_message=f"Upload {plan.repo_type} files"
    )
    for key, value in (variables or {}).items():
        api.add_space_variable(plan.repo_id, key, value)
    return f"https://huggingface.co/{'spaces/' if plan.repo_type == 'space' else ''}{plan.repo_id}"
