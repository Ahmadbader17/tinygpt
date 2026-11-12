"""Static check of editor rules 1, 2, 4 and 5 on Python source.

Rule 1: no gradients, backward calls, optimizers or training loops.
Rule 2: no fitting weights to data (no least squares, no closed form regression).
Rule 4: the model runs only through the sandbox, never by constructing it directly.
Rule 5: edit code never names sealed data.
A scan is a tripwire, not a proof: it catches the named patterns, nothing cleverer.
"""

from __future__ import annotations

import ast

FORBIDDEN_MODULES = (
    "torch.optim",
    "sklearn",
    "scipy.optimize",
    "tinygpt.training",
    "tinygpt.inference",
    "tinygpt.model",
    "tinygpt.wsp.scoring",
    "tinygpt.wsp.activations",
    "tinygpt.wsp.evaluate",
)
FORBIDDEN_NAMES = {
    "backward",
    "requires_grad",
    "requires_grad_",
    "enable_grad",
    "set_grad_enabled",
    "autograd",
    "optim",
    "Optimizer",
    "AdamW",
    "SGD",
    "Adam",
    "lstsq",
    "pinv",
    "solve",
    "cholesky_solve",
    "linear_regression",
    # running the model anywhere but through the sandbox (rule 4)
    "GPT",
    "from_pretrained",
    "generate_all",
    "generate_batch",
    "forward_capture",
    "forward_with_attention",
    "story_nll",
    "sequence_nll",
    "scaled_dot_product_attention",
}
FORBIDDEN_STRINGS = ("sealed", "GPT4-valid", "TinyStoriesV2")


def scan_source(source: str) -> list[str]:
    """Violations in one file's source, as readable strings (empty means clean)."""
    problems: list[str] = []
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.startswith(FORBIDDEN_MODULES):
                    problems.append(f"line {node.lineno}: import {alias.name}")
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            if module.startswith(FORBIDDEN_MODULES):
                problems.append(f"line {node.lineno}: from {module} import ...")
            for alias in node.names:
                if alias.name in FORBIDDEN_NAMES:
                    problems.append(f"line {node.lineno}: imports {alias.name}")
        elif isinstance(node, ast.Attribute) and node.attr in FORBIDDEN_NAMES:
            problems.append(f"line {node.lineno}: uses .{node.attr}")
        elif isinstance(node, ast.Name) and node.id in FORBIDDEN_NAMES:
            problems.append(f"line {node.lineno}: uses {node.id}")
        elif isinstance(node, ast.Constant) and isinstance(node.value, str):
            for marker in FORBIDDEN_STRINGS:
                if marker in node.value:
                    problems.append(f"line {node.lineno}: mentions '{marker}'")
    return problems
