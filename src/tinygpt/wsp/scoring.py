"""Per-story loss, so confidence intervals can resample whole stories."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch
import torch.nn.functional as F
from numpy.typing import NDArray

from tinygpt.model import GPT

_IGNORE = -100


@torch.inference_mode()
def story_nll(
    model: GPT, stories: Sequence[Sequence[int]], eot_id: int, batch_size: int = 32
) -> tuple[NDArray[np.float64], NDArray[np.int64]]:
    """Summed negative log likelihood (nats) and predicted-token count of each story.

    A story is scored as `<|endoftext|> story <|endoftext|>`: the model predicts every story
    token and the final stop token, starting from the same context a story start had in
    training. Stories longer than the context are cut into consecutive windows that do not
    overlap; a later window starts without earlier context, as random training windows did.
    Mean loss is `nll.sum() / count.sum()`.
    """
    if model.training:
        raise ValueError("put the model in eval mode before scoring")
    ctx = model.cfg.context_length
    windows: list[tuple[int, list[int], list[int]]] = []  # (story index, inputs, targets)
    for i, story in enumerate(stories):
        seq = [eot_id, *story, eot_id]
        for start in range(0, len(seq) - 1, ctx):
            x = seq[start : start + ctx]
            y = seq[start + 1 : start + ctx + 1]
            windows.append((i, x[: len(y)], y))
    windows.sort(key=lambda w: (-len(w[1]), w[0]))  # similar lengths together: less padding
    nll = np.zeros(len(stories), dtype=np.float64)
    count = np.zeros(len(stories), dtype=np.int64)
    for start in range(0, len(windows), batch_size):
        chunk = windows[start : start + batch_size]
        width = len(chunk[0][1])
        xb = torch.full((len(chunk), width), eot_id, dtype=torch.long)
        yb = torch.full((len(chunk), width), _IGNORE, dtype=torch.long)
        for r, (_, xs, ys) in enumerate(chunk):
            # padding sits on the right, so with causal attention it cannot change real positions
            xb[r, : len(xs)] = torch.tensor(xs)
            yb[r, : len(ys)] = torch.tensor(ys)
        logits, _ = model(xb)  # (B, T, vocab)
        # (B, T, vocab) -> (B*T, vocab); per-token loss back to (B, T), padding scores 0
        per_token = F.cross_entropy(
            logits.view(-1, logits.size(-1)), yb.view(-1), ignore_index=_IGNORE, reduction="none"
        ).view(len(chunk), width)
        row_nll = per_token.double().sum(dim=1).numpy()
        for r, (i, _, ys) in enumerate(chunk):
            nll[i] += row_nll[r]
            count[i] += len(ys)
    return nll, count
