"""Per-story loss, so confidence intervals can resample whole stories."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch
import torch.nn.functional as F
from numpy.typing import NDArray
from torch import Tensor

from tinygpt.model import GPT

_IGNORE = -100


def penalise_seen(logits: Tensor, idx: Tensor, eot_id: int, penalty: float) -> Tensor:
    """Apply a decoding-time repetition penalty to teacher-forced logits.

    At position t every token already in `idx[:, : t + 1]` has its logit divided by `penalty`
    if positive and multiplied otherwise, exactly as `generation.next_token` does while
    sampling. This turns the penalised decoder into a next-token distribution whose loss on
    real text can be measured. `<|endoftext|>` is never penalised: generation never has it in
    the context either, and penalising it would only stop stories from ending.
    """
    vocab = logits.size(-1)
    seen = torch.zeros(idx.shape[0], idx.shape[1], vocab, dtype=torch.uint8)
    seen.scatter_(2, idx.unsqueeze(-1), 1)  # (B, T, vocab): token at position t
    seen = seen.cummax(dim=1).values.bool()  # ... or at any earlier position
    seen[:, :, eot_id] = False
    penalised = torch.where(logits > 0, logits / penalty, logits * penalty)
    return torch.where(seen, penalised, logits)


@torch.inference_mode()
def story_nll(
    model: GPT,
    stories: Sequence[Sequence[int]],
    eot_id: int,
    batch_size: int = 32,
    repetition_penalty: float = 1.0,
) -> tuple[NDArray[np.float64], NDArray[np.int64]]:
    """Summed negative log likelihood (nats) and predicted-token count of each story.

    A story is scored as `<|endoftext|> story <|endoftext|>`: the model predicts every story
    token and the final stop token, starting from the same context a story start had in
    training. Stories longer than the context are cut into consecutive windows that do not
    overlap; a later window starts without earlier context, as random training windows did.
    Mean loss is `nll.sum() / count.sum()`. With `repetition_penalty` other than 1 the loss
    is that of the penalised decoder (see `penalise_seen`).
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
        if repetition_penalty != 1.0:
            logits = penalise_seen(logits, xb, eot_id, repetition_penalty)
        # (B, T, vocab) -> (B*T, vocab); per-token loss back to (B, T), padding scores 0
        per_token = F.cross_entropy(
            logits.view(-1, logits.size(-1)), yb.view(-1), ignore_index=_IGNORE, reduction="none"
        ).view(len(chunk), width)
        row_nll = per_token.double().sum(dim=1).numpy()
        for r, (i, _, ys) in enumerate(chunk):
            nll[i] += row_nll[r]
            count[i] += len(ys)
    return nll, count


@torch.inference_mode()
def sequence_nll(
    model: GPT, sequences: Sequence[Sequence[int]], batch_size: int = 32
) -> tuple[NDArray[np.float64], NDArray[np.int64]]:
    """Summed NLL of each sequence's tokens 2..T given their prefixes, with no framing.

    Used by the budgeted `read` call, where the editor chooses the token sequences.
    Sequences longer than the context are rejected rather than windowed.
    """
    if model.training:
        raise ValueError("put the model in eval mode before scoring")
    ctx = model.cfg.context_length
    for seq in sequences:
        if not 2 <= len(seq) <= ctx:
            raise ValueError(f"sequence length {len(seq)} must be between 2 and {ctx}")
    order = sorted(range(len(sequences)), key=lambda i: -len(sequences[i]))
    nll = np.zeros(len(sequences), dtype=np.float64)
    count = np.zeros(len(sequences), dtype=np.int64)
    for start in range(0, len(order), batch_size):
        rows = order[start : start + batch_size]
        width = max(len(sequences[i]) - 1 for i in rows)
        xb = torch.zeros((len(rows), width), dtype=torch.long)
        yb = torch.full((len(rows), width), _IGNORE, dtype=torch.long)
        for r, i in enumerate(rows):
            seq = sequences[i]
            xb[r, : len(seq) - 1] = torch.tensor(seq[:-1], dtype=torch.long)
            yb[r, : len(seq) - 1] = torch.tensor(seq[1:], dtype=torch.long)
        logits, _ = model(xb)
        per_token = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            yb.reshape(-1),
            ignore_index=_IGNORE,
            reduction="none",
        ).reshape(len(rows), width)
        row_nll = per_token.double().sum(dim=1).numpy()
        for r, i in enumerate(rows):
            nll[i] = row_nll[r]
            count[i] = len(sequences[i]) - 1
    return nll, count


@torch.inference_mode()
def continuation_nll(
    model: GPT,
    prompts: Sequence[Sequence[int]],
    continuations: Sequence[Sequence[int]],
    batch_size: int = 32,
) -> tuple[NDArray[np.float64], NDArray[np.int64]]:
    """Summed NLL of each continuation given its prompt, and its token count.

    Used for the judge loss: generations of an edited model scored by the unedited model.
    The caller appends the stop token to a continuation that ended by itself.
    """
    if model.training:
        raise ValueError("put the model in eval mode before scoring")
    if len(prompts) != len(continuations):
        raise ValueError("one continuation per prompt")
    nll = np.zeros(len(prompts), dtype=np.float64)
    count = np.asarray([len(c) for c in continuations], dtype=np.int64)
    order = [i for i in sorted(range(len(prompts)), key=lambda i: -len(continuations[i]))]
    order = [i for i in order if count[i] > 0]
    order.sort(key=lambda i: -(len(prompts[i]) + len(continuations[i])))
    for start in range(0, len(order), batch_size):
        rows = order[start : start + batch_size]
        width = max(len(prompts[i]) + len(continuations[i]) for i in rows) - 1
        xb = torch.zeros((len(rows), width), dtype=torch.long)
        yb = torch.full((len(rows), width), _IGNORE, dtype=torch.long)
        for r, i in enumerate(rows):
            seq = [*prompts[i], *continuations[i]]
            xb[r, : len(seq) - 1] = torch.tensor(seq[:-1], dtype=torch.long)
            first = len(prompts[i]) - 1  # the input position that predicts the first new token
            yb[r, first : len(seq) - 1] = torch.tensor(continuations[i], dtype=torch.long)
        logits, _ = model(xb)
        per_token = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            yb.reshape(-1),
            ignore_index=_IGNORE,
            reduction="none",
        ).reshape(len(rows), width)
        row_nll = per_token.double().sum(dim=1).numpy()
        for r, i in enumerate(rows):
            nll[i] = row_nll[r]
    return nll, count
