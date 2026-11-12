"""Step 7: turn the weight statistics of steps 1 to 6 into one confidence per head and class.

Zero forward passes. The statistics are recomputed here from the weights; the mapping from
statistics to confidences is spelled out below, with the judgment calls marked as such.
Writes experiments/wsp/results/exp1_predictions.json.
"""

from __future__ import annotations

import json
import math

import torch
from common import HEADS, REPO, centre, embedding_scale, ln1, load, qkv, tokenizer, w_out

p = load()
tok = tokenizer()
s0 = embedding_scale(p)
wte, wpe = centre(p["wte.weight"]), centre(p["wpe.weight"])
resid = float((wte[1:].pow(2).sum(dim=1).mean() + wpe[16:].pow(2).sum(dim=1).mean()).sqrt())
mean_in = (wte[1:1024].mean(dim=0) + wpe[16:].mean(dim=0)) / s0
frequent = torch.arange(1, 1024)


def sigmoid(x: float) -> float:
    return 1 / (1 + math.exp(-x))


def clip(x: float, low: float, high: float) -> float:
    return max(low, min(high, x))


stats = {}
for layer, head in HEADS:
    wq, wk, wv, bq, bk, bv = qkv(p, layer, head)
    g, b = ln1(p, layer)
    q_pos, k_pos = (wpe * g) @ wq.T / s0, (wpe * g) @ wk.T / s0
    q_tok, k_tok = (wte * g) @ wq.T / s0, (wte * g) @ wk.T / s0
    q_const = wq @ b + bq
    q_mean = q_pos + q_const + q_tok[1:].mean(dim=0)
    # position-only attention to the previous position (step 2)
    T = wpe.shape[0]
    mask = torch.tril(torch.ones(T, T, dtype=torch.bool))
    att = torch.softmax((q_mean @ k_pos.T / math.sqrt(32)).masked_fill(~mask, float("-inf")), -1)
    idx = torch.arange(1, T)
    prev = float(att[idx, idx - 1].mean())
    # extra score of the <|endoftext|> key at position 0 over an average key (step 2)
    typical_key = k_pos[16:].mean(dim=0) + k_tok[1:].mean(dim=0)
    eot0 = float(
        (q_mean[16:] @ (k_pos[0] + k_tok[tok.eot_id] - typical_key)).mean() / math.sqrt(32)
    )
    qp = float((q_tok[1:2000] @ k_pos[16:].T / math.sqrt(32)).std())
    # same-token matching on the direct path (step 2)
    gen = torch.Generator().manual_seed(0)
    pa = torch.randint(1, 4096, (20000,), generator=gen)
    pb = torch.randint(1, 4096, (20000,), generator=gen)
    tt_all = (q_tok[pa] * k_tok[pb]).sum(dim=1) / math.sqrt(32)
    dup_z = float(
        ((q_tok[1:] * k_tok[1:]).sum(dim=1).mean() / math.sqrt(32) - tt_all.mean()) / tt_all.std()
    )
    # content spread of scores (step 5)
    iso = float(((wq * g).T @ (wk * g)).norm()) / math.sqrt(32)
    gen = torch.Generator().manual_seed(1 + layer * 8 + head)
    a = frequent[torch.randint(0, len(frequent), (4000,), generator=gen)]
    bb = frequent[torch.randint(0, len(frequent), (4000,), generator=gen)]
    q_base = q_pos[16:].mean(dim=0) + q_const
    sig_freq = float(((q_base + q_tok[a]) @ k_tok[bb].T / math.sqrt(32)).std(dim=1).mean())
    # what a diffuse head writes (step 6)
    const = float((w_out(p, layer, head) @ (wv @ (g * mean_in) + wv @ b + bv)).norm()) / resid
    stats[(layer, head)] = dict(
        prev=prev, eot0=eot0, qp=qp, dup_z=dup_z, iso=iso, sig_freq=sig_freq, const=const
    )

# JUDGMENT: probability that zeroing the head moves dev loss by less than 0.002 nats.
# Layer-0 heads write tiny vectors (step 6), and a constant shift of the attention output is a
# direction the trained output bias already controls, so removing it costs only at second order.
# The two most diffuse heads with the smallest constant write are judged more likely than not.
INACTIVE_L0 = {0: 0.45, 1: 0.40, 2: 0.58, 3: 0.30, 4: 0.55, 5: 0.25, 6: 0.30, 7: 0.30}
INACTIVE_L1 = {0: 0.05, 1: 0.05, 2: 0.08, 3: 0.05, 4: 0.08, 5: 0.08, 6: 0.05, 7: 0.05}

heads = {}
for (layer, head), s in stats.items():
    # start anchor: needs about 4.5 nats of advantage for 40% attention among ~135 keys.
    # Layer 0 is read exactly and falls far short. Deeper, the direct path shows at most 2.3;
    # MLPs may amplify the start position, which cannot be read, so the confidence stays low.
    start = 0.02 if layer == 0 else clip(0.05 + 0.12 * s["eot0"], 0.03, 0.30)
    # previous token: position embeddings of neighbours are nearly identical (cosine 0.92 at
    # every small offset) and the position-only attention to i-1 is at most 0.04.
    prev = 0.01 if layer == 0 else (0.04 if layer <= 3 else 0.03)
    induction = 0.005 if layer == 0 else (0.02 if layer == 1 else 0.03)
    duplicate = clip(0.02 + 0.06 * max(0.0, s["dup_z"]), 0.02, 0.10)
    # positional: cosine between patterns on different text ~ exp(-sigma^2).
    # Layer 0: sigma read exactly (token terms on frequent tokens, plus query-token x position).
    # Deeper: twice the isotropic value, the ratio seen in layer 0 between real and isotropic.
    if layer == 0:
        sigma2 = s["sig_freq"] ** 2 + s["qp"] ** 2
    else:
        sigma2 = (2 * s["iso"]) ** 2
    cos_est = math.exp(-sigma2)
    diffuse = clip(sigmoid((cos_est - 0.80) / 0.08), 0.02, 0.90)
    positional = diffuse + (1 - diffuse) * 0.8 * start  # a start-anchored head is positional too
    if layer == 0:
        inactive = INACTIVE_L0[head]
    elif layer == 1:
        inactive = INACTIVE_L1[head]
    else:
        inactive = 0.03
    stat_txt = (
        f"pos-only attn to i-1 {s['prev']:.2f}; eot@0 advantage {s['eot0']:+.2f} nats; "
        f"same-token z {s['dup_z']:+.2f}; content sigma est {math.sqrt(sigma2):.2f} "
        f"-> pattern cosine est {cos_est:.2f}; constant write {s['const']:.2f} of residual"
    )
    heads[f"{layer}.{head}"] = {
        "previous_token": {
            "confidence": round(prev, 3),
            "reason": "neighbouring position embeddings are not separable; "
            + f"position-only attention to i-1 is {s['prev']:.2f}",
        },
        "start_anchor": {
            "confidence": round(start, 3),
            "reason": f"direct-path advantage of the position-0 <|endoftext|> key: {s['eot0']:+.2f} nats (about 4.5 needed)",
        },
        "induction": {
            "confidence": round(induction, 3),
            "reason": "no earlier head offers a sharp token-before signal and same-token matching through any earlier head has z < 1",
        },
        "duplicate_token": {
            "confidence": round(duplicate, 3),
            "reason": f"same-token query-key z-score on the direct path is {s['dup_z']:+.2f}",
        },
        "positional": {
            "confidence": round(positional, 3),
            "reason": f"content moves scores by sigma ~ {math.sqrt(sigma2):.2f} nats, so patterns on different text should have cosine ~ {cos_est:.2f}",
        },
        "inactive": {
            "confidence": round(inactive, 3),
            "reason": f"a diffuse head writes a near-constant vector of {s['const']:.2f} of the residual norm"
            + (
                "; judged removable at second order"
                if layer == 0
                else "; too large to remove freely"
            ),
        },
        "stats": stat_txt,
    }

out = {
    "meta": {
        "forward_passes": 0,
        "editor": "Claude Fable 5.2 (main session)",
        "method": "weights only: direct-path query-key and output-value circuits, one-step composition through earlier heads, LayerNorm folded as a fixed gain. MLPs were not read.",
        "positive_at": 0.5,
        "scripts": "experiments/wsp/analysis/exp1/01..07",
    },
    "heads": heads,
}
path = REPO / "experiments" / "wsp" / "results" / "exp1_predictions.json"
path.write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")
print("positives (confidence >= 0.5):")
for cls in (
    "previous_token",
    "start_anchor",
    "induction",
    "duplicate_token",
    "positional",
    "inactive",
):
    pos = [h for h, v in heads.items() if v[cls]["confidence"] >= 0.5]
    top = sorted(heads, key=lambda h: -heads[h][cls]["confidence"])[:6]
    print(
        f"  {cls:16s} {pos}   top: "
        + ", ".join(f"{h} {heads[h][cls]['confidence']:.2f}" for h in top)
    )
