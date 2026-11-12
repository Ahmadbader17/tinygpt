"""On identical contexts (the unedited model's pass-1 generations), compare the standard sampler's probability
(T=0.8, top-k 40) of emitting a token that completes an already-seen 4-gram: unedited (pass 2) vs an edited model.
usage: 12_sampler_copy_prob.py <edited read npz>"""
import json, sys
from pathlib import Path
import numpy as np
meta = json.loads(Path("work/outputs/p02_rows.json").read_text()); rows = np.array(meta["rows"]); B, T = rows.shape
def sampler_probs(lg):
    lg = lg.astype(np.float64) / 0.8
    kth = np.sort(lg, -1)[..., -40][..., None]
    lg = np.where(lg < kth, -np.inf, lg)
    return np.exp(lg - np.logaddexp.reduce(lg, -1, keepdims=True))
base = sampler_probs(np.load("work/outputs/read_call002.npz")["logits"]); edit = sampler_probs(np.load(sys.argv[1])["logits"])
pb, pe, ctx, took, story = [], [], [], [], []
allb = np.zeros(B); alle = np.zeros(B); npos = np.zeros(B)
for b in range(B):
    x = rows[b]; seen = {}
    for t in range(T - 1):
        ok = (t + 1 >= meta["prompt_len"][b]) and (t + 1 < meta["valid"][b])
        if t >= 2:
            tri = tuple(x[t - 2 : t + 1])
            if ok: npos[b] += 1
            if tri in seen and ok:
                c = sorted(seen[tri])
                pb.append(base[b, t, c].sum()); pe.append(edit[b, t, c].sum()); took.append(x[t + 1] in seen[tri]); story.append(b)
                allb[b] += pb[-1]; alle[b] += pe[-1]
            seen.setdefault(tri, set()).add(x[t + 1])
pb, pe, took, story = map(np.array, (pb, pe, took, story))
print(f"positions with a repeatable trigram context: {len(pb)} of {int(npos.sum())}")
print(f"sampler prob of completing a seen 4-gram there: unedited {pb.mean():.4f}  edited {pe.mean():.4f}  ratio {pe.mean()/pb.mean():.3f}")
print(f"expected 4-gram-repeat tokens per generated token: unedited {allb.sum()/npos.sum():.5f}  edited {alle.sum()/npos.sum():.5f}  (observed in these texts {took.sum()/npos.sum():.5f})")
rng = np.random.default_rng(0); idx = rng.integers(0, B, (10000, B)); r = alle[idx].sum(1) / allb[idx].sum(1)
print(f"ratio edited/unedited, bootstrap over texts: {alle.sum()/allb.sum():.3f}  95% [{np.quantile(r,.025):.3f}, {np.quantile(r,.975):.3f}]")
for lo, hi in [(0, .1), (.1, .5), (.5, .9), (.9, 1.01)]:
    m = (pb >= lo) & (pb < hi)
    print(f"   unedited prob in [{lo},{hi}): n={m.sum():4d}  unedited {pb[m].mean():.3f}  edited {pe[m].mean():.3f}")
