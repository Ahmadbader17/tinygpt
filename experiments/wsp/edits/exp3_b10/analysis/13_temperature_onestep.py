"""Unedited logits only (pass 2): the same one-step quantity as analysis/12 (sampler probability of completing an
already-seen 4-gram on the model's own texts), as a function of the decoding temperature. Gives the scale that links
the one-step measure to sampled rep-4 (T=1.0 was sampled in pass 4), and the one-step effect of a logit rescale by 0.9
(edit final_gain_090 is exactly T = 0.8/0.9)."""
import json
from pathlib import Path
import numpy as np
meta = json.loads(Path("work/outputs/p02_rows.json").read_text()); rows = np.array(meta["rows"]); B, T = rows.shape
raw = np.load("work/outputs/read_call002.npz")["logits"].astype(np.float64)
def sampler_probs(temp):
    lg = raw / temp; kth = np.sort(lg, -1)[..., -40][..., None]; lg = np.where(lg < kth, -np.inf, lg)
    return np.exp(lg - np.logaddexp.reduce(lg, -1, keepdims=True))
res = {}
for temp in (0.8, 0.8 / 0.9, 1.0):
    p = sampler_probs(temp); tot = 0.0
    for b in range(B):
        x = rows[b]; seen = {}
        for t in range(2, T - 1):
            tri = tuple(x[t - 2 : t + 1])
            if tri in seen and (t + 1 >= meta["prompt_len"][b]) and (t + 1 < meta["valid"][b]): tot += p[b, t, sorted(seen[tri])].sum()
            seen.setdefault(tri, set()).add(x[t + 1])
    res[temp] = tot; print(f"T={temp:.3f}: expected seen-4-gram completions {tot:.1f}  ratio to T=0.8 {tot/res[0.8]:.3f}")
