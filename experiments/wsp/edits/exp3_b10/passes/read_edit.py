"""Read an edited checkpoint on the same 64 real dev stories as pass 3 and compare per-token loss (paired).
usage: read_edit.py <edit name> <purpose text file>"""
import json, sys
import numpy as np
sys.path.insert(0, "passes")
from common import *
name = sys.argv[1]; purpose = Path(sys.argv[2]).read_text().strip()
meta = json.loads((WS / "work/outputs/p03_rows.json").read_text()); rows = meta["rows"]
sb = sandbox()
out = sb.read(WS / "edited" / name, rows, purpose=purpose, capture=("token_loss",))
base = np.load(WS / "work/outputs/read_call003.npz")["token_loss"]; new = np.load(out["capture_file"])["token_loss"]
d = new - base
# classes as in analysis/05: ctx3 = trigram seen before; copy = the next token repeats a 4-gram
R = np.array(rows); B, T = R.shape; ctx3 = np.zeros((B, T - 1), bool); copy = np.zeros((B, T - 1), bool)
for b in range(B):
    seen = {}
    for t in range(2, T - 1):
        tri = tuple(R[b, t - 2 : t + 1])
        if tri in seen: ctx3[b, t] = True; copy[b, t] = R[b, t + 1] in seen[tri]
        seen.setdefault(tri, []).append(R[b, t + 1])
per_story = d.mean(1); rng = np.random.default_rng(0)
boot = [per_story[rng.integers(0, B, B)].mean() for _ in range(5000)]
print(json.dumps({"call": out["call"], "mean_nll": out["mean_nll"], "base_mean_nll": float(base.mean()),
    "delta": float(d.mean()), "delta_ci95": [float(np.quantile(boot, .025)), float(np.quantile(boot, .975))],
    "delta_at_copy_positions": float(d[copy].mean()), "delta_at_ctx3_not_copy": float(d[ctx3 & ~copy].mean()),
    "delta_elsewhere": float(d[~ctx3].mean()), "share_of_total_from_copy_positions": float(d[copy].sum() / d.sum()),
    "first_half_delta": float(d[:, :127].mean()), "second_half_delta": float(d[:, 127:].mean())}, indent=1))
print("used", sb.used, "remaining", sb.remaining)
