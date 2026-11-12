"""Free: pooled paired statistics, base vs edit v1, 256 prompts x seeds {0,1}; bootstrap over prompts."""
import json
import numpy as np
from tinygpt.wsp.metrics import has_loop, repeated_ngram_fraction
base_calls = {0: [1, 3, 4, 5], 1: [15, 16, 17, 18]}
edit_calls = {0: [11, 12, 13, 14], 1: [19, 20, 21, 22]}
def load(calls):
    m = {}
    for seed, cs in calls.items():
        for k, c in enumerate(cs):
            for r in json.load(open(f"work/outputs/generations_call{c:03d}.json")):
                t = r["tokens"]
                m[(64 * k + r["prompt"], seed)] = (float(has_loop(t)), float(has_loop(t, 8)), repeated_ngram_fraction(t, 4), len(t), float(r["hit_eot"]))
    return m
A, B = load(base_calls), load(edit_calls)
keys = sorted(A); assert keys == sorted(B) and len(keys) == 512
a = np.array([A[k] for k in keys]).reshape(256, 2, 5).mean(1); b = np.array([B[k] for k in keys]).reshape(256, 2, 5).mean(1)
rng = np.random.default_rng(0); idx = rng.integers(0, 256, size=(10000, 256))
for j, name in enumerate(["loop_rate", "long_loop_rate", "rep4", "length", "eot_rate"]):
    d = b[:, j] - a[:, j]; bs = d[idx].mean(1); lo, hi = np.quantile(bs, [0.025, 0.975])
    print(f"{name:15s} base {a[:, j].mean():.4f} edit {b[:, j].mean():.4f} diff {d.mean():+.4f} 95% CI [{lo:+.4f}, {hi:+.4f}]")
for seed in (0, 1):
    ka = [k for k in keys if k[1] == seed]
    print("seed", seed, "loop", np.mean([A[k][0] for k in ka]), "->", np.mean([B[k][0] for k in ka]),
          "rep4", round(np.mean([A[k][2] for k in ka]), 4), "->", round(np.mean([B[k][2] for k in ka]), 4))
r = [json.loads(l) for l in open("analysis/eval_results.jsonl")]
L = {(x["ckpt"], x["what"]): x for x in r if x["what"].startswith("loss")}
na = np.array(L[("checkpoint", "loss1")]["nll"] + L[("checkpoint", "loss2")]["nll"]); nb = np.array(L[("copy_heads_08", "loss1")]["nll"] + L[("copy_heads_08", "loss2")]["nll"])
nt = np.array(L[("checkpoint", "loss1")]["ntok"] + L[("checkpoint", "loss2")]["ntok"])
i2 = rng.integers(0, 128, size=(10000, 128)); bs = (nb - na)[i2].sum(1) / nt[i2].sum(1)
print(f"loss base {na.sum()/nt.sum():.4f} edit {nb.sum()/nt.sum():.4f} diff {(nb-na).sum()/nt.sum():+.4f} 95% CI [{np.quantile(bs,0.025):+.4f}, {np.quantile(bs,0.975):+.4f}] over 128 stories, {nt.sum()} tokens")
