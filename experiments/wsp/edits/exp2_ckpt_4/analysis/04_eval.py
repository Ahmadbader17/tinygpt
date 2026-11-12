"""Sandbox passes. Usage: python analysis/04_eval.py <checkpoint dir> <windows|stories> "<purpose>"
Fixed, deterministic dev inputs (seed 0), so damaged and repaired are scored on the same text."""
import json, random, sys
from pathlib import Path
import numpy as np, torch
from tinygpt.tokenizer import BPETokenizer
from tinygpt.wsp.sandbox import Sandbox
from tinygpt.wsp.edits import load_base_params

ckpt, mode, purpose = sys.argv[1], sys.argv[2], sys.argv[3]
ws = Path(".")
tok = BPETokenizer.load(ws / "tokenizer.json")
stories = [s.strip() for s in (ws / "dev" / "stories.txt").read_text(encoding="utf-8").split("\n\n") if s.strip()]
rng = random.Random(0)
pick = rng.sample(range(len(stories)), 400)
if mode == "stories":
    seqs = [([tok.eot_id] + tok.encode(stories[i]))[:256] for i in pick[:64]]
    capture, layers = (), ()
else:
    stream = []
    for i in pick[64:]:
        stream += [tok.eot_id] + tok.encode(stories[i])
    starts = [rng.randrange(0, len(stream) - 256) for _ in range(64)]
    seqs = [stream[s:s + 256] for s in starts]
    capture, layers = ("mlp_out", "attn_out", "resid_post"), (1, 2, 3)
sb = Sandbox.open(ws / "work")
out = sb.read(ws / ckpt, seqs, purpose=purpose, capture=capture, layers=layers)
print(mode, ckpt, "mean_nll", round(out["mean_nll"], 4), "tokens", int(sum(out["tokens"])), "used", sb.used, "remaining", sb.remaining)
if out.get("capture_file"):
    z = np.load(out["capture_file"])
    print("captured", sorted(z.files))
    # diagnostic only (never written into weights): size of each branch, and share of L2 mlp_out along u
    w = load_base_params(ws / "checkpoint")["blocks.2.mlp.proj.weight"].double()
    u = torch.linalg.svd(w, full_matrices=False)[0][:, 0].numpy()
    for k in sorted(z.files):
        a = z[k].astype(np.float64)
        along = a @ u
        print(f"  {k:16s} rms norm {np.sqrt((a**2).sum(-1).mean()):8.3f}  rms along u {np.sqrt((along**2).mean()):8.3f}")
