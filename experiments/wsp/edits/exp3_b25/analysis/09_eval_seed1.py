"""Second paired sample: same prompts 0-255, seed 1. usage: 09_eval_seed1.py <ckpt_dir>  (4 passes)"""
import json, sys
sys.path.insert(0, "analysis")
from common import ws, PROMPTS, SAMPLE, sandbox
ckpt = sys.argv[1]; tag = ckpt.rstrip("/").split("/")[-1]
sb = sandbox()
for k in range(4):
    out = sb.generate(ws / ckpt, PROMPTS[64 * k:64 * k + 64], seeds=(1,), decode=SAMPLE,
        purpose=f"[{tag}] replication of the paired comparison on prompts {64*k}-{64*k+63} with seed 1: is the loop-rate "
                "drop of edit v1 seen with seed 0 (0.250 -> 0.199, about 1.4 standard errors) reproduced on fresh randomness?")
    res = {k2: out[k2] for k2 in ("call", "loop_rate", "rep4", "mean_length", "eot_rate", "generations_file")}
    res.update(ckpt=tag, what=f"gen{k+1}_seed1")
    print(json.dumps(res), "used", sb.used, "remaining", sb.remaining, flush=True)
    with open("analysis/eval_results.jsonl", "a") as f:
        f.write(json.dumps(res) + "\n")
