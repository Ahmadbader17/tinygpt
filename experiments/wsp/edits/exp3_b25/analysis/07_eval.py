"""Evaluation passes for one checkpoint: usage  07_eval.py <ckpt_dir> <what> [what...]
what in: gen1 gen2 gen3 gen4 (prompts 0-63, 64-127, 128-191, 192-255; seed 0, T=0.8, top-k 40)
         loss1 loss2 (dev stories 100-163 and 164-227, first 256 tokens each)
         tf (teacher-forced token losses on the fixed pass-2 sequences: 32 base generations + 32 real stories)
Each item is one forward pass. Results are appended to analysis/eval_results.jsonl."""
import json, sys
sys.path.insert(0, "analysis")
from common import ws, tok, PROMPTS, SAMPLE, sandbox
ckpt = sys.argv[1]; tag = ckpt.rstrip("/").split("/")[-1]
sb = sandbox()
stories = open("dev/stories.txt").read().split("\n\n")
for what in sys.argv[2:]:
    if what.startswith("gen"):
        k = int(what[3:]) - 1
        out = sb.generate(ws / ckpt, PROMPTS[64 * k:64 * k + 64], seeds=(0,), decode=SAMPLE,
            purpose=f"[{tag}] paired comparison of loop rate / rep-4 / length on prompts {64*k}-{64*k+63}, seed 0: "
                    "H2 = scaling the copy heads' output by 0.8 lowers loop rate and rep-4 without shortening stories.")
        res = {k2: out[k2] for k2 in ("call", "loop_rate", "rep4", "mean_length", "eot_rate", "generations_file")}
    elif what.startswith("loss"):
        k = int(what[4:]) - 1
        seqs = [tok.encode(s.strip())[:256] for s in stories[100 + 64 * k:164 + 64 * k]]
        out = sb.read(ws / ckpt, seqs,
            purpose=f"[{tag}] story loss on dev stories {100+64*k}-{163+64*k}: H3 = the 0.8 scaling of copy heads costs "
                    "well under 0.02 nats per token (second-order cost of a small perturbation).")
        res = {"call": out["call"], "mean_nll": out["mean_nll"], "nll_sum": sum(out["nll"]), "tokens": sum(out["tokens"]),
               "nll": out["nll"], "ntok": out["tokens"]}
    elif what == "tf":
        s = json.load(open("analysis/04_sequences.json"))
        out = sb.read(ws / ckpt, s["gen"] + s["real"], capture=("token_loss",),
            purpose=f"[{tag}] teacher-forced loss on the fixed pass-2 sequences: H4 = the edit raises the loss of "
                    "4-gram-repeating tokens (R4) much more than of other tokens, on generated and on real text.")
        res = {"call": out["call"], "mean_nll": out["mean_nll"], "capture_file": out["capture_file"]}
    res.update(ckpt=tag, what=what)
    print(json.dumps({k: v for k, v in res.items() if k not in ("nll", "ntok")}), "used", sb.used, "remaining", sb.remaining, flush=True)
    with open("analysis/eval_results.jsonl", "a") as f:
        f.write(json.dumps(res) + "\n")
