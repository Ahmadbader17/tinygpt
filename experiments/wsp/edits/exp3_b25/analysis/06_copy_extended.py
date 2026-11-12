"""Weights-only: copy score of layer 3-5 heads when the attended token is represented by the
'extended embedding' x(t) = wte[t] + mlp0(ln2_0(wte[t]))  (layer-0 MLP is by far the largest early writer),
instead of the raw embedding used in 01_overview.py. Also the same with full OV sign structure:
selfz = z-score of own-token logit; top10 = share of tokens whose own logit is in the head's top 10."""
from pathlib import Path
import torch
from tinygpt.wsp.edits import load_base_params
P = load_base_params(Path("checkpoint"))
E = P["wte.weight"]; C = 256; D = 32
def ln(x, w, b):
    x = x - x.mean(-1, keepdim=True); x = x / x.pow(2).mean(-1, keepdim=True).add(1e-5).sqrt(); return x * w + b
h0 = torch.nn.functional.gelu(ln(E, P["blocks.0.ln2.weight"], P["blocks.0.ln2.bias"]) @ P["blocks.0.mlp.fc.weight"].T + P["blocks.0.mlp.fc.bias"])
X = E + h0 @ P["blocks.0.mlp.proj.weight"].T + P["blocks.0.mlp.proj.bias"]
print("norm raw", E.norm(dim=1).mean().item(), "extended", X.norm(dim=1).mean().item())
lw = P["ln_f.weight"]
for l in range(1, 6):
    qkv = P[f"blocks.{l}.attn.qkv.weight"]; proj = P[f"blocks.{l}.attn.proj.weight"]
    xin = ln(X, P[f"blocks.{l}.ln1.weight"], P[f"blocks.{l}.ln1.bias"])
    line = []
    for h in range(8):
        s = slice(D * h, D * h + D)
        out = (xin @ qkv[2 * C:][s].T) @ proj[:, s].T
        out = out - out.mean(-1, keepdim=True)
        lg = (out * lw) @ E.T
        d = lg.diag(); z = ((d - lg.mean(1)) / lg.std(1)).mean().item()
        top10 = ((lg > d[:, None]).sum(1) < 10).float().mean().item()
        gain = (d - lg.mean(1)).mean().item()
        line.append(f"{l}.{h} z={z:5.2f} top10={top10:.3f} rawgain={gain:6.2f}")
    print("\n".join(line))
