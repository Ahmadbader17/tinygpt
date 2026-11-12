"""Weights-only overview: parameter names, norms, per-head OV / QK structure."""
from pathlib import Path
import torch
from tinygpt.wsp.edits import load_base_params
from tinygpt.tokenizer import BPETokenizer

torch.manual_seed(0)
ws = Path(".")
P = load_base_params(ws / "checkpoint")
tok = BPETokenizer.load(ws / "tokenizer.json")
for n, v in P.items():
    if "blocks.0" in n or "blocks" not in n:
        print(f"{n:32s} {tuple(v.shape)} norm={v.norm():.3f}")

E = P["wte.weight"]            # (V, C) tied with unembedding
Wp = P["wpe.weight"]
V, C = E.shape
L, H, D = 6, 8, 32
print("wte row norm mean", E.norm(dim=1).mean().item(), "wpe row norm mean", Wp.norm(dim=1).mean().item())

def ln(x, w, b):
    x = x - x.mean(-1, keepdim=True)
    x = x / x.pow(2).mean(-1, keepdim=True).add(1e-5).sqrt()
    return x * w + b

lnf_w, lnf_b = P["ln_f.weight"], P["ln_f.bias"]
print("\nPer-head norms and direct-path token copying score")
print("copy: x = OV(ln1(wte[t])); logits = wte @ (lnf_w * x) ; report mean over tokens of z-score of own-token logit,")
print("      and share of tokens whose own logit ranks in the top 10 of 4096 (chance 0.0024)")
print("L.h  |OV|_F  |QK|_F  selfz  top10   posQK(prev-1)  eigpos")
rows = []
for l in range(L):
    qkv = P[f"blocks.{l}.attn.qkv.weight"]; qkv_b = P[f"blocks.{l}.attn.qkv.bias"]
    proj = P[f"blocks.{l}.attn.proj.weight"]
    w1, b1 = P[f"blocks.{l}.ln1.weight"], P[f"blocks.{l}.ln1.bias"]
    Xtok = ln(E, w1, b1)                 # (V, C) token-only input, as if residual == embedding
    Xpos = ln(Wp, w1, b1)
    for h in range(H):
        s = slice(D * h, D * h + D)
        Wq, Wk, Wv = qkv[0:C][s], qkv[C:2 * C][s], qkv[2 * C:][s]   # (32, C)
        bq, bk = qkv_b[0:C][s], qkv_b[C:2*C][s]
        O = proj[:, s]                                            # (C, 32)
        OV = O @ Wv                                               # (C, C)
        QK = Wq.T @ Wk
        out = (Xtok @ Wv.T) @ O.T                                 # (V, C) what the head writes when attending token t
        logits = (out * lnf_w) @ E.T                              # (V, V)  (centering/scale of ln_f ignored: direction only)
        diag = logits.diag()
        z = (diag - logits.mean(1)) / logits.std(1)
        rank = (logits > diag[:, None]).sum(1)
        top10 = (rank < 10).float().mean().item()
        # positional QK: does query at position p prefer key at p-1 (within positional subspace)?
        sc = ((Xpos @ Wq.T + bq) @ (Xpos @ Wk.T + bk).T) / D ** 0.5   # (T, T)
        T = sc.shape[0]
        idx = torch.arange(8, 200)
        prev = sc[idx, idx - 1].mean() - torch.stack([sc[i, :i + 1].mean() for i in idx]).mean()
        # eigen copying score of the token->token circuit restricted through E:  E^T-side: M = OV
        ev = torch.linalg.eigvals(OV * lnf_w[:, None])
        eigpos = (ev.real.sum() / ev.abs().sum()).item()
        rows.append((l, h, OV.norm().item(), QK.norm().item(), z.mean().item(), top10, prev.item(), eigpos))
        print(f"{l}.{h}  {OV.norm():6.2f}  {QK.norm():6.2f}  {z.mean():6.2f}  {top10:5.3f}  {prev:8.2f}      {eigpos:5.2f}")

# direct path (embedding -> unembedding) self-prediction, and MLP token->token
x = ln(E, lnf_w, lnf_b)
lg = x @ E.T
d = lg.diag(); z = (d - lg.mean(1)) / lg.std(1)
print("\ndirect path wte->ln_f->wte: own-token z", z.mean().item(), "top10", ((lg > d[:, None]).sum(1) < 10).float().mean().item())
for l in range(L):
    w2, b2 = P[f"blocks.{l}.ln2.weight"], P[f"blocks.{l}.ln2.bias"]
    hdn = torch.nn.functional.gelu(ln(E, w2, b2) @ P[f"blocks.{l}.mlp.fc.weight"].T + P[f"blocks.{l}.mlp.fc.bias"])
    out = hdn @ P[f"blocks.{l}.mlp.proj.weight"].T + P[f"blocks.{l}.mlp.proj.bias"]
    lg = (out * lnf_w) @ E.T
    d = lg.diag(); z = (d - lg.mean(1)) / lg.std(1)
    print(f"mlp{l} on raw token embedding: own-token z {z.mean():.2f} top10 {((lg > d[:, None]).sum(1) < 10).float().mean():.3f} |proj|={P[f'blocks.{l}.mlp.proj.weight'].norm():.2f}")
