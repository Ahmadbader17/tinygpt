"""Rule out a second fault: MLP neuron balance (fc row vs proj column), Q/K balance, LN outliers, embedding rows."""
from pathlib import Path
import torch
from tinygpt.wsp.edits import load_base_params
torch.set_printoptions(precision=3, linewidth=220, sci_mode=False)
p = load_base_params(Path("checkpoint"))
for L in range(6):
    fc = p[f"blocks.{L}.mlp.fc.weight"].double(); fb = p[f"blocks.{L}.mlp.fc.bias"].double(); pr = p[f"blocks.{L}.mlp.proj.weight"].double()
    a = (fc.norm(dim=1)**2 + fb**2).sqrt(); b = pr.norm(dim=0)
    r = b / a
    qkv = p[f"blocks.{L}.attn.qkv.weight"].double()
    Q, K = qkv[:256], qkv[256:512]
    qk = torch.corrcoef(torch.stack([Q.norm(dim=1), K.norm(dim=1)]))[0, 1]
    print(f"L{L} mlp: corr(in,out norm) {torch.corrcoef(torch.stack([a,b]))[0,1]:.3f} ratio mean {r.mean():.3f} cv {r.std()/r.mean():.3f} min {r.min():.3f} max {r.max():.3f} | QK row-norm corr {qk:.3f}")
    for nm in ("ln1", "ln2"):
        w = p[f"blocks.{L}.{nm}.weight"]; z = (w - w.mean()) / w.std()
        print(f"    {nm} weight min {w.min():.3f} max {w.max():.3f} n|z|>4: {(z.abs()>4).sum().item()}  bias max {p[f'blocks.{L}.{nm}.bias'].abs().max():.3f}")
w = p["ln_f.weight"]; print("ln_f min/max", w.min().item(), w.max().item())
rn = p["wte.weight"].norm(dim=1); print("wte row norm: mean %.3f min %.3f max %.3f; argmin %d argmax %d" % (rn.mean(), rn.min(), rn.max(), rn.argmin(), rn.argmax()))
rn = p["wpe.weight"].norm(dim=1); d = (p["wpe.weight"][1:] - p["wpe.weight"][:-1]).norm(dim=1)
print("wpe neighbour-diff norm: median %.3f max %.3f at %d" % (d.median(), d[8:].max(), d[8:].argmax() + 8))
# weights-only sigma estimate for L3 proj
W = p["blocks.3.attn.proj.weight"].double(); V = p["blocks.3.attn.qkv.weight"].double()[512:]
print("L3: ||proj||^2 %.3f  ||V||^2 %.3f  sigma_hat %.5f  256*sigma^2 %.4f" % (W.norm()**2, V.norm()**2, ((W.norm()**2 - V.norm()**2) / W.numel()).sqrt(), (W.norm()**2 - V.norm()**2) / 256))
