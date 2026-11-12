"""Weights-only: eigenvalues of each head's token->token OV circuit (copying score, Elhage et al. 2021).

Full circuit (4096x4096): D_h = [Ec G_f C P_h] [V_h G_1 N(E)^T]   (source token -> logits, LayerNorms linearised:
N(E) = centred, unit-RMS embedding rows; G_1 = ln1 gain of the head's layer; C = centring; G_f = ln_f gain).
Its non-zero eigenvalues are those of the 32x32 matrix M_h = [V_h G_1 N(E)^T] [Ec G_f C P_h].
Positive eigenvalues = modes where attending to a token raises that same token's logit (copying).
"""
from pathlib import Path
import torch
from tinygpt.wsp.edits import load_base_params
torch.set_printoptions(precision=2, linewidth=220, sci_mode=False)
P = {k: v.double() for k, v in load_base_params(Path("checkpoint")).items()}
E = P["wte.weight"]; Ec = E - E.mean(0, keepdim=True)
NE = E - E.mean(1, keepdim=True); NE = NE / NE.pow(2).mean(1, keepdim=True).sqrt()
gf = P["ln_f.weight"]

def circuit(L, h):
    s = slice(32 * h, 32 * h + 32)
    V = P[f"blocks.{L}.attn.qkv.weight"][512:][s]            # (32,256)
    Pr = P[f"blocks.{L}.attn.proj.weight"][:, s]             # (256,32)
    read = (NE * P[f"blocks.{L}.ln1.weight"]) @ V.T          # (4096,32) a_t
    Prc = Pr - Pr.mean(0, keepdim=True)                      # centring of the head's write
    write = Ec @ (gf[:, None] * Prc)                         # (4096,32) b_t
    return read.T @ write                                    # M (32,32)

if __name__ == "__main__":
    for L in range(6):
        for h in range(8):
            ev = torch.linalg.eigvals(circuit(L, h))
            re = ev.real
            score = (re.sum() / ev.abs().sum()).item()
            pos = re[re > 0]
            print(f"L{L}H{h}: copy score sum(Re)/sum|.| = {score:5.2f}  n_pos={len(pos):2d}  sum_pos={pos.sum():9.1f}  sum_neg={re[re<0].sum():9.1f}  top Re: "
                  + " ".join(f"{x:.0f}" for x in re.sort(descending=True).values[:5]))
