"""From the pass-1 capture: what L5 head 6 attends to, and how similar its pattern is to the other L5 heads.
Used only to describe the head and to judge the 'redundant with h1/h7' hypothesis; no number from here goes into a weight."""
from pathlib import Path
import numpy as np
d = np.load(sorted(Path("work/outputs").glob("read_call001.npz"))[0])
print(d.files)
A = d["attn.5"] if "attn.5" in d.files else d[[k for k in d.files if k.startswith("attn")][0]]
print(A.shape)
R, H, T, _ = A.shape
ar = np.arange(T)
for h in range(H):
    a = A[:, h]
    self_ = a[:, ar, ar].mean()
    prev = a[:, ar[1:], ar[:-1]].mean()
    first = a[:, 8:, 0].mean()
    dist = (a * (ar[:, None] - ar[None, :])[None]).sum(-1)[:, 8:].mean()
    ent = (-(a * np.log(a + 1e-12)).sum(-1))[:, 8:].mean()
    mx = a.max(-1)[:, 8:].mean()
    print(f"h{h}: self {self_:.3f} prev {prev:.3f} pos0 {first:.3f} mean distance {dist:.1f} entropy {ent:.2f} max {mx:.3f}")
print("pattern cosine between heads (rows flattened over query positions >= 8):")
F = A[:, :, 8:, :].reshape(R, H, -1).astype(np.float64)
C = np.zeros((H, H))
for a in range(H):
    for b in range(H):
        C[a, b] = (F[:, a] * F[:, b]).sum() / np.sqrt((F[:, a]**2).sum() * (F[:, b]**2).sum())
np.set_printoptions(precision=3, suppress=True, linewidth=200)
print(C)
tl = d["token_loss"]
print("token_loss mean", tl.mean(), "by position quartile", [round(float(tl[:, i:i+64].mean()), 3) for i in range(0, 255, 64)])
