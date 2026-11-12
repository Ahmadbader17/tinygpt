"""Descriptive statistic of the unedited model on real text (pass 3): entropy and variance of log p under p.
A logit rescale by s costs about (1-s)^2/2 * mean varentropy nats (second order; first order is ~0 if calibrated).
Used to PREDICT the cost of the already-fixed s=0.9; not used to set s."""
import numpy as np
z = np.load("work/outputs/read_call003.npz")
lg = z["logits"][:, :-1].astype(np.float64); lp = lg - np.logaddexp.reduce(lg, -1, keepdims=True); p = np.exp(lp)
H = -(p * lp).sum(-1); V = (p * (lp + H[..., None]) ** 2).sum(-1)
tl = z["token_loss"]
print(f"mean loss {tl.mean():.4f}  mean entropy {H.mean():.4f} (calibration: entropy ~ loss)  mean varentropy {V.mean():.3f}")
print(f"first-order term of a rescale, mean(loss - entropy) = {(tl - H).mean():.4f} per unit (1-s)")
for s in (0.9,):
    print(f"second-order prediction for s={s}: dLoss ~ {(1-s)*(H - tl).mean()*-1*-1:.4f} + {(1-s)**2/2*V.mean():.4f}")
