# Journal: editor, workspace exp2_ckpt_3

All scripts are in `analysis/` (outputs in `analysis/*.out`), run from the workspace root. I read only this
workspace (TASK.md, model_source, harness_source, dev data, checkpoint). No other folder, no web, no agents.

## 1. Weight analysis (free)

### analysis/01_norms.py: per-tensor norms, per-head norms, top singular values
Stand-out: `blocks.3.attn.proj.weight` has Frobenius norm 18.79; the neighbours have 8.3, 9.3, 11.7, 13.7.
Per-head norms (O = proj columns of the head, V = value rows of qkv):

    layer 2  V 3.23 3.57 3.09 3.26 3.24 3.43 3.20 3.46   O 3.32 3.48 3.18 3.18 3.27 3.36 3.18 3.41
    layer 3  V 4.20 3.49 3.44 3.75 3.66 3.77 3.47 3.81   O 6.90 6.43 6.56 6.60 6.67 6.70 6.50 6.77
    layer 4  V 4.03 4.01 3.53 3.71 4.37 4.26 4.70 3.95   O 4.04 3.99 3.49 3.68 4.44 4.36 4.84 4.00
    layer 5  V 4.09 5.07 3.99 4.06 4.52 3.96 5.43 6.27   O 4.11 5.22 4.06 4.06 4.55 3.92 5.69 6.51

In healthy layers O and V norms agree head by head to 1-3 %. In layer 3 all O norms are inflated and almost
equal to each other, V/Q/K norms and the proj bias are ordinary. Entry kurtosis of the tensor is 3.00 and its
singular spectrum is unusually flat (top 2.34, 2.29, 2.27, ...; stable rank 64.5).

### analysis/02_proj_spectra.py: compare attn.proj across layers
    L2 corr(col norm O, row norm V) 0.924  ratio 0.998     cos(O^T O, V V^T) 0.94-0.98
    L3 corr(col norm O, row norm V) 0.458  ratio 1.802     cos(O^T O, V V^T) 0.88-0.92
    L4 corr 0.980 ratio 1.009 ; L5 corr 0.992 ratio 1.017 ; L1 0.966 / 0.970 ; L0 0.930 / 0.867
A pure rescale would keep the column-norm correlation; it drops instead, while the Gram still resembles V V^T.
That is what added noise does.

### analysis/03_gram.py: noise or rescale?
For each head, write O_h^T O_h in the eigenbasis of G_h = V_h V_h^T and describe its diagonal as a*eig(G)+c
(a descriptive fit on weights only; nothing from it is written into a weight).

    layer 3 slope a : 1.11 0.88 1.07 0.75 1.00 0.88 0.84 0.77
    layer 3 offset c: 0.877 0.957 0.949 1.030 0.971 1.015 1.005 1.082  -> sigma = sqrt(c/256) = 0.059..0.065
    layer 3 trace(O^T O) - trace(G) per head: 29.99 29.18 31.16 29.50 31.03 30.71 30.23 31.36
    other layers: offsets -0.05..0.18, trace differences within about +-3

Slope about 1 (signal intact, not rescaled) plus the same isotropic offset in all 8 heads:
`proj = W0 + N`, N iid, sigma about 0.061. Clean norm^2 about ||V||^2 = 109.9, noise norm^2 about 243, so the
tensor is more noise than signal (clean entry std about 0.041, noise about 1.5x that).

### analysis/04_other_checks.py: is there a second fault?
MLP neuron in/out balance, Q/K row-norm correlation, LayerNorm gains and biases, embedding row norms, wpe
smoothness: nothing out of family in any layer. I found one fault only.
Weights-only estimate used later: sigma_hat = sqrt((||proj||^2 - ||V||^2)/65536) = 0.06091, 256 sigma^2 = 0.950.

## 2. Edits

The noise itself is unknown, so the clean tensor cannot be restored exactly. What the weights do give:
the clean second moment of each head's 32 input columns (G_h, from the undamaged value rows) and the noise
variance (from the balance). With clean rows ~ N(0, G_h/256) and noise N(0, sigma^2 I), the posterior mean is
the Wiener filter `O_h <- O_h G_h (G_h + 256 sigma^2 I)^-1`. That is `repair.py`. It needs no activations or
outputs. Result: relative change of the tensor 0.697, new norm 6.11 (per-head 1.9-2.6): it deliberately
under-shoots the clean norm of about 10.5 because directions where noise dominates are shrunk hardest.

A second, purely diagnostic edit (`analysis/05_edit_ablate.py`, written to `edited/`): zero the tensor.

I considered and rejected: singular-value truncation of the whole tensor or of head blocks (the clean
singular values, 0.3-1.0 per head, sit at or under the noise bulk, edge about 0.58-1.3, so the clean subspace
is not recoverable that way); a plain rescale by 1/1.8 (keeps the noise direction mix unchanged).

## 3. Forward passes (8 of 20 used)

| call | checkpoint | data | purpose | mean nll |
|---|---|---|---|---|
| 1 | checkpoint | 64 windows x 256, seed 0 | baseline of the damaged model | 1.7173 |
| 2 | edited/ablate_l3_attn_proj | same | if the L3 proj noise is the damage, zeroing should recover most of the loss | 1.7286 |
| 3 | repaired (Wiener) | same | filtering should keep signal and drop noise: at or below the ablated value | 1.6788 |
| 4 | checkpoint | 64 fresh windows, seed 1 | replicate the baseline | 1.8039 |
| 5 | repaired | same as 4 | replicate the gain | 1.7661 |
| 6 | checkpoint | 64 whole stories from `<|endoftext|>` | baseline for the story metric | 1.6718 |
| 7 | repaired | same as 6 | gain on the story metric | 1.6409 |
| 8 | repaired | generate, 8 dev prompts x 2 seeds, 120 tokens, T 0.8, top-k 40 | sanity: coherent text | loop_rate 0.125, rep4 0.016, fluent stories |

Reading of the results:
* Call 2 did not come out as I predicted. Zeroing the tensor is slightly worse than leaving the noise in
  (1.729 vs 1.717). So the damaged tensor still delivers useful signal, the noise costs less than I expected,
  and layer-3 attention as a whole is worth only about 0.05 nats. The ablation is therefore a weak test of
  localisation; the localisation rests on the weight evidence and on call 3.
* The repair lowers loss on all three data sets: -0.038, -0.038, -0.031 nats.
* Sample-to-sample spread is large (1.717 vs 1.804 for the same damaged model on two window sets), so I
  cannot compare my numbers with the healthy references 1.688 / 1.605, which were measured on other samples.
  On seed 0 the repaired model is below 1.688, on seed 1 above it. My story sample is also not the
  reference one (stories 401.. of at most 255 tokens, truncation avoided), where the repaired model is
  0.036 above 1.605. How much of the damage remains is not measured.

## 4. Revisions
None. `repair.py` is the first and only version of the repair; the ablation was a diagnostic and was never a
candidate. I did not tune the filter strength: sigma comes from the weight balance and was fixed before call 3.

## 5. Open points
* The repair is partial by construction: about 70 % of the tensor's norm was removed, and the kept part
  still contains filtered noise. The residual loss gap to healthy is unknown.
* The noise model (iid Gaussian, one sigma, whole tensor) fits very well; whether sigma is exactly 0.061 or
  was specified as a multiple of the tensor's std (about 1.5x) is a guess.
* Not tried: per-head optimal singular-value shrinkage on top of the Wiener filter. Expected gain small.
* Rule check: no gradients, no fitting to data, no search over strengths, sandbox only for model runs,
  no sealed or outside data read, ledger untouched. Interpretation I1 kept: every number written into the
  weight is a function of `blocks.3.attn.proj.weight` and `blocks.3.attn.qkv.weight` alone.
