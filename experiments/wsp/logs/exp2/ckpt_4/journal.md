# Journal: exp2_ckpt_4

## Summary
- Damage: `blocks.2.mlp.proj.weight` carries one inflated rank-1 component (sigma_1 = 36.24; next 3.10).
- Repair (`repair.py`): subtract sigma_1 u_1 v_1^T, taken from the tensor's own SVD. One tensor touched.
- Passes used: 4 of 20. Revisions of the edit: 0.
- Loss, same inputs: windows 1.7636 -> 1.6578; stories 1.6609 -> 1.5563.

## Step 1: weight survey (free) - `analysis/01_norms.py`, output in `analysis/01_norms.out`
Per-tensor mean/std/max/kurtosis, per-head Q/K/V/O norms, top singular values of every matrix.
Everything varies smoothly with depth except one tensor:

    blocks.2.mlp.proj.weight  std 0.0824 (others 0.036 to 0.053), kurtosis 6.35 (others 3.2 to 4.2)
    top singular values       [36.244 3.098 3.05 2.819 2.8 2.688]
    other mlp.proj top values L0 2.36, L1 4.40, L3 3.56, L4 4.21, L5 4.78

Per-head attention norms, LayerNorm gains (all positive), biases and wpe (row 0 is the largest,
as usual) show no outlier. K biases are about 0 in every layer, which is normal (they cancel in softmax).

## Step 2: the suspect tensor (free) - `analysis/02_l2_mlp_proj.py`
- After removing the top SVD term the remainder has std 0.042 and top value 3.098, between its
  neighbours (L1 0.036, L3 0.048). So the damage is confined to one rank-1 direction.
- u (output side) and v (hidden side) are both dense: participation ratio 88 of 256 and 328 of 1024,
  no single row, column or neuron stands out. Not a block, row or neuron fault.

## Step 3: learned direction scaled up, or foreign direction added? (free) - `analysis/03_direction.py`
Two candidate repairs differ here: set sigma_1 to 0, or shrink it to a normal value (36.24/10 = 3.62
would be a plausible healthy top value).
- How later layers read u: |L3 mlp.fc @ u| = 1.62, |L3 qkv @ u| = 1.27, |L4 mlp.fc @ u| = 1.60,
  |L4 qkv @ u| = 1.36. A random unit vector gives fro/sqrt(256) = 1.6 and 1.3. The other top
  directions of the same matrix, and of other layers, are read with 2.2 to 3.4 / 2.0 to 2.4.
  So u is a direction nothing downstream has learned to use.
- corr(|v|, fc row norm) = 0.007, against 0.12 to 0.17 for the next three directions.
- Kurtosis of the matrix falls from 6.35 to 3.40 once the term is removed, same as L3 to L5.
Decision: treat it as a foreign rank-1 addition and remove it entirely (sigma_1 -> 0). I did not
test the "shrink to 3.6" variant: it would only differ by one direction of ordinary size, and trying
both would be search.

## Step 4: forward passes (`analysis/04_eval.py`; inputs fixed by seed 0)
1. read, `checkpoint`, 64 random 256-token dev windows, capture attn_out/mlp_out/resid_post at L1-3.
   Hypothesis: loss above 1.688, layer-2 mlp_out dominated by u.
   Result: mean_nll 1.7636. mlp_out rms norm L1 3.82, L2 8.92, L3 6.43; along u: 0.26, 7.45, 0.52.
   resid_post2 along u 7.53 (resid_post1 0.85). Confirmed.
2. read, `repaired`, same windows and captures. Hypothesis: loss back near 1.688, L2 mlp_out in line.
   Result: mean_nll 1.6578. mlp_out L2 rms 4.90 (along u 0.016); resid_post2 along u 0.97. Confirmed.
3. read, `checkpoint`, 64 whole dev stories from `<|endoftext|>` (cut at 256 tokens).
   Result: mean_nll 1.6609.
4. read, `repaired`, same stories. Hypothesis: near 1.605, otherwise a second fault.
   Result: mean_nll 1.5563.
The captured activations were used only to check the hypothesis; no value from them enters the edit.

## What I am unsure of
- My 64-row samples are not the reference's samples. Repaired loss is below both healthy reference
  figures (1.658 vs 1.688; 1.556 vs 1.605), which I read as sample difference (the damaged-to-repaired
  drop is 0.105 on both sets), not as beating the healthy model. I cannot show exact equality with
  the healthy loss from this, and a small second fault (a few hundredths of a nat) could hide in it.
  The weight survey showed no second outlier.
- Removal versus shrinking: if the original matrix had a real component along (u, v), I have removed
  it too. After repair L2 mlp_out along u is 0.016 rms, where a typical direction carries about 0.3,
  so a little genuine signal in that direction is lost. The effect should be small.
- The exact form of the injected fault (added random outer product versus a scaled singular value)
  is inferred from the read-norm evidence, not proven. Confidence 0.85.

## Rules
No gradients, no fitting, no search (one edit, no revisions), all model runs through the sandbox
(4 passes), nothing read outside the workspace, no sealed data, no other checkpoint, no other agents,
no web. Edit values come only from the SVD of the damaged tensor.
