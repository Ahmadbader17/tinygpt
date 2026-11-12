# Journal: workspace exp2_ckpt_5

## Outcome in brief

- Fault: head 5 of block 5, output side. Columns 160-191 of `blocks.5.attn.proj.weight` carry a
  gain of about 5.5.
- Repair (`repair.py`): divide those 32 columns by 5.556, a number derived from the weights alone.
  One tensor touched, 8192 weights changed.
- Loss, same 64 rows before and after: windows 1.845 -> 1.762, whole stories 1.742 -> 1.661.
  A fresh draw of 64 windows on the repaired model: 1.694 (healthy reference 1.688).
- Forward passes used: 5 of 20. Revisions of the edit: 0.

## Step 1. Weights-only survey (`analysis/01_norms.py`, output in `analysis/01_norms.out`)

Per-tensor norms, per-head norms of Q, K, V rows of `qkv.weight` and of each head's columns
of `proj.weight`, top singular values. One thing stands out:

```
blocks.5.attn.proj.weight  fro 25.994  std 0.1015  absmax 1.017  kurt 13.70   (blocks 1-4: fro 8.3-11.7, kurt ~3)
5 V tensor([4.092, 5.071, 3.985, 4.059, 4.519, 3.962, 5.430, 6.265])
5 O tensor([ 4.105,  5.224,  4.061,  4.062,  4.549, 22.432,  5.688,  6.508])
```

Everywhere else a head's output-column norm is nearly equal to its value-row norm. Head 5 of
block 5 has value norm 3.96 and output norm 22.43.

## Step 2. Balance check (`analysis/02_balance.py`, output in `analysis/02_balance.out`)

Reasoning: a head's function is unchanged by O_h -> O_h M^-1, V_h -> M V_h, and training keeps
O_h^T O_h close to V_h V_h^T. So the ratio |O_h| / |V_h| is a weights-only reference for the
gain of a head's output columns.

```
L h |    |O|     |V|  ratio | cos(GO,GV)
5 0 |  4.105   4.092  1.003 | 0.9516
5 1 |  5.224   5.071  1.030 | 0.9876
5 2 |  4.061   3.985  1.019 | 0.9411
5 3 |  4.062   4.059  1.001 | 0.9699
5 4 |  4.549   4.519  1.007 | 0.9704
5 5 | 22.432   3.962  5.662 | 0.9708
5 6 |  5.688   5.430  1.047 | 0.9924
5 7 |  6.508   6.265  1.039 | 0.9939
```

- Blocks 2-5, all other heads: ratio 0.97-1.05. Block 1: 0.93-1.02. Block 0: 0.78-0.92.
- Head 5's per-column ratio is uniform (5.52-5.81; a healthy head spreads 0.98-1.05), and the
  cosine between its two Gram matrices is ordinary (0.971). So the columns were multiplied by one
  scalar; they were not replaced or made noisy. Entry kurtosis within the head is 3.03.
- Q/K balance per head and MLP neuron balance (fc row vs proj column) show no outlier.

`analysis/04_other_checks.py` (output `analysis/04_other_checks.out`): robust z-scores of row
and column norms of every tensor. The only new outlier is again the head-5 columns of
`blocks.5.attn.proj.weight` (max z 17.8 at column 185). Other large values sit in residual
dimensions 5, 16, 68, 143, 251, which recur in every layer and look like the model's own special
directions; `wpe` row 0 is large, which is usual for a first position. I conclude one fault.

## Step 3. The edit and where its number comes from

Divisor = (|O|/|V| of head 5) / (median of |O|/|V| over the other seven heads of block 5)
= 5.662 / 1.019 = 5.556. Computed in `repair.py` from the weights; no activation or model output
enters it (rule 2, interpretation I1). After the edit, head 5 has the layer's typical balance.

Uncertainty on the number: the other heads' ratios span 1.001-1.047, so the true gain could be
anywhere in about 5.4-5.8 (a round 5.5 is inside that). The repair is then off by at most a few
percent on one head.

## Step 4. Forward passes (`analysis/03_passes.py`, `analysis/05_noise_pass.py`)

Evaluation rows are fixed by `analysis/evalsets.py` (seed 1234): 64 random 256-token windows of
the dev stream, and 64 whole stories (`<|endoftext|>` + story) that fit in 256 tokens. The first
entry of `dev/stories.txt` is a fragment and is skipped.

| call | checkpoint | rows | purpose (hypothesis) | result |
|---|---|---|---|---|
| 1 | damaged | 64 windows, capture head_out/resid_pre at block 5 | H1: loss above 1.688 and head 5 of block 5 writes several times more than its neighbours | mean nll 1.845; head output norms [2.28 2.79 1.94 2.05 2.39 **11.07** 2.99 2.83]; residual norm entering block 27.6 |
| 2 | damaged | 64 whole stories | H1b: loss above 1.605 | 1.742 |
| 3 | repaired | same 64 windows, capture head_out at block 5 | H2: dividing by 5.556 restores loss | 1.762; head norms [2.28 2.79 1.94 2.05 2.39 **1.99** 2.99 2.83] |
| 4 | repaired | same 64 stories | H2b: same for stories | 1.661 |
| 5 | repaired | 64 fresh windows (seed 98765) | H3: the remaining 0.07 gap to 1.688 on the first windows is sampling noise | 1.694; per-window std 0.41, standard error 0.05 |

Reading:
- H1 confirmed: head 5 writes 11.07 against 1.9-3.0 for the others. That ratio agrees with the
  weight-derived gain, but the edit value was fixed before any pass and was not adjusted after.
- H2 confirmed in direction: the repair removes 0.083 nats on windows and 0.081 on stories.
- The repaired numbers on my first rows (1.762, 1.661) are above the quoted healthy 1.688 and
  1.605. Those references come from a different sample that I do not have. Call 5 shows a fresh
  draw lands at 1.694, and the standard error of a 64-window mean is about 0.05, so the gap is
  within sampling noise. I cannot rule out a small remaining fault from these numbers alone; the
  weights show none.
- The capture files of calls 1 and 3 (about 130 MB each) were deleted after the summary norms
  were printed; the printed norms are in `analysis/03_passes.out`.

## Decisions made alone

- Divisor taken from the layer's median balance (5.556) and not rounded to 5.5: the weights give
  this number directly and I have no evidence the injected gain was round.
- Stopped at 5 passes: the diagnosis is settled by the weights, confirmed by the capture, and
  further passes would only be re-measuring.
- My whole-story set keeps only stories that fit in 256 tokens, so it is not the same population
  as the 1.605 reference.

## Rules

No gradients, no fitting, no search (one edit, zero revisions). All model runs went through the
sandbox. Nothing outside the workspace was read apart from importing `tinygpt`. No sealed data,
no other checkpoint, no fault specification seen. The ledger was not edited.
