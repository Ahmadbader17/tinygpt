# Journal: editor, workspace exp2_ckpt_1

All scripts are in `analysis/` and were run from the workspace root. Forward passes used: 4 of 20.

## 1. Reading the task
Read `TASK.md`, `model_source/` and `harness_source/` (sandbox, edits, activations). Nothing
outside the workspace was read. Decision: start with free weight analysis, spend passes only to
size the damage and to test one written hypothesis.

## 2. Weight analysis

### analysis/01_norms.py: statistics of all 76 tensors (output in `analysis/01_norms.out`)
Frobenius norm, mean, std, max, kurtosis, zero fraction, top singular value for every tensor.
Everything follows a smooth trend with depth except one tensor:

```
blocks.3.ln2.weight  fro 19.250  mean 1.19546  std 0.13582  absmax 1.461
blocks.4.ln2.weight  fro 20.245  mean 1.26048  std 0.11103  absmax 1.439
blocks.5.ln2.weight  fro 46.504  mean 2.26387  std 1.82643  absmax 5.812   <-- outlier
blocks.5.ln1.weight  fro 18.364  mean 1.14589  std 0.06547  absmax 1.294
```
No zeros, no NaN, no rank-1 spikes elsewhere. `blocks.0.attn.proj.weight` is smaller than in later
layers (4.09 against 8.3) and `blocks.0.ln1.weight` has mean 0.55, but both fit a monotone trend
in depth that is usual for a first layer, so I treat them as healthy (not tested; see doubts).

### analysis/02_ln.py: the LayerNorm gains (output in `analysis/02_ln.out`)
- `blocks.5.ln2.weight` splits into two clusters with an empty gap: 192 entries in 0.906 to 1.352
  (mean 1.216, std 0.058) and exactly 64 entries in 3.862 to 5.812 (mean 5.408, std 0.318).
- The 64 indices are scattered over the 256 channels (list in `diagnosis.json`).
- In layers 1 to 4 the ln2 gain correlates 0.94 to 0.96 with the column norm of `mlp.fc.weight`;
  in layer 5 that correlation is -0.005 over all channels, but 0.89 inside the small cluster and
  0.93 inside the large one. So the large cluster still carries the structure of a trained gain
  vector: it was multiplied by a common factor, not overwritten.
- The fc column norms of the two groups are equal (1.624 and 1.634), so fc was not rescaled to
  compensate: this is damage, not a trained reparametrisation.
- Std of the large cluster (0.318) is about 4.45 x the small one (0.256), again multiplicative
  rather than additive (additive would leave 0.058).

Hypothesis H1: the fault is a multiplication of 64 gains of `blocks.5.ln2.weight` by one factor.
Effect: a quarter of the input channels of the last MLP are overdriven about 4.5 times, which
distorts its hidden activations and its write into the final residual stream.

### analysis/03_factor.py: the factor, from weights only (output in `analysis/03_factor.out`)
```
ratio of group means              4.448
ratio of group medians            4.507
ratio of gain/fc-colnorm (means)  4.473   (medians 4.493)
per-channel implied factor from the gain~colnorm relation of the 192 clean channels:
   mean 4.477, median 4.478, std 0.105, standard error of the mean 0.013
control: the same procedure on layer 4 with the same index set gives 0.997 (std 0.028)
```
The per-channel spread (2.3 percent) equals the natural spread seen in the layer 4 control
(2.8 percent), which fits one common factor. The estimates bracket 4.5 (4.45 to 4.51), so I chose
the round scalar 4.5. This number comes from the weights alone; no activation or model output
was used to produce it.

## 3. The edit (`repair.py`)
Divide the entries of `blocks.5.ln2.weight` that are above 2.0 (the 64 in the upper cluster) by
4.5. One tensor touched, 64 weights changed, relative L2 change of that tensor 0.233. The script
asserts that it finds exactly 64 entries and writes `repaired/` with `write_edit`.
The edit was written before any forward pass was made.

## 4. Forward passes (`analysis/04_eval.py`, fixed inputs, seed 0)
Inputs: (a) 64 windows of 256 tokens at random offsets in a stream of 300 dev stories joined by
`<|endoftext|>`; (b) 64 other whole dev stories, each `<|endoftext|>` + story, cut at 256 tokens.

| pass | weights | input | purpose | mean nll |
|---|---|---|---|---|
| 1 | checkpoint | (a) | baseline: size of the damage against healthy 1.688 | 2.512 |
| 2 | repaired | (a) | test of H1 and of the repair: expect about 1.688 | 1.674 |
| 3 | repaired | (b) | second check on whole stories: expect about 1.605 | 1.522 |
| 4 | checkpoint | (b) | baseline on the same whole stories | 2.371 |

Reading: the damage costs about 0.84 nats per token on both measures and the single edit removes
it. The repaired values are at (slightly below) the healthy reference figures; the small
differences are what I would expect from my samples being different from the ones the reference
figures were computed on (64 rows each), not evidence of being better than healthy.

## 5. Decision
H1 confirmed. No revision of the edit was made (0 of 3 revisions used). I stopped at 4 passes:
the loss is back at the healthy level, so more passes would only be searching.

## 6. Doubts
- The exact factor. Weights say 4.48 +- 0.013 on the mean; I used 4.5. If the true factor was,
  for example, 4.4 or 4.5, the residual error per gain is at most about 2 percent, inside the
  natural spread of the gains. I did not tune it against the loss (that would be search).
- Whether the original values of the 64 gains were exactly `value / 4.5`: yes only if the fault
  was a single common factor; a per-entry noisy factor cannot be told apart from the natural
  2 to 3 percent spread.
- I cannot compare the repaired loss with the healthy loss on identical inputs, so a second,
  much smaller fault elsewhere cannot be excluded by the passes. The weight statistics show no
  other anomaly.

## 7. Rules
No gradients, no fitting to data, no search over strengths or locations, no revision. Every
model run went through the sandbox (4 calls, all in `work/ledger.jsonl`). No model was built or
run by hand. Nothing outside the workspace was read apart from importing the installed `tinygpt`
package; no sealed data, no other checkpoint, no web, no other agents.
