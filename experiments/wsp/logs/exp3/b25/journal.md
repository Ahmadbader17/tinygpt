# Journal (editor, workspace exp3_b25, budget 25 passes)

All scripts are in `analysis/` and `edits/`, run from the workspace root; printed output is kept
next to each script as `*.out` and summarised here.

## Step 0. Reading
Read TASK.md, model_source/, harness_source/. Model: 6 layers x 8 heads, d=256, head dim 32,
pre-norm, tied embedding/unembedding, learned positions, vocab 4096.

## Step 1. Weights-only overview (`analysis/01_overview.py`, free)
For every head: OV = proj[:, head cols] @ W_V[head rows]; "self z" = z-score of a token's own
logit when the head attends that token's raw embedding (wte -> ln1 -> OV -> ln_f gain -> wte^T);
"eigpos" = sum(Re eig)/sum|eig| of OV*ln_f gain (positive = copies what it attends to, negative =
suppresses it). Full table in `analysis/01_overview.out`. Reading:
- Layer 0 heads 0.3/0.5/0.6/0.7 are copy-like but tiny (|OV| ~0.6): they move token identity.
- Layers 1-3 are overwhelmingly *anti*-copy (eigpos -0.5..-0.8): they push down attended tokens.
- Copy-like late heads (they write the attended token's own unembedding direction):
  4.6 (selfz 0.95, eigpos 0.66, |OV| 4.3), 5.0 (0.80, 0.59, 3.2), 5.6 (0.73, 0.39, 5.8),
  5.7 (0.55, 0.23, 7.6), 5.1 (0.28, 0.38, 5.4); weaker: 3.0, 4.0.
- Direct path wte->wte strongly self-predicting in z terms (tied weights), MLPs 2-5 anti-self.
Hypothesis H1 from weights: repetition under sampling is copying of earlier context tokens by
the late copy heads 4.6, 5.0, 5.6 (maybe 5.1, 5.7).

## Pass 1 (generate, base, prompts 0-63, seed 0) `analysis/02_baseline_gen.py`
Purpose: baseline for paired comparisons + material to see what loops are.
Result: loop_rate 0.234, rep4 0.0255, mean_length 162.6, eot_rate 0.72 (matches the task table).
`analysis/03_inspect_loops.py` (free): the 15 looping generations loop on re-mentioned entities
and phrases: ' Velvet' x9, ' Bouncy.' x5, 'The lawyer', ' the vacation' x5, ' big, red ball',
' did not move.' x4, ' wanted to play with' x4. No tight cycles. So "loops" under sampling are
re-use of a noun phrase / multi-token name that is already in the context, i.e. copying from
context (multi-token rare words make 4-grams by themselves). Consistent with H1's premise.

## Pass 2 (read + capture, base) `analysis/04_capture.py`, analysed in `analysis/05_attribution.py`
Purpose: test H1. 64 rows x 160 tokens: 32 of the pass-1 generations and 32 real dev stories.
Captured attn, head_out, mlp_out, resid_pre/post at all layers and token_loss.
Positions are split by what the next token is: R4 = it completes a 4-gram already seen in the row,
R1 = seen before but not R4, N = new. Direct logit attribution (DLA) to the actual next token.
Results (full table in `analysis/05_attribution.out`):
- R4 positions are 2.6% of generated text and 1.5% of real text. Loss at R4: 0.44 (gen), 1.32 (real).
- Heads whose DLA to the next token is larger at R4 than at N (generated text), with attention mass
  on induction sources (earlier copy of the next token that follows a copy of the current token):
  5.6 (+0.59, ind attn 0.23), 5.7 (+0.57, 0.27), 4.5 (+0.40, 0.22), 5.1 (+0.34, 0.18),
  3.0 (+0.18, 0.13), 4.6 (+0.17, 0.13), 5.4 (+0.17, 0.13). Every other head is below 0.07 or negative.
- The same heads do the same thing on real text (5.6 +0.61, 5.7 +0.68, 4.5 +0.36, 5.1 +0.34): the
  mechanism is not a malfunction that only shows in generations; it is the model's normal
  "re-mention what is in the context" circuit. Any reduction is a trade against real-text prediction.
- H1 partly confirmed: 5.6, 5.7, 5.1, 4.6 predicted from weights were right; 5.0 was wrong (its DLA
  gap is negative); 4.5, 3.0, 5.4 were missed by the raw-embedding score.

## Step 2. Weights-only refinement (`analysis/06_copy_extended.py`, free)
The raw embedding (norm ~1) is not what late heads read; MLP0 writes ~18x more. Using the
"extended embedding" wte[t] + mlp0(wte[t]) as the attended token's representation, the share of
tokens whose own logit lands in the head's top-10 outputs (chance 0.002):
5.7 0.76, 4.6 0.68, 5.6 0.62, 4.4 0.49, 5.1 0.43, 3.0 0.42, 3.4 0.28, 4.5 0.20, 5.4 0.14, 3.5 0.12;
all others < 0.07. This weights-only score now agrees with the pass-2 DLA ranking (4.4, 3.4, 3.5 copy
by OV but show little R4-specific DLA: 4.4 +0.06).

## Passes 3-7 (base) `analysis/07_eval.py checkpoint gen2 gen3 gen4 loss1 loss2`
Purpose: baseline halves of paired comparisons.
gen (prompts 64-127 / 128-191 / 192-255, seed 0): loop 0.297 / 0.234 / 0.234, rep4 0.0302 / 0.0244 /
0.0275, length 168.0 / 159.4 / 159.0. With pass 1: loop 0.250, rep4 0.0269, length 162.3 over 256 generations.
loss (dev stories 100-163, 164-227, first 256 tokens): 1.5722 (11641 tok), 1.6172 (11828 tok); pooled 1.5949.

## Edit v1 `edits/copy_heads_08.py` (written before any edited-model pass)
Components: heads that pass BOTH criteria fixed here: (a) weights-only copy score (extended embedding
top-10 share) > 0.1 and (b) pass-2 DLA gap R4-N > 0.15 on generated text: 3.0, 4.5, 4.6, 5.1, 5.4, 5.6, 5.7.
Operation: multiply each head's 32 input columns of `blocks.L.attn.proj.weight` by 0.8 (nothing else).
Why 0.8: the evaluation samples at temperature 0.8, i.e. every logit contribution, including the copy
heads' evidence "this token was said before, say it again", is multiplied by 1.25 relative to the
distribution the model was trained to match. Multiplying the copy heads' output by 0.8 (the temperature)
returns that particular evidence to its trained strength under the evaluation's sampling, while the rest
of the distribution stays sharpened. The number comes from the decoding temperature, not from any
activation or output. Expected loss cost: the model is near a training optimum, so a 20% change in
7 of 48 heads costs loss only at second order; H3 says < 0.02 nats, to be tested.
Built: `edited/copy_heads_08/`, 3 tensors touched, 57,344 weights changed (0.98% of all), relative L2
change of the whole model 0.0196, sha 4afd6061... (re-running the script reproduces the same sha).

## Passes 8-9 (read, v1) `analysis/07_eval.py edited/copy_heads_08 loss1 loss2`
Purpose: H3, loss cost on the same 128 dev stories. Result: 1.5803 and 1.6239; pooled 1.6023 vs 1.5949,
difference +0.0074 nats/token (bootstrap over stories 95% CI +0.0054..+0.0094). H3 holds (< 0.02).

## Pass 10 (read + token_loss, v1) `07_eval.py ... tf`, compared in `analysis/08_tf_compare.py`
Purpose: H4, teacher-forced loss on the fixed pass-2 sequences, by category.
generated text: N -0.006, R1 +0.022, R4 +0.017 (se 0.005); real text: N -0.013, R1 +0.020, R4 +0.050 (se 0.014).
The edit moves probability from already-seen tokens to new tokens, as intended, on both kinds of
text. H4 only partly holds: the R4 change is not larger than the R1 change on generated text
(repeat tokens there are already near probability 1, so their loss hardly moves). The effect in
log-probability is small (hundredths of a nat per token).

## Passes 11-14 (generate, v1, prompts 0-255, seed 0) `07_eval.py edited/copy_heads_08 gen1..gen4`
Purpose: H2, paired against passes 1, 3, 4, 5.
loop 0.234 / 0.219 / 0.188 / 0.156 (base 0.234 / 0.297 / 0.234 / 0.234); pooled 0.199 vs 0.250.
rep4 pooled 0.0244 vs 0.0269; length 162.4 vs 162.3. Direction as predicted, but with 256 pairs the
loop-rate difference is only about 1.4 standard errors.

## Passes 15-22 (generate, base then v1, prompts 0-255, seed 1) `analysis/09_eval_seed1.py`
Purpose: replicate on fresh randomness instead of revising the edit (decision: evidence for one
defensible edit is worth more than a second edit strength, which would be tuning on evaluation).
seed 1: base loop 0.258, rep4 0.0280; v1 loop 0.215, rep4 0.0231. Same direction, similar size.

## Pooled statistics (`analysis/10_paired_stats.py`, free; bootstrap over the 256 prompts)
loop rate 0.254 -> 0.207, diff -0.047, 95% CI [-0.092, -0.002]
rep-4     0.0275 -> 0.0237, diff -0.0037, 95% CI [-0.0065, -0.0009]
long loops 0.006 -> 0.004 (no evidence either way)
length    160.9 -> 162.0 (+1.05, CI [-2.4, +4.6]); eot rate 0.754 -> 0.711 (CI includes 0)
dev loss  1.5949 -> 1.6023, +0.0074, 95% CI [+0.0054, +0.0094]

## Decision
Final = `copy_heads_08` (edit v1, no revisions used). 22 of 25 passes spent, 3 left unspent on
purpose: a stronger factor would have no reason other than "the loss budget allows it" (that would
be tuning the strength on evaluation results), and three passes are too few to measure a control
edit. Things not done: no control edit on non-copy heads (the final evaluation does that), no greedy
decoding test.

## Rule notes
- Passes: 22 used of 25 (ledger calls 1-22, none refused). Revisions used: 0 of 3.
- The data-format inspection (310 prompts of 9 tokens with text; 5001 dev stories, mean ~200 tokens)
  was first run as an inline snippet, not from a file. It ran no model. I saved it afterwards as
  `analysis/00_inspect_data.py` with its output. Every other script was a file from the start.
- `analysis/01_overview.py` and `06_copy_extended.py` apply layer norms, OV matrices and (in 06) the
  layer-0 MLP to the rows of the embedding matrix. That is a computation over the vocabulary table,
  not a run of the model on a token sequence, and I treated it as weights-only analysis. If the
  experimenters read "no hand-written forward pass" more strictly, 06 is the script to look at. Its
  output was used only to choose heads, never as a weight value.
- Activations (pass 2) were used to decide which heads matter. The only number written into weights
  is 0.8, the decoding temperature.
- `diff_summary.json` stores the script path as an absolute path because `write_edit` records what
  `Path(__file__)` gives it; the scripts themselves use relative paths only.
- No gradients, no fitting, no loop over candidate edits, nothing read outside this folder (the
  installed `tinygpt` package was imported only), no agents, no web.
