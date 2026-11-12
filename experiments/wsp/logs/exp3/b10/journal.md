# Journal — exp3_b10 (editor)

All scripts are run from the workspace root with the study's Python. Forward passes go through the
sandbox only (`passes/pNN_*.py`); everything in `analysis/` is free analysis of weights or of
activations already captured by a logged pass.

## 0. Setup and reading

- Read `TASK.md`, `model_source/`, `harness_source/`. Model: 6 layers, 8 heads of 32 dims, d=256,
  vocab 4096, learned positions, pre-norm, tied unembedding.
- Decision: prompts in `dev/prompts.json` are objects with `tokens`; I fixed the generation test set as
  the first 64 prompts with seed 0 (`passes/common.py`) before seeing any model output, and the loss
  test set as the first 64 dev stories with at least 255 tokens, each as `[EOT] + first 255 tokens`.

## 1. Free analysis of the weights (before any pass)

- `analysis/01_overview.py`: parameter norms; per-head OV and QK Frobenius norms. OV norms grow with
  depth; L5H7 (7.6), L5H6 (5.8), L5H1 (5.4) are the largest writers.
- `analysis/02_copy_scores.py`: token -> same-token logit through each head's OV with the raw embedding
  as input. Mean own-token logit is highest for L5H7 (0.77), L4H6 (0.75), L5H6 (0.69), L5H0 (0.42), but
  z-scores are below 1 and own-token top-10 rates are a few percent: read from the raw embedding, no head
  is a clean copier. Positional attention in layers 0-1 shows only a mild recency slope (wpe is small).
- `analysis/03_data_look.py` (no model): real dev stories, first 200 tokens: loop rate 0.193, rep-4
  0.0176. So "loops" by this metric are common in real text too; the gap to close is 24% -> 19%.
- `analysis/06_ov_eigs.py` (run later, weights only): eigenvalues of the 32x32 token->token OV circuit
  per head. Largest positive (copying) mass: L5H7 3554, L5H6 3496, L4H6 2359, L5H1 2041, L5H0 1901.

## 2. Passes

### Pass 1 — generate, unedited (`passes/p01_baseline_generate.py`)
Purpose: baseline for paired comparison, and model-written text to attribute.
Result: loop rate 0.234, rep-4 0.0255, mean length 162.6, EOT rate 0.72 (64 prompts, seed 0). Matches the
table in TASK.md (24%, 0.027, 163).
`analysis/04_look_generations.py 001`: the 15 looping generations loop on (a) multi-token names
(" Velvet" = 4 tokens, x9; "Bouncy", "Jojo", "lawyer"), (b) re-used noun phrases ("the big red ball",
"big, round thing", "the vacation"), (c) stock dialogue frames (" and said, \"" is the most common
repeated 4-gram overall, 13 repeats over the 64 texts). None is a degenerate endless loop.

### Pass 2 — read with captures on the pass-1 generations (`passes/p02_read_generated.py`)
Purpose (hypothesis): repeat-completing tokens get their logit from a small set of late induction/copy
heads. Rows are prompt + generation, padded to 212 tokens with EOT + unrelated dev text (padding is
masked out of every statistic). mean_nll 1.14 (own text, includes padding).

### Pass 3 — read with captures on 64 real dev stories (`passes/p03_read_real.py`)
Purpose (hypothesis): same heads carry copying on real text and the model is calibrated there; also the
paired loss baseline. mean_nll 1.7626 on these 64 long stories (16,320 predicted tokens).

Analysis of passes 2 and 3 is in section 3.

### Pass 4 — generate, unedited, T=1.0 (`passes/p04_generate_T1.py`)
Purpose (hypothesis, written before the call): the excess repetition is produced by the sampler's
sharpening, so at T=1.0 (same prompts, seed, top-k) loop rate and rep-4 fall to about the real-story level.
Result: loop rate 0.172 (was 0.234), rep-4 0.0163 (was 0.0255), mean length 160.8 (was 162.6), EOT rate
0.73. Real stories: 0.193 / 0.0176. Prediction confirmed: at its own temperature the model repeats no more
than real stories do.

**Decision recorded here, before computing anything about its cost:** the only weight-space handle on
this cause is the model's own output scale (final LayerNorm gain and bias: logits scale exactly with
them). Candidate edit B = multiply `ln_f.weight` and `ln_f.bias` by s = 0.9. The value is chosen by
reasoning, not from data: s = 0.8 would cancel the sampler's T=0.8 completely, but a rescale of logits
costs loss quadratically in (1-s) (about (1-s)^2/2 times the variance of log p under the model), and
4x less at 0.9 than at 0.8; 0.9 is the round value that undoes half of the sharpening
(effective temperature 0.8/0.9 = 0.89). I will not tune s on evaluations.

### Pass 5 — generate, probe A = heads L4H5, L4H4, L5H7, L5H6 zeroed (`passes/p05_probe_generate.py`)
Purpose (hypothesis): causal test of "these four heads drive repetition". My written prediction was that
the loop rate would stay near 0.234 because their direct log-odds push is small.
Result: loop rate 0.125, rep-4 0.0129, mean length 161.1, EOT rate 0.78. **My prediction was wrong**: removing
the four heads halves both repetition measures (15 -> 8 looping generations of 64, same prompts and
seed). The direct-path estimate in `analysis/09` understated their effect (indirect paths through the last
MLPs, and feedback: fewer repeats create fewer repeated contexts).
Reading the texts (`analysis/04_look_generations.py 005 4 44`): the ablated model drops rare entities. Prompt
44 introduces "a dog named Velvet" and never mentions Velvet again; it continues with stock "Lily and her
friends". So the same heads carry legitimate in-context copying (names), and full ablation is expected to
cost real-text loss. Ablation is a probe, not a candidate.

**Decision before pass 6:** candidate A1 = the same four heads scaled by 0.5 (`edits/copyheads_half.py`).
Reason for the value: 0 removes entity memory outright and overshoots (0.125 is below the real-story
0.19); 1 is the unedited model; the midpoint is the natural first candidate. Because the trained model is
at a stationary point for any output scale (`analysis/08`: first-order loss effects ~0), the loss cost
should grow like (1-a)^2 while the behavioural effect grows like (1-a), so a partial scale should keep
a larger share of the benefit than of the cost. Pass 6 reads its loss first; a generate pass follows only
if the loss is inside the 0.02 limit with margin.

### Pass 6 — read, A1 = four heads x0.5, same 64 real stories as pass 3 (`passes/read_edit.py copyheads_half`)
Purpose (hypothesis): cost below 0.02 nats, and concentrated on tokens that legitimately repeat earlier text.
Result: loss 1.7852 vs 1.7626, **+0.0225 nats** (bootstrap over stories, 95%: +0.018 to +0.027). Over the limit.
Second part of the hypothesis also wrong: 4-gram-repeating tokens lose most per token (+0.077) but are only
9% of the total cost; all other positions lose +0.022 on average. Cost is larger in the second half of a
story (+0.029) than the first (+0.016). So these heads do general in-context work (re-using earlier words
of any kind), not only verbatim 4-gram copying. A1 is rejected without spending a generate pass on it.

**Revision 1 (A1 -> A2), rationale:** A1 breaks the loss limit. The cost of scaling is second order
(stationary point), so halving the step should cut the cost to about a quarter (prediction: about +0.006)
while keeping about half of A1's effect on repetition. A2 = the same four heads x0.75
(`edits/copyheads_075.py`). The value is "half of A1's step", not a number solved from the measured loss.
Prediction for generation (from the probe, assuming a linear dose-response): about a quarter of the ablation
effect, i.e. loop rate about 0.21 and rep-4 about 0.022 on the 64 test prompts; too small to be significant
on 64 texts alone, so the last passes go to a second, disjoint prompt set.

### Pass 7 — read, A2 = four heads x0.75, same 64 real stories (`passes/read_edit.py copyheads_075`)
Purpose (hypothesis): second-order cost, so about a quarter of A1's, near +0.006.
Result: loss 1.7683 vs 1.7626, **+0.0057 nats** (95%: +0.0036 to +0.0078). Prediction confirmed (0.0225/4 =
0.0056). By position: 4-gram-repeating tokens +0.031, repeatable context but different continuation
-0.008, elsewhere +0.005; first half of story +0.003, second half +0.008.

### Pass 8 — generate, A2, standard decoder, same 64 prompts and seed (`passes/gen_edit.py edited/copyheads_075 0`)
Purpose (hypothesis): about a quarter of the ablation effect: loop about 0.21, rep-4 about 0.022, length same.
Result: loop rate 0.203, rep-4 0.0238, mean length 163.9, EOT rate 0.66.
`analysis/11_paired.py 001 008`: differences -0.031 loop (95% -0.17 to +0.11), -0.0017 rep-4 (-0.010 to
+0.006), -0.010 rep-2 (-0.027 to +0.007), length +1.3. Right direction, right order of size, **not
significant**. Important side finding: the "paired" design does not pair. No two texts are identical between
any two calls, and loops are lost and gained in similar numbers (11 lost, 9 gained); the trajectories
diverge at the first changed token. So 64 texts carry +-0.14 on loop rate and +-0.008 on rep-4, and two more
generate passes could not have resolved an effect of the expected size (about 0.003 on rep-4). I changed
the plan for the last two passes accordingly.

### Pass 9 — read, A2, on the unedited model's own pass-1 texts, with logits (`passes/p09_read_edit_on_generated.py`)
Purpose (hypothesis): low-noise version of the pass-8 test. On identical contexts, the standard sampler
(T=0.8, top-k 40) under A2 gives less probability to tokens that complete an already-seen 4-gram;
prediction a 5-15% fall.
Result (`analysis/12_sampler_copy_prob.py work/outputs/read_call009.npz`): at the 722 repeatable contexts
the probability falls from 0.4261 to 0.4193, ratio **0.984** (bootstrap over texts 0.978 to 0.989). The fall
is real but much smaller than I predicted. `analysis/13_temperature_onestep.py` (unedited logits only) gives
the scale: the same one-step measure falls 6.5% when T goes from 0.8 to 1.0, while sampled rep-4 fell 36%
(pass 4), so the one-step measure understates the sampled effect about five-fold (repeats feed on
repeats). On that scale A2 is worth roughly 7-9% of rep-4.

### Pass 10 — read, probe A (ablation), same texts, with logits (`passes/p10_read_probe_on_generated.py`)
Purpose (hypothesis): dose-response is linear and the one-step measure understates sampled rep-4 by about
5x; prediction for the ablation: ratio 0.90-0.94.
Result: ratio **0.888** (0.857 to 0.917), just outside my interval on the convex side. One-step -11.2% against
sampled -49% (pass 5): factor 4.4, consistent with the temperature factor 5.5. But the dose-response is
convex: the 0.75 edit takes a quarter of the step and gets a seventh of the one-step effect (1.6 of 11.2).
Sampled rep-4 estimate for A2: about -7% (a seventh of -49%, or 1.6% x 4.4 to 5.5), which agrees with the
-7% point estimate of pass 8.

Budget: 10 of 10 passes used, none refused. Ledger verifies (`work/ledger.jsonl`).

## 3. Analysis of captured activations (free; unedited model only unless stated)

Full printed outputs are in `analysis/outputs/`. Key numbers:

- `05_attribute.py` (passes 2, 3). Repeatable contexts (trigram seen before) are 6.9% of generated
  positions and 6.6% of real ones; the repeat is taken 41% / 40% of the time. On **real** text the model
  gives the most recent continuation probability 0.312 on average while real text takes it 0.364 of the
  time: the copy pathway is calibrated to slightly under-confident, not over-active. Heads with the largest
  copy-specific direct logit attribution (to the actual token at 4-gram-repeat positions minus elsewhere):
  generated text L5H7 +0.52, L5H6 +0.49, L4H5 +0.28, L5H1 +0.23, L5H4 +0.16; real text L5H7 +0.23, L4H5
  +0.14, L5H6 +0.13, L5H4 +0.12, L5H1 +0.10. The same heads put about twice the average attention on the
  position after an earlier occurrence of the current token (0.14-0.20), so they are diffuse, not sharp,
  induction heads.
- `06_ov_eigs.py` + `07_copy_mode_split.py`: hypothesis that the copy-specific part of these heads' output
  lives in the OV eigen-modes with positive real part (weights-only definition, threshold fixed before
  looking). **Rejected**: the positive modes carry most of the heads' ordinary contribution too, and the
  copy-specific excess is split between both parts (e.g. L5H7 on generated text: +0.25 in copy modes,
  +0.27 in the rest). No surgical "remove the copy modes" edit was built.
- `08_selectivity.py`: first-order effect on real-text loss of scaling any component, direct path: all
  within +-0.02 nats per unit scale, most below 0.002. The model sits at a stationary point, so cost is
  second order. (The "copy_push" column of this script saturates when p is near 1 and was not used.)
- `09_second_order.py`: direct log-odds given to the 4-gram-completing token at repeatable contexts
  (p < 0.95): only four heads are positive on both generated and real text: L4H5 +0.103 / +0.079, L5H7
  +0.088 / +0.092, L5H6 +0.038 / +0.038, L4H4 +0.036 / +0.041. L5H1 and L5H4, which ranked high in `05`,
  are ~0 here. The last MLPs and several heads (L5H0, L5H3, L5H5, L4H7) push the other way. This is the
  set used for probe A. The same script's direct-path magnitudes predicted a negligible effect of removing
  them, which pass 5 refuted; I trust the script for the sign and the ranking only.
- `10_varentropy.py`: mean loss 1.7626, mean entropy 1.7675 (calibrated), mean variance of log p 3.28;
  predicted cost of the untested edit B (logits x0.9): +0.017 nats.

## 4. Edits and revisions

| Name | What | Status |
|---|---|---|
| `probe_copyheads_ablate` | L4H5, L4H4, L5H7, L5H6 output x0 | diagnostic probe, passes 5 and 10; never a candidate |
| `copyheads_half` (A1) | same heads x0.5 | rejected at pass 6: loss +0.0225 |
| `copyheads_075` (A2) | same heads x0.75 | revision 1; passes 7, 8, 9; **final** |
| `final_gain_090` (B) | `ln_f.weight`, `ln_f.bias` x0.9 | written after pass 4, **never run through the model**; kept only as a record |

One revision used (A1 -> A2) of the three allowed. I did not make a further revision toward a factor
between 0.5 and 0.75, although the measured costs (0.0225, 0.0057) fit a quadratic that would put such a
factor inside the limit: that would be solving a weight value from measured losses and bracketing a
strength by evaluation, which rules 2-3 and I1 forbid.

## 5. Decisions I made alone

- Counted the ablation as a probe, not a revision, because it was never proposed as the final edit.
- Chose the four heads from `09` (sign-consistent on both texts) rather than the five from `05`.
- The choice of 0.75 after seeing A1's loss is informed by a model output (it had to be: A1 broke the
  limit). I kept the value to "half the previous step" and did not solve for it. A reader who counts this
  as a weight value influenced by outputs should know it is the only place where that happened.
- Predictions that failed are left in the ledger purposes and above: pass 5 (expected no effect of
  ablation), pass 6 (expected cost under 0.02 and concentrated on repeats), pass 9 (expected 5-15%).
- `analysis/09_second_order.py` takes about four minutes; its output was saved from the original run.
