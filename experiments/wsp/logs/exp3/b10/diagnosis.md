# Diagnosis

Final version: `copyheads_075` — the output of attention heads L4H5, L4H4, L5H7 and L5H6 scaled by 0.75.
It costs +0.006 nats on real dev stories and is expected to lower rep-4 by roughly 7%. That benefit was
not shown significantly on sampled dev text; it rests on a low-noise one-step measurement and on the
ablation of the same heads. Whether so small a gain is worth an edit is a judgement call; the case for
`unedited` is stated at the end.

## What drives the repetition

**1. Most of the gap to real stories comes from the sampler, not from a circuit.**

- Sampled at T=1.0 instead of 0.8 (same prompts, seed, top-k 40), the unedited model gives loop rate
  0.172 and rep-4 0.0163, against 0.234 and 0.0255 at T=0.8. Real stories give 0.193 and 0.0176 (pass 4).
- On real stories the model is not over-confident about repeating. Where the last three tokens have
  occurred before, it gives the earlier continuation probability 0.31 on average, and real text takes it
  0.36 of the time (pass 3).
- The loops themselves are ordinary: multi-token names (" Velvet" is four tokens), re-used noun phrases
  ("the big red ball"), and stock frames (" and said, \""). No 8-token loop occurred in any call.

**2. Within the model, four late heads are the causal lever for re-using earlier text.**

- L4H5, L5H7, L5H6 and L4H4 are the only heads whose direct output raises the log-odds of the token
  that would complete an already-seen 4-gram, on both model-written and real text (+0.10, +0.09, +0.04,
  +0.04 logits). They attend about twice as much as other heads to the position after an earlier
  occurrence of the current token (passes 2 and 3).
- Zeroing all four halves repetition under the standard decoder: loop rate 0.234 to 0.125, rep-4 0.0255 to
  0.0129 (95% interval of the rep-4 change: -0.020 to -0.006), length unchanged (pass 5).
- They are not a repetition fault. The ablated model forgets entities: a story that introduces "a dog
  named Velvet" never mentions Velvet again. Scaling them by 0.5 costs +0.0225 nats on real stories, and
  91% of that cost falls on tokens that do not complete a repeated 4-gram (pass 6).
- From the weights alone these heads are only partly visible. L5H7 and L5H6 have the largest positive
  eigenvalue mass of the token-to-token OV circuit and the largest OV norms; L4H5 and L4H4 do not stand out.
  L4H6 and L5H0 look like copiers in the weights but do not promote repeats in the activations.

**3. Hypotheses that failed.**

- "The copy-specific output of these heads lives in the positive eigen-modes of their OV circuit" —
  rejected on captured activations; the split does not separate copying from the heads' other work.
- "Removing the heads will barely change repetition, because their direct logit push is small" — my
  prediction before pass 5, refuted. Direct-path attribution understated their effect.
- "The cost of scaling them falls on repeated tokens" — refuted in pass 6.

## Reasoning behind each edit

| Edit | Reasoning | Evidence |
|---|---|---|
| `probe_copyheads_ablate` (x0) | Upper bound for any "scale these heads down" edit; a probe only | Loop 0.125, rep-4 0.0129; entity memory lost |
| `copyheads_half` (x0.5) | Midpoint between unedited and ablated; cost is second order in the step, effect first order | Loss +0.0225 (95% +0.018 to +0.027): over the limit, rejected |
| `copyheads_075` (x0.75) | Revision 1: half the step, so about a quarter of the cost | Loss +0.0057 (95% +0.0036 to +0.0078), as predicted |
| `final_gain_090` (logits x0.9) | Undo half of the sampler's sharpening through the model's own output scale | Never run through the model; predicted cost +0.017 from the variance of log p. Not proposed |

All factors are scalars chosen by the reasoning in the table. The head set was chosen from activations,
which the rules allow; no weight value was computed from activations or outputs. The step from 0.5 to
0.75 was prompted by a measured loss and is "half the previous step", not a value solved from that loss.

## Evidence for the final edit

| Measure | Unedited | `copyheads_075` | Note |
|---|---|---|---|
| Loss, 64 real dev stories of 256 tokens | 1.7626 | 1.7683 | +0.0057, paired per token |
| Loop rate, 64 prompts, seed 0 | 0.234 | 0.203 | 95% of difference: -0.17 to +0.11 |
| rep-4, same texts | 0.0255 | 0.0238 | 95% of difference: -0.010 to +0.006 |
| Mean length | 162.6 | 163.9 | |
| EOT rate | 0.72 | 0.66 | 95% of difference: -0.20 to +0.08 |
| Sampler probability of completing a seen 4-gram, identical contexts | 0.4261 | 0.4193 | ratio 0.984 (0.978 to 0.989) |

The last row is the only significant benefit. To read it as an effect on sampled text I used two
calibrations of the same one-step measure: raising T to 1.0 moves it by 6.5% and sampled rep-4 by 36%;
ablating the heads moves it by 11.2% and sampled rep-4 by 49%. On that scale (4.4 to 5.5) the edit is worth
about 7-9% of rep-4, which matches the pass-8 point estimate of -7%.

## What I am unsure of

- **Size of the benefit.** About -7% on rep-4 and one to three points of loop rate is an estimate built on
  a one-step measurement and a scaling factor from two other interventions. It was not demonstrated on
  sampled text; the dev sample cannot distinguish it from zero.
- **Dose-response is convex.** A quarter of the step buys about a seventh of the ablation's one-step
  effect, while cost is quadratic. A factor between 0.5 and 0.75 would probably do more inside the loss
  limit; I did not build one, because choosing it would mean solving a weight value from measured losses.
- **Loss on other data.** My loss sample is 64 long stories. The cost is larger late in a story (+0.008 in
  the second half, +0.003 in the first), so shorter test stories should cost less, but I measured only
  this sample.
- **Sampled comparisons are unpaired in practice.** Common random numbers do not keep texts aligned once
  one token differs, so every sampled difference here has the noise of independent samples.
- **Kind of fix.** The edit weakens a mechanism that is calibrated on real text, in exchange for less
  repetition under a sharpened sampler. It does not repair a fault, because I found none in the weights.
  If the standard for an edit is "removes a defect", the right answer is `unedited`, and the finding is that
  the repetition is a decoding-temperature effect.
