# Diagnosis: what drives repetition, and the edit

Final version: `copy8_x080` (`edits/copy8_x080.py`). 40 of 50 forward passes used.

## Summary

Repetition under sampling is in-context copying: the model re-emits n-grams that already occur in the
story (names, noun phrases such as "the big red ball", dialogue frames). Eight attention heads in
layers 3-5 carry that copying. They are not broken: on real stories the model's probability of
repeating an n-gram is about right at temperature 1. The excess appears when the decoder sharpens the
distribution (T=0.8, top-k 40) and the model then conditions on its own, more repetitive, text.

The edit multiplies the output of those eight heads by 0.8, the decoding temperature. On dev data it
lowers loop rate from 26.1% to 17.1% and rep-4 from 0.0282 to 0.0213 (620 paired generations), raises
loss by about 0.007 nats (tolerance 0.02) and leaves length unchanged (+1.5%).

## The components

Core four: L5H6, L5H7, L5H1, L4H5. Secondary four: L4H6, L4H4, L3H0, L5H4.

Evidence, in the order it was gathered:

1. **Weights (free).** With the token embedding passed through the layer-0 MLP as input, the OV circuit
   of these heads maps a token to its own logit: attending to token *t* makes *t* the top-1 output for
   56% of the vocabulary in L5H7, 42% in L5H6, 41% in L4H6, 27% in L4H4, 21% in L5H1, 17% in L3H0
   (`analysis/08_out.txt`, part 1). Their QK circuit matches the current token against what the layer-0
   local-context heads (L0H3, L0H6, L0H7, which look 0-3 positions back) write about a nearby earlier
   token: strongest in L4H5, L5H6, L5H7, L5H4, L4H6 (part 3). That is the induction pattern: look at
   the position after an earlier occurrence of the current token, and copy what is there.
   L4H5 is weak on the direct copy score and strong on the induction score.
2. **Pass 2 (capture on 32 generations and 32 real stories).** At positions where the true next token
   completes an already-seen 4-gram, these heads put 16-35% of their attention on earlier copies of
   that next token (uniform would be about 3.5%). Their direct contribution to the true next token's
   logit is 0.74, 0.73, 0.39, 0.38 logits higher at such positions than at new-token positions for the
   core four, and 0.17, 0.17, 0.15, 0.06 for the secondary four. No other head exceeds 0.06
   (`analysis/07_out.txt`).
3. **Passes 3-7 (zero-ablation, same prompts and random numbers).** Zeroing the core four: loop rate
   23.4% -> 10.9%, rep-4 0.0255 -> 0.0124, loss +0.146. Zeroing four other heads of the same layers
   (L5H0, L5H2, L5H3, L4H1): loop rate 23.4% -> 23.4%, rep-4 0.0255 -> 0.0237, loss +0.034.
4. **Passes 18-20.** Zeroing the secondary four: loop rate 26.6% -> 16.4%, rep-4 0.0279 -> 0.0227 on
   128 generations, loss +0.058.

So copying is causal for repetition, and it is also useful: removing it costs 0.06-0.15 nats.

## Why the factor is 0.8

Pass 2 also measured calibration. On real stories the model at T=1 puts 0.0105 probability per token on
tokens that would complete a seen 4-gram; the true rate is 0.0140 (about 70 events in 32 stories, so the
difference is within noise). For seen bigrams: 0.128 predicted, 0.135 actual. The model does not
over-predict repetition. Under T=0.8 and top-k 40 the same positions get 0.0116 and 0.143. On the
model's own generations the 4-gram repeat rate is 0.029, twice that of real text.

The conclusion I drew: no component needs repair. Decoding at T=0.8 multiplies every logit
contribution by 1/0.8, the copy heads' included, and the repeats this adds feed back into later
context. Multiplying the copy heads' output by 0.8 gives their contribution under T=0.8 decoding the
size it has in the trained model at T=1. The number 0.8 is the decoding temperature from TASK.md. It
was not computed from activations or outputs; the measurements only told me which heads and that the
problem is amplification, not miscalibration.

Scaling a head's 32 columns in `attn.proj.weight` scales exactly what that head adds to the residual
stream. Because the model sits near a loss optimum, a small change of a calibrated component costs
loss only at second order, which is why 20% attenuation of all eight costs about 0.007 nats while removing
the core four alone costs 0.146.

## The edits and their results

| Version | Heads | Loss, stories 1-64 | Loss, stories 65-128 | Loop rate | rep-4 | Length |
|---|---|---|---|---|---|---|
| Unedited | none | 1.6500 | 1.5126 | 24.5% / 27.7% | 0.0275 / 0.0289 | 160.7 / 158.3 |
| v1 `copy4_x080` | core four | +0.0038 | not run | 22.9% / not run | 0.0239 / not run | 161.9 / not run |
| v2 `copy8_x080` (final) | all eight | +0.0069 | +0.0078 | 17.7% / 16.5% | 0.0218 / 0.0207 | 164.4 / 159.3 |

Generation cells are seed 0 / seed 1 on the 310 dev prompts (T=0.8, top-k 40). Paired differences of v2
against the unedited model, with a bootstrap over generations:

| Measure | Seed 0 | Seed 1 | Pooled (620) |
|---|---|---|---|
| Loop rate | -6.8 points [-12.9, -0.7] | -11.3 points [-17.4, -5.5] | -9.0 points [-13.4, -4.5] |
| rep-4 | -0.0056 [-0.0095, -0.0017] | -0.0082 [-0.0120, -0.0044] | -0.0069 [-0.0097, -0.0042] |
| Length | +2.3% | +0.6% | +1.5% [-0.8%, +3.7%] |

v1 alone: loop rate -1.6 points [-7.7, +4.8], rep-4 -0.0036 [-0.0075, +0.0003] on seed 0.

Greedy decoding, prompts 0-63 (passes 39-40): loop rate 42.2% -> 28.1%, long-loop rate 14.1% -> 1.6%,
rep-4 0.111 -> 0.043, length 165 -> 158 (more stories end with the stop token: 72% -> 88%).

There was one revision (v1 to v2). Its rationale: v1's reason applies to every head that carries the
copy signal; v1 was limited to the four that had been causally tested, and passes 18-20 then tested the
other four. The factor was not changed.

## What I am unsure of

- **Size of the effect.** I predicted 2-6 points of loop rate for v2 from the ablations and measured 9.
  The intervals are wide; the sealed result may be nearer the prediction. rep-4 is the steadier measure.
- **Loss margin.** +0.007 was measured on 128 dev stories cut to 256 tokens with a leading stop token. I
  do not know how the sealed loss is computed. The two blocks agree (+0.0069, +0.0078), so I expect the
  sealed delta to be well under 0.02, but I did not measure a confidence interval per story.
- **Name consistency.** Copying is also how the model keeps a multi-token name intact. In v2 samples I
  saw "Clvet" for "Velvet" and a story that swapped "Jojo" for "Anna"; the unedited model makes the same
  kind of slip ("Jojody", "Melvet"). I did not count these. Part of the loss increase is this cost.
- **Loop rate now below real stories** (17% against 19-21%), while rep-4 is still above (0.021 against
  0.018). The edit removes repetition that real stories also have; it is a trade bought with loss, not a
  repair of a fault.
- **The temperature argument is approximate.** LayerNorm follows the heads, the L3/L4 heads also feed
  later layers, and top-k is not accounted for. The greedy result (larger effect than I predicted) shows
  the mechanism is not only temperature compensation: attenuating the copy heads weakens the
  self-reinforcement of copying in general.
- **Tied to this decoder.** The factor is the temperature of this evaluation. At T=1 the edit makes the
  model under-copy slightly.
- **Specificity of the strength test.** I did not run a random edit of the same size at factor 0.8 (the
  final evaluation does). The zero-ablation control (passes 6-7) is my only specificity evidence.
- **Weights-only step.** The "extended embedding" in `analysis/08` applies the layer-0 MLP to each
  vocabulary vector on its own. I judged this to be weight analysis, not a forward pass on a sequence.

## Considered and not done

- Scaling `ln_f` by 0.8 would cancel the temperature for every logit. It changes the whole model's
  calibration and would raise loss well beyond tolerance; it is not an edit of the components that drive
  repetition.
- A stronger factor than 0.8. The measured cost leaves room under 0.02, but I had no reason for another
  number other than fitting it to the loss measurements, which the rules exclude.
