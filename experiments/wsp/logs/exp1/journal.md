# Exp 1 journal: reading 48 attention heads from the weights

**Editor:** Claude Fable 5.2, main session. **Forward passes spent: 0.**
Scripts: `experiments/wsp/analysis/exp1/01..07` (they import only the weight loader and the
tokenizer; the static rule test forbids them from constructing or running the model).
Outputs: `experiments/wsp/analysis/exp1/out/`. Predictions: `results/exp1_predictions.json`.

## What I could and could not read

Allowed by interpretation I2: products of weight matrices along one path, LayerNorm folded as a
fixed gain, a softmax over one score matrix. So I read the **direct path** (token and position
embeddings into each head's query-key and output-value circuits) and **one-step composition**
through an earlier head. I did **not** read anything that passes through an MLP, because that
needs the MLP to be run. From layer 1 on, most of the residual stream has been through an MLP,
so everything below about layers 1 to 5 is a weaker inference than what is said about layer 0.

## Steps and what each showed

1. **Sizes (`01_norms`).** No head is dead by norm. Layer-0 heads have very small
   output-value circuits (Frobenius 0.17 to 0.38, against 1 to 9 in later layers). Query-key
   circuits are small in layers 0 and 1 (top singular value 0.6 to 1.5) and larger from layer 2
   (1.6 to 3.7). Key biases are zero, as they must be (a key bias cannot change a softmax).
2. **Positional query-key circuit (`02`, `03`).** Position embeddings are smooth: the cosine
   between positions i and i-1 is 0.915, and it is the same for i-2 and i-4. Five singular
   directions hold 90% of their energy. A 32-dimensional query-key circuit therefore cannot
   tell the previous position from the one before it. The position-only attention to i-1 is
   0.02 to 0.04 for every head. Scores do fall gently with distance (head 0.7: +1.65 nats at
   distance 0, +0.77 at distance 32), which is a broad recency preference, not a
   previous-token head. Position 0 is special (its embedding is 3.7 times larger than later
   ones), but the direct-path advantage of the position-0 `<|endoftext|>` key is at most
   2.3 nats, about half of what 40% attention needs.
3. **Token circuits (`04`).** Same-token matching is weak everywhere: z-scores of the
   diagonal are below 1 on the direct path and through every earlier head. Copying through the
   output-value circuit is weak too (largest z 1.4, head 0.7). So I see no linear skeleton of
   an induction or duplicate-token circuit.
4. **Content spread (`05`).** Content moves attention scores by about 0.2 to 1.5 nats. If
   scores of different keys differ by sigma because of content, two different texts give
   patterns with cosine about exp(-sigma^2). Layer 0 is read exactly: heads 0.0 and 0.1 come
   out position-dominated, heads 0.3, 0.5, 0.6, 0.7 content-dominated. Deeper, I used twice
   the isotropic estimate, the ratio seen in layer 0.
5. **What a diffuse head writes (`06`).** A layer-0 head that averages over the context writes
   a nearly constant vector of 6% to 18% of the residual norm. From layer 1 on, the constant
   write is 60% or more.

## Predictions (confidence at or above 0.5 counts as a positive)

| Class | Predicted positive | Main reason |
|---|---|---|
| previous token | none (all at or below 0.04) | neighbouring positions are not separable in the embeddings |
| start anchor | none (highest 0.30: heads 4.0, 5.0) | direct-path advantage too small; MLP amplification unreadable |
| induction | none (all at or below 0.03) | no token-before signal to compose with; same-token matching z below 1 |
| duplicate token | none (all at or below 0.06) | same-token z below 1 |
| positional | 0.0, 0.1, 1.0, 1.2, 1.4, 1.5 | small content spread, so the pattern should not depend on the text |
| mostly inactive | 0.2, 0.4 | tiny, nearly constant write; a constant shift is covered by the trained output bias |

## Where I expect to be wrong

- **Start anchor and positional in layers 2 to 5.** Heads that park attention on the first
  position are common in GPT-style models, and whether these do depends on what the MLPs make
  of position 0. If such heads exist I will miss them, and they also count as positional.
- **Previous token.** My "none" rests on the direct path. An MLP could sharpen the position
  code. If the verification finds previous-token heads, the direct-path argument was the wrong
  tool, not just imprecise.
- **Inactive.** The 0.002-nat line is a guess about second-order effects. The ordering
  (0.2 and 0.4 most removable in layer 0) is what the weights support; the level is not.
- **The base rates.** If four of six classes are empty, as I predict, the scored comparison
  reduces to "positional" and "inactive", and the within-layer chance test has little to work
  with.

The reasoning above is mine, written before verification. Whether it describes what the model
does is exactly what the verification measures.

## Verification (run after the predictions were committed at `1f61782`)

`exp1_verify.py`, dev data only, 336 s: attention on 256 natural windows and 64 repeated-random
sequences, and 48 single-head ablations on 500 dev stories. Results:
`results/exp1_verification.json`. These were harness passes, not editor passes.

| Class | Heads truly in class | Predicted | Hit | False alarm | Missed |
|---|---|---|---|---|---|
| previous token | none (largest score 0.33) | none | | | |
| start anchor | none (largest 0.06) | none | | | |
| induction | none (largest 0.03) | none | | | |
| duplicate token | none (largest 0.01) | none | | | |
| positional | 0.0, 0.1, 1.0, 1.4, 1.5 | 0.0, 0.1, 1.0, 1.2, 1.4, 1.5 | 5 | 1 (head 1.2) | 0 |
| mostly inactive | 0.2, 0.4, 1.2, 1.7 | 0.2, 0.4 | 2 | 0 | 2 (heads 1.2, 1.7) |

Micro-precision 0.875, micro-recall 0.78, micro-F1 0.82. Shuffling heads within their layer
gives a mean F1 of 0.29 and reaches 0.82 in 3 of 10,000 shuffles (p = 0.0003). **H1 is met.**

### What supports the result, and what weakens it

- **It was not a leak.** The predictions file is the committed one (the script checks this and
  records the commit). The analysis scripts cannot construct the model (static scan), and the
  only model run before the commit was a smoke test on an untrained network.
- **Two hits were close calls.** Heads 0.1 and 1.0 measured 0.801 and 0.807 against a threshold
  of 0.80, and I had given them 0.55 and 0.52. `exp1_sensitivity.py` (post hoc, not
  preregistered) moves the thresholds: with "positional" at 0.85 the score falls to 5 hits,
  3 false alarms (F1 0.67, p = 0.002); with both thresholds made stricter, precision is exactly
  0.5 and p = 0.015. H1 holds in every variant tried, but the F1 of 0.82 is partly luck.
- **The reading worked only where the weights could be read.** For the "positional" score my
  confidences rank the 16 heads of layers 0 and 1 well (Spearman 0.64, p = 0.004) and the 32
  heads of layers 2 to 5 not at all (0.16, p = 0.20). All nine true positives are in layers 0
  and 1. This matches the limit stated before verification: nothing behind an MLP was read.
- **Four classes were empty, which made "none" an easy call.** The model has no head that
  reaches the previous-token, start-anchor, induction or duplicate-token thresholds. Saying so
  from weights was correct, but those calls add nothing to the scored F1.

### Where the stated mechanism was wrong, although the label came out right

- **Previous token.** I argued that no head can single out the previous position because
  neighbouring position embeddings are almost identical. Six heads in layers 2 to 4 put 25% to
  33% of their attention on the previous token (3.1: 0.33, 3.6: 0.33, 3.2: 0.28). That is
  below the 0.40 line, so "none" stands, but a third of the attention on one position out of a
  hundred is a real previous-token tendency that the direct path cannot produce. The MLPs must
  sharpen position, as I listed under "where I expect to be wrong". My mechanism was incomplete.
- **Start anchor.** I gave heads 4.0 and 5.0 a 0.30 chance because of a positive direct-path
  preference for the first position. Measured attention there is 0.003 and 0.002. Across heads
  my confidence is *negatively* related to the measured score (Spearman -0.74). The direct-path
  number was worse than useless for this class in deep layers.
- **Inactive.** The two heads I named (0.2, 0.4) are among the four that cost less than
  0.002 nats, and the confidence ranks the ablation effects well overall (Spearman 0.63). The
  two I missed are in layer 1 (1.2, 1.7), where I had ruled the class out because the constant
  write is large. So "a large constant write means the head matters" is false for this model.
- **A pattern I did not predict:** the three heads whose positional score I overestimated most
  (0.2, 0.4, 1.2) are all in the "mostly inactive" class. A head the model barely uses seems to
  have a less regular attention pattern than its weights suggest.

Evidence grading for the write-up: the class labels are measured facts. The explanation for
layers 0 and 1 (diffuse, position-dominated attention with small writes) is supported by both
the weights and the measured patterns. The explanations for layers 2 to 5 are not supported.
