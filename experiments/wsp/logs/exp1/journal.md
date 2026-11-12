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
