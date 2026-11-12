# Journal

Editor run in `exp3_b50`. Budget 50 forward passes; 40 used, 0 refused. Final version: `copy8_x080`.
Every script is in `analysis/` or `edits/` and runs from the workspace root. Raw printed outputs are in
`analysis/*_out.txt` and `analysis/*compare*.txt`, and are copied in the appendix below.
Predictions were written to `analysis/predictions.md` before the passes that tested them.

## Decisions taken alone

- Loss rows: `<eot>` + dev story + `<eot>`, cut at 256 tokens. Story 0 of the file starts mid-sentence, so
  I used stories 1-64 and 65-128.
- Generation sets: the 310 dev prompts in five blocks of 64 (the last has 54), seed 0, then seed 1.
  Comparisons are paired by prompt and seed (the sandbox uses common random numbers).
- Zero-ablation checkpoints (`edited/test_*`) are hypothesis tests built by scripts in `analysis/`. They
  are not candidate edits and I do not count them as revisions.
- `analysis/08` applies the layer-0 MLP to each vocabulary vector on its own ("extended embedding"). I
  treated this as weight analysis: no token sequence, no attention, no positions.
- I stopped with 10 passes unspent because the final version had been replicated and I had no further
  hypothesis that needed a pass.

## Steps

**0. Reading.** `TASK.md`, `model_source/`, `harness_source/`. Model: 6 layers, 8 heads, d=256, pre-norm,
tied embeddings.

**1. Weights: inventory (`analysis/01_norms.py`).** Layer-0 heads have tiny OV norms (0.3-0.7); the largest
OV norms in the model are L5H7 (7.6), L5H6 (5.8), L5H1 (5.4).

**2. Weights: OV copy score on bare embeddings (`analysis/02_ov_copy.py`).** Weak everywhere. Best late
heads: L4H6, L5H6, L5H0, L5H7. Bare embeddings are a poor stand-in for a late-layer residual; this pass
only gave candidates.

**3. Dev data (`analysis/03_dev_data.py`).** 310 prompts of 4-12 tokens, 5000 stories. Real stories, first
200 tokens: loop rate 21%, rep-4 0.018. (First run failed: prompts are dicts with `tokens`; fixed.)

**4. Pass 1, generate, unedited, prompts 0-63 seed 0.** Purpose: baseline and loop examples. Result: loop
rate 0.234, rep-4 0.0255, length 162.6. Reading the loops (`analysis/05_look_loops.py`): they are
multi-token names ("Velvet", "Bouncy"), noun phrases ("the big red ball"), dialogue frames ("and said, ").
No token stutter. So loops are re-use of earlier n-grams of the same story.

**5. Pass 2, read with capture** (attention, head outputs, MLP outputs, residuals, logits; 32 generations
from pass 1 and 32 real stories, 160 tokens each). Purpose: which heads copy from context; is the repeat
probability calibrated. Analysis in `analysis/07_capture_analysis.py`:
- Direct logit attribution to the true next token, repeated-4-gram positions minus new-token positions:
  L5H6 0.74, L5H7 0.73, L5H1 0.39, L4H5 0.38, L4H6 0.17, L3H0 0.17, L5H4 0.15; everything else <= 0.06
  among heads. (MLP0 is 0.71 but it is the token's own identity path, 2-3 logits everywhere.)
- Attention to earlier copies of the true next token at those positions: L5H7 0.35, L5H6 0.32, L4H5 0.25,
  L5H1 0.25, L4H4 0.23, L4H6 0.18, L3H0 0.18, L5H4 0.16; uniform is about 0.035.
- Calibration on real stories: 4-gram repeat mass 0.0105 predicted at T=1, 0.0140 actual; bigram 0.128
  against 0.135. Not over-predicting. Under T=0.8/top-40: 0.0116 and 0.143.
- Generated text repeats 4-grams at 0.029 per token, real text at 0.014.
My weights-only candidate L5H0 was wrong; L5H1 and L4H5 were missed by step 2.

**6. Weights again (`analysis/08_weights_circuit.py`).** With the extended embedding the copy OV shows
clearly (own token top-1 for 56% of vocabulary in L5H7, 42% L5H6, 41% L4H6, 27% L4H4, 21% L5H1, 17% L3H0).
Layer-0 heads H3, H6, H7 look 0-3 positions back on positional grounds. Induction QK match through those
heads is strongest in L4H5, L5H6, L5H7, L5H4, L4H6. Weights and capture agree on the same eight heads.

**7. Passes 3-7, zero-ablation test (`analysis/09_pass_ablation.py`).**
- Pass 3 read, unedited, stories 1-64: loss 1.64999.
- Pass 4 generate, core four zeroed: loop 0.109, rep-4 0.0124, length 155.9.
- Pass 5 read, core four zeroed: loss 1.79589 (+0.146).
- Pass 6 generate, control four zeroed (L5H0, L5H2, L5H3, L4H1): loop 0.234, rep-4 0.0237.
- Pass 7 read, control four zeroed: loss 1.68383 (+0.034).
The copy heads are causal for repetition and specific; copying is worth 0.146 nats, so removal is out.

**8. Edit v1, `copy4_x080`.** Core four times 0.8. Reason for 0.8: it is the decoding temperature; the
model is calibrated at T=1 and decoding multiplies the copy contribution by 1.25. Prediction written first:
loss +0.003 to +0.012, loop rate -2 to -3 points, rep-4 -0.002 to -0.003.
- Passes 8-11 generate, unedited, prompt blocks 1-4 seed 0: loop 0.297, 0.234, 0.234, 0.222.
- Pass 12 read, v1, stories 1-64: 1.65383 (+0.0038).
- Passes 13-17 generate, v1, blocks 0-4 seed 0.
- Paired on 310 rows: loop 24.5% -> 22.9% (CI -7.7 to +4.8 points), rep-4 0.0275 -> 0.0239
  (CI -0.0075 to +0.0003), length +0.8%. As predicted, but loop rate unresolved.

**9. Passes 18-20, zero-ablation of the secondary four** (`analysis/13_pass_ablation_secondary.py`).
Prediction first: rep-4 -0.003 to -0.008 on 128 rows, loss +0.03 to +0.08.
- Passes 18, 19 generate, blocks 0, 1: loop 0.125, 0.203; rep-4 0.0190, 0.0265.
- Pass 20 read: 1.70843 (+0.058).
- Paired on 128 rows: loop -10.2 points (CI -19.5 to -0.8), rep-4 -0.0051 (CI -0.0111 to 0.0000),
  rep-2 -0.022 (CI excludes 0). Confirmed.

**10. Revision 1: edit v2, `copy8_x080`.** Rationale: the temperature reason covers every head that carries
the copy signal; the secondary four are now causally tested too. Same factor. Prediction first: loss
+0.005 to +0.010, rep-4 -0.004 to -0.006, loop -2 to -4 points.
- Pass 21 read, v2, stories 1-64: 1.65693 (+0.0069).
- Passes 22-26 generate, v2, blocks 0-4 seed 0.
- Paired on 310 rows: loop 24.5% -> 17.7% (CI -12.9 to -0.7 points), rep-4 0.0275 -> 0.0218
  (CI -0.0095 to -0.0017), length +2.3%. Loop rate fell more than predicted.

**11. Replication, passes 27-38.** Because the loop-rate drop exceeded my prediction, I repeated on seed 1 and
on new stories. Prediction first: rep-4 -0.003 to -0.007, loop -2 to -6 points, loss +0.005 to +0.010.
- Pass 27 read, unedited, stories 65-128: 1.51264. Passes 28-32 generate, unedited, seed 1.
- Pass 33 read, v2, stories 65-128: 1.52046 (+0.0078). Passes 34-38 generate, v2, seed 1.
- Seed 1: loop 27.7% -> 16.5% (CI -17.4 to -5.5 points), rep-4 0.0289 -> 0.0207 (CI -0.0120 to -0.0044).
- Pooled 620 rows: loop 26.1% -> 17.1% (CI -13.4 to -4.5), rep-4 0.0282 -> 0.0213 (CI -0.0097 to -0.0042),
  length +1.5%.
(First run of the comparison failed on a wrong call number; no pass was involved.)

**12. Greedy check, passes 39-40.** Prediction first: smaller relative effect than under sampling (loop
0 to -10 points, rep-4 -10 to -25%). Result on prompts 0-63: loop 42.2% -> 28.1%, long loops 14.1% -> 1.6%,
rep-4 0.111 -> 0.043 (-61%), length 165 -> 158, stop-token rate 72% -> 88%. My prediction was wrong: the
effect is larger under greedy. The edit weakens the self-reinforcement of copying in general, not only the
temperature amplification.

**13. Qualitative read of v2 samples.** Stories are coherent. Name slips occur ("Clvet" for "Velvet", "Anna"
appearing in a "Jojo" story); the unedited model has the same kind ("Jojody", "Melvet"). Not counted.

**14. Wrap-up.** Both edit scripts were re-run from the workspace root and reproduce the same checkpoint
hashes (`copy8_x080` 0bae6140..., `copy4_x080` 39b9e452...). `final.txt` = `copy8_x080`.

## Pass ledger

| Passes | Kind | Weights | What |
|---|---|---|---|
| 1 | generate | unedited | prompts 0-63, seed 0 |
| 2 | read + capture | unedited | 32 generations + 32 real stories |
| 3 | read | unedited | stories 1-64 |
| 4, 5 | generate, read | core four zeroed | ablation test |
| 6, 7 | generate, read | control four zeroed | ablation control |
| 8-11 | generate | unedited | blocks 1-4, seed 0 |
| 12-17 | read, 5 generate | v1 | stories 1-64; blocks 0-4, seed 0 |
| 18-20 | 2 generate, read | secondary four zeroed | ablation test |
| 21-26 | read, 5 generate | v2 | stories 1-64; blocks 0-4, seed 0 |
| 27-32 | read, 5 generate | unedited | stories 65-128; blocks 0-4, seed 1 |
| 33-38 | read, 5 generate | v2 | stories 65-128; blocks 0-4, seed 1 |
| 39, 40 | generate | unedited, v2 | greedy, prompts 0-63 |

## Rules

1. No gradients, no training. 2. No fitting: the only number written into weights is 0.8, the decoding
temperature. 3. No search: one factor, two head sets, one revision, each with a reason and a prediction
written beforehand. 4. All model runs went through the sandbox; 40 passes, none refused; the ledger was not
touched. 5. No sealed data read. 6. Only `checkpoint/` was read; edited checkpoints are under `edited/`.
I read nothing outside the workspace except importing `tinygpt`; no agents, no web.

# Appendix: raw outputs

## analysis/01_out.txt

```
wte.weight                       (4096, 256)    norm=  63.320
wpe.weight                       (256, 256)     norm=   5.520
blocks.0.ln1.weight              (256,)         norm=   8.834
blocks.0.ln1.bias                (256,)         norm=   1.327
blocks.0.attn.qkv.weight         (768, 256)     norm=  21.511
blocks.0.attn.qkv.bias           (768,)         norm=   4.093
blocks.0.attn.proj.weight        (256, 256)     norm=   4.091
blocks.0.attn.proj.bias          (256,)         norm=   0.091
blocks.0.ln2.weight              (256,)         norm=  15.276
blocks.0.ln2.bias                (256,)         norm=   1.965
blocks.0.mlp.fc.weight           (1024, 256)    norm=  22.223
blocks.0.mlp.fc.bias             (1024,)        norm=   2.994
blocks.0.mlp.proj.weight         (256, 1024)    norm=  18.848
blocks.0.mlp.proj.bias           (256,)         norm=   0.774
blocks.1.ln1.weight              (256,)         norm=  12.190
blocks.1.ln1.bias                (256,)         norm=   1.488
blocks.1.attn.qkv.weight         (768, 256)     norm=  15.455
blocks.1.attn.qkv.bias           (768,)         norm=   4.951
blocks.1.attn.proj.weight        (256, 256)     norm=   8.315
blocks.1.attn.proj.bias          (256,)         norm=   0.598
blocks.1.ln2.weight              (256,)         norm=  16.140
blocks.1.ln2.bias                (256,)         norm=   3.837
blocks.1.mlp.fc.weight           (1024, 256)    norm=  24.370
blocks.1.mlp.fc.bias             (1024,)        norm=   2.656
blocks.1.mlp.proj.weight         (256, 1024)    norm=  18.443
blocks.1.mlp.proj.bias           (256,)         norm=   0.476
blocks.2.ln1.weight              (256,)         norm=  14.562
blocks.2.ln1.bias                (256,)         norm=   1.078
blocks.2.attn.qkv.weight         (768, 256)     norm=  18.986
blocks.2.attn.qkv.bias           (768,)         norm=   2.372
blocks.2.attn.proj.weight        (256, 256)     norm=   9.328
blocks.2.attn.proj.bias          (256,)         norm=   0.585
blocks.2.ln2.weight              (256,)         norm=  17.725
blocks.2.ln2.bias                (256,)         norm=   3.642
blocks.2.mlp.fc.weight           (1024, 256)    norm=  25.030
blocks.2.mlp.fc.bias             (1024,)        norm=   2.081
blocks.2.mlp.proj.weight         (256, 1024)    norm=  21.675
blocks.2.mlp.proj.bias           (256,)         norm=   0.492
blocks.3.ln1.weight              (256,)         norm=  16.789
blocks.3.ln1.bias                (256,)         norm=   1.009
blocks.3.attn.qkv.weight         (768, 256)     norm=  21.154
blocks.3.attn.qkv.bias           (768,)         norm=   1.814
blocks.3.attn.proj.weight        (256, 256)     norm=  10.485
blocks.3.attn.proj.bias          (256,)         norm=   0.532
blocks.3.ln2.weight              (256,)         norm=  19.250
blocks.3.ln2.bias                (256,)         norm=   3.568
blocks.3.mlp.fc.weight           (1024, 256)    norm=  25.704
blocks.3.mlp.fc.bias             (1024,)        norm=   2.131
blocks.3.mlp.proj.weight         (256, 1024)    norm=  24.755
blocks.3.mlp.proj.bias           (256,)         norm=   0.383
blocks.4.ln1.weight              (256,)         norm=  17.331
blocks.4.ln1.bias                (256,)         norm=   0.906
blocks.4.attn.qkv.weight         (768, 256)     norm=  21.214
blocks.4.attn.qkv.bias           (768,)         norm=   1.753
blocks.4.attn.proj.weight        (256, 256)     norm=  11.666
blocks.4.attn.proj.bias          (256,)         norm=   0.569
blocks.4.ln2.weight              (256,)         norm=  20.245
blocks.4.ln2.bias                (256,)         norm=   3.385
blocks.4.mlp.fc.weight           (1024, 256)    norm=  26.227
blocks.4.mlp.fc.bias             (1024,)        norm=   2.063
blocks.4.mlp.proj.weight         (256, 1024)    norm=  26.953
blocks.4.mlp.proj.bias           (256,)         norm=   0.561
blocks.5.ln1.weight              (256,)         norm=  18.364
blocks.5.ln1.bias                (256,)         norm=   0.958
blocks.5.attn.qkv.weight         (768, 256)     norm=  21.907
blocks.5.attn.qkv.bias           (768,)         norm=   1.549
blocks.5.attn.proj.weight        (256, 256)     norm=  13.707
blocks.5.attn.proj.bias          (256,)         norm=   0.685
blocks.5.ln2.weight              (256,)         norm=  19.427
blocks.5.ln2.bias                (256,)         norm=   2.630
blocks.5.mlp.fc.weight           (1024, 256)    norm=  26.124
blocks.5.mlp.fc.bias             (1024,)        norm=   2.279
blocks.5.mlp.proj.weight         (256, 1024)    norm=  27.099
blocks.5.mlp.proj.bias           (256,)         norm=   0.532
ln_f.weight                      (256,)         norm=  39.537
ln_f.bias                        (256,)         norm=   2.191

per-head Frobenius norms: Q, K, V rows and proj columns; and norm of OV = proj_h @ V_h
L0H0: |Q|= 2.98 |K|= 3.11 |V|= 1.23 |O|= 1.13 |OV|= 0.31 |QK|=  2.86
L0H1: |Q|= 3.07 |K|= 3.30 |V|= 1.26 |O|= 1.15 |OV|= 0.30 |QK|=  3.33
L0H2: |Q|= 4.10 |K|= 4.13 |V|= 1.37 |O|= 1.26 |OV|= 0.36 |QK|=  5.27
L0H3: |Q|= 6.77 |K|= 6.60 |V|= 2.16 |O|= 1.69 |OV|= 0.69 |QK|= 10.36
L0H4: |Q|= 4.73 |K|= 4.78 |V|= 1.38 |O|= 1.24 |OV|= 0.36 |QK|=  6.43
L0H5: |Q|= 6.87 |K|= 6.43 |V|= 1.93 |O|= 1.58 |OV|= 0.58 |QK|=  9.93
L0H6: |Q|= 6.37 |K|= 5.84 |V|= 1.89 |O|= 1.66 |OV|= 0.60 |QK|=  8.97
L0H7: |Q|= 6.07 |K|= 5.72 |V|= 1.89 |O|= 1.70 |OV|= 0.62 |QK|=  7.95
L1H0: |Q|= 3.03 |K|= 3.18 |V|= 3.07 |O|= 2.98 |OV|= 2.11 |QK|=  2.30
L1H1: |Q|= 3.74 |K|= 3.82 |V|= 3.54 |O|= 3.47 |OV|= 2.73 |QK|=  3.55
L1H2: |Q|= 2.07 |K|= 2.43 |V|= 2.54 |O|= 2.37 |OV|= 1.51 |QK|=  1.50
L1H3: |Q|= 3.52 |K|= 3.56 |V|= 3.27 |O|= 3.19 |OV|= 2.57 |QK|=  3.01
L1H4: |Q|= 2.05 |K|= 2.41 |V|= 2.15 |O|= 2.00 |OV|= 1.22 |QK|=  1.42
L1H5: |Q|= 2.14 |K|= 2.72 |V|= 2.48 |O|= 2.35 |OV|= 1.39 |QK|=  1.57
L1H6: |Q|= 3.59 |K|= 3.65 |V|= 3.13 |O|= 3.14 |OV|= 2.28 |QK|=  3.44
L1H7: |Q|= 4.22 |K|= 4.16 |V|= 3.57 |O|= 3.62 |OV|= 2.52 |QK|=  3.88
L2H0: |Q|= 4.93 |K|= 4.50 |V|= 3.23 |O|= 3.32 |OV|= 2.01 |QK|=  5.29
L2H1: |Q|= 4.09 |K|= 4.01 |V|= 3.57 |O|= 3.48 |OV|= 2.49 |QK|=  3.58
L2H2: |Q|= 4.52 |K|= 4.29 |V|= 3.09 |O|= 3.18 |OV|= 1.87 |QK|=  4.75
L2H3: |Q|= 3.64 |K|= 3.65 |V|= 3.26 |O|= 3.18 |OV|= 2.16 |QK|=  3.28
L2H4: |Q|= 4.53 |K|= 4.28 |V|= 3.24 |O|= 3.26 |OV|= 2.08 |QK|=  5.00
L2H5: |Q|= 3.64 |K|= 3.61 |V|= 3.43 |O|= 3.36 |OV|= 2.39 |QK|=  3.08
L2H6: |Q|= 4.04 |K|= 3.97 |V|= 3.20 |O|= 3.18 |OV|= 2.16 |QK|=  3.89
L2H7: |Q|= 4.09 |K|= 3.98 |V|= 3.46 |O|= 3.41 |OV|= 2.43 |QK|=  3.83
L3H0: |Q|= 3.98 |K|= 3.90 |V|= 4.19 |O|= 4.22 |OV|= 3.38 |QK|=  3.48
L3H1: |Q|= 5.63 |K|= 5.07 |V|= 3.49 |O|= 3.46 |OV|= 2.28 |QK|=  5.96
L3H2: |Q|= 5.07 |K|= 4.75 |V|= 3.44 |O|= 3.47 |OV|= 2.35 |QK|=  4.86
L3H3: |Q|= 3.92 |K|= 3.93 |V|= 3.75 |O|= 3.67 |OV|= 2.73 |QK|=  3.44
L3H4: |Q|= 4.15 |K|= 4.05 |V|= 3.66 |O|= 3.68 |OV|= 2.56 |QK|=  3.71
L3H5: |Q|= 4.69 |K|= 4.45 |V|= 3.77 |O|= 3.86 |OV|= 2.74 |QK|=  4.19
L3H6: |Q|= 5.45 |K|= 5.00 |V|= 3.47 |O|= 3.42 |OV|= 2.27 |QK|=  5.52
L3H7: |Q|= 4.55 |K|= 4.39 |V|= 3.81 |O|= 3.80 |OV|= 2.73 |QK|=  3.96
L4H0: |Q|= 5.40 |K|= 5.04 |V|= 4.03 |O|= 4.04 |OV|= 3.03 |QK|=  5.33
L4H1: |Q|= 4.69 |K|= 4.53 |V|= 4.01 |O|= 3.99 |OV|= 3.11 |QK|=  4.24
L4H2: |Q|= 4.14 |K|= 4.14 |V|= 3.53 |O|= 3.49 |OV|= 2.45 |QK|=  4.22
L4H3: |Q|= 4.84 |K|= 4.63 |V|= 3.71 |O|= 3.68 |OV|= 2.68 |QK|=  4.99
L4H4: |Q|= 4.05 |K|= 3.98 |V|= 4.37 |O|= 4.44 |OV|= 3.72 |QK|=  3.50
L4H5: |Q|= 4.51 |K|= 4.34 |V|= 4.26 |O|= 4.36 |OV|= 3.69 |QK|=  3.80
L4H6: |Q|= 4.46 |K|= 4.36 |V|= 4.70 |O|= 4.84 |OV|= 4.27 |QK|=  3.89
L4H7: |Q|= 3.87 |K|= 3.89 |V|= 3.95 |O|= 4.00 |OV|= 3.13 |QK|=  3.31
L5H0: |Q|= 4.55 |K|= 4.39 |V|= 4.09 |O|= 4.10 |OV|= 3.20 |QK|=  4.18
L5H1: |Q|= 4.46 |K|= 4.36 |V|= 5.07 |O|= 5.22 |OV|= 5.38 |QK|=  3.78
L5H2: |Q|= 3.73 |K|= 3.71 |V|= 3.99 |O|= 4.06 |OV|= 3.27 |QK|=  3.02
L5H3: |Q|= 4.45 |K|= 4.33 |V|= 4.06 |O|= 4.06 |OV|= 3.33 |QK|=  3.95
L5H4: |Q|= 4.22 |K|= 4.11 |V|= 4.52 |O|= 4.55 |OV|= 3.92 |QK|=  3.52
L5H5: |Q|= 4.51 |K|= 4.38 |V|= 3.96 |O|= 3.92 |OV|= 2.97 |QK|=  4.36
L5H6: |Q|= 4.65 |K|= 4.49 |V|= 5.43 |O|= 5.69 |OV|= 5.83 |QK|=  3.92
L5H7: |Q|= 4.48 |K|= 4.38 |V|= 6.27 |O|= 6.51 |OV|= 7.58 |QK|=  3.76
ln_f gain mean/std tensor(2.465) tensor(0.172)
wte row norm mean tensor(0.952) wpe row norm mean tensor(0.337)
```

## analysis/02_out.txt

```
head   diag_mean  off_mean  off_std  z=(diag-off)/off_std  frac_diag_top1  frac_diag_top10  median_rank  trace(OV-in-E-basis)
L0H0     0.002     0.003    0.018     -0.03    0.001   0.004     1967
L0H1     0.017     0.017    0.023      0.01    0.000   0.004     2050
L0H2     0.008    -0.002    0.024      0.41    0.001   0.012     1371
L0H3     0.077     0.019    0.054      1.07    0.012   0.072      382
L0H4     0.018     0.008    0.028      0.37    0.000   0.008     1306
L0H5     0.019    -0.021    0.055      0.73    0.003   0.027      529
L0H6     0.049    -0.000    0.046      1.08    0.013   0.082      312
L0H7     0.063     0.006    0.042      1.37    0.021   0.103      256
L1H0    -0.164    -0.025    0.333     -0.42    0.000   0.001     2839
L1H1    -0.039     0.035    0.341     -0.22    0.000   0.001     2689
L1H2    -0.276    -0.195    0.226     -0.36    0.000   0.000     2709
L1H3    -0.085     0.021    0.390     -0.27    0.000   0.001     2525
L1H4    -0.198    -0.163    0.177     -0.20    0.000   0.000     2415
L1H5    -0.050     0.019    0.211     -0.33    0.000   0.000     2694
L1H6    -0.012    -0.029    0.264      0.06    0.000   0.004     2037
L1H7     0.270     0.230    0.331      0.12    0.000   0.004     1840
L2H0    -0.112    -0.091    0.303     -0.07    0.001   0.008     1869
L2H1    -0.207    -0.040    0.369     -0.45    0.000   0.001     2774
L2H2     0.080     0.141    0.250     -0.24    0.000   0.001     2430
L2H3    -0.089     0.035    0.307     -0.40    0.000   0.000     2733
L2H4    -0.113     0.027    0.330     -0.43    0.000   0.000     2889
L2H5    -0.114     0.010    0.361     -0.34    0.000   0.001     2745
L2H6    -0.057     0.128    0.362     -0.51    0.000   0.002     2794
L2H7     0.028     0.170    0.382     -0.37    0.000   0.002     2630
L3H0     0.019    -0.159    0.528      0.34    0.002   0.016     1629
L3H1    -0.110     0.028    0.408     -0.34    0.000   0.001     2790
L3H2     0.090     0.091    0.387     -0.00    0.001   0.008     1815
L3H3    -0.283     0.042    0.467     -0.70    0.000   0.000     3213
L3H4     0.045     0.036    0.364      0.03    0.000   0.007     1940
L3H5    -0.044    -0.036    0.367     -0.02    0.000   0.003     2145
L3H6    -0.562    -0.346    0.474     -0.45    0.001   0.003     2725
L3H7    -0.441    -0.151    0.484     -0.60    0.000   0.000     3116
L4H0     0.202     0.098    0.450      0.23    0.002   0.011     1543
L4H1     0.020     0.210    0.520     -0.37    0.000   0.002     2830
L4H2    -0.072     0.022    0.397     -0.24    0.000   0.002     2387
L4H3     0.172     0.226    0.379     -0.14    0.001   0.006     2435
L4H4    -0.333    -0.258    0.559     -0.14    0.001   0.002     2437
L4H5    -0.334    -0.203    0.690     -0.19    0.000   0.002     2545
L4H6     0.736     0.206    0.610      0.87    0.011   0.049      879
L4H7    -0.123    -0.141    0.447      0.04    0.000   0.003     1938
L5H0     0.406     0.056    0.501      0.70    0.004   0.024      899
L5H1     0.082    -0.129    0.806      0.26    0.001   0.007     1687
L5H2    -1.028    -0.656    0.695     -0.54    0.000   0.002     2795
L5H3     0.058    -0.007    0.442      0.15    0.002   0.010     1929
L5H4    -0.540    -0.453    0.705     -0.12    0.001   0.005     1810
L5H5    -0.097    -0.140    0.465      0.09    0.001   0.015     1914
L5H6     0.680    -0.002    0.942      0.72    0.002   0.022      963
L5H7     0.727     0.159    1.163      0.49    0.000   0.003     1109

sorted by z:
L0H7   1.37
L0H6   1.08
L0H3   1.07
L4H6   0.87
L0H5   0.73
L5H6   0.72
L5H0   0.70
L5H7   0.49
L0H2   0.41
L0H4   0.37
L3H0   0.34
L5H1   0.26
L4H0   0.23
L5H3   0.15
L1H7   0.12
L5H5   0.09
L1H6   0.06
L4H7   0.04
L3H4   0.03
L0H1   0.01
L3H2  -0.00
L3H5  -0.02
L0H0  -0.03
L2H0  -0.07
L5H4  -0.12
L4H4  -0.14
L4H3  -0.14
L4H5  -0.19
L1H4  -0.20
L1H1  -0.22
L4H2  -0.24
L2H2  -0.24
L1H3  -0.27
L1H5  -0.33
L3H1  -0.34
L2H5  -0.34
L1H2  -0.36
L4H1  -0.37
L2H7  -0.37
L2H3  -0.40
L1H0  -0.42
L2H4  -0.43
L3H6  -0.45
L2H1  -0.45
L2H6  -0.51
L5H2  -0.54
L3H7  -0.60
L3H3  -0.70
```

## analysis/03_out.txt

```
<class 'list'> 310
prompt lengths 4 12 9.016129032258064
'One day, a little boy named Tim went'
'One day, a boy named Tim went to'
'Once there was a little girl called Pip.'
'John was a popular boy. He was always'
'Once upon a time, there was a little'
stories 5000
story token lengths: mean 198.21666666666667 median 178.0 share>=256 0.11
'u don\'t have to be scared of the loud dog, I\'ll protect you". The mole felt so safe with the little girl. She was very kind and the mole soon came to trust her. He leaned against her and she kept him safe. The mole had found his best friend.'
first tokens of story 0 [85, 860, 489, 499, 266, 322, 606, 370, 263, 912] prompt0 [383, 327, 12, 259, 391, 497, 402, 341, 427]
real stories first 200 tok: loop 0.21 rep4 0.018043443441417215
```

## analysis/07_out.txt

```
gen {'R4': 0.029, 'R2': 0.154, 'R1': 0.341, 'N ': 0.476} mean token loss by class {'R4': 0.402, 'R2': 0.628, 'R1': 0.726, 'N ': 1.298}
real {'R4': 0.014, 'R2': 0.121, 'R1': 0.336, 'N ': 0.529} mean token loss by class {'R4': 1.322, 'R2': 0.911, 'R1': 1.196, 'N ': 2.23}

(a) mean DLA to the true next token (logit above vocab mean), by class; gen and real rows pooled, and R4 split
comp      R4     R2     R1     N    | R4-N  | R4 gen  R4 real
L0H0    -0.00   0.00  -0.00   0.00 |  -0.00 |  -0.00  -0.00
L0H1    -0.00  -0.00   0.00   0.00 |  -0.00 |  -0.00  -0.00
L0H2    -0.00  -0.00  -0.00  -0.00 |   0.00 |  -0.00  -0.00
L0H3     0.00  -0.00  -0.00   0.00 |  -0.00 |   0.00   0.00
L0H4     0.00  -0.00   0.00   0.00 |   0.00 |   0.00  -0.00
L0H5     0.00   0.00   0.00  -0.00 |   0.00 |   0.00   0.00
L0H6     0.00  -0.00  -0.00   0.00 |  -0.00 |   0.00  -0.00
L0H7     0.00   0.00  -0.00  -0.00 |   0.00 |   0.00   0.00
MLP0     2.78   2.73   2.35   2.07 |   0.71 |   3.04   2.22
L1H0     0.12   0.14   0.20   0.11 |   0.01 |   0.13   0.12
L1H1     0.07   0.06   0.07   0.08 |  -0.02 |   0.07   0.07
L1H2     0.01   0.02   0.02   0.00 |   0.01 |   0.01   0.02
L1H3     0.03   0.02   0.05   0.06 |  -0.03 |   0.03   0.02
L1H4    -0.05  -0.05  -0.05  -0.03 |  -0.02 |  -0.05  -0.05
L1H5     0.10   0.09   0.12   0.04 |   0.06 |   0.10   0.10
L1H6     0.02   0.04   0.05   0.07 |  -0.05 |   0.02   0.02
L1H7     0.08   0.04   0.05   0.08 |   0.00 |   0.10   0.03
MLP1     0.78   0.79   0.81   0.71 |   0.07 |   0.81   0.73
L2H0     0.10   0.07   0.06   0.09 |   0.01 |   0.10   0.09
L2H1     0.05   0.05   0.05   0.08 |  -0.03 |   0.05   0.05
L2H2     0.05   0.05   0.05   0.07 |  -0.02 |   0.05   0.06
L2H3     0.05   0.05   0.05   0.07 |  -0.02 |   0.05   0.05
L2H4     0.08   0.06   0.07   0.07 |   0.01 |   0.08   0.08
L2H5     0.04   0.06   0.06   0.07 |  -0.03 |   0.03   0.05
L2H6     0.09   0.05   0.09   0.08 |   0.01 |   0.09   0.07
L2H7     0.08   0.04   0.05   0.05 |   0.02 |   0.08   0.07
MLP2     0.83   0.92   0.92   0.95 |  -0.12 |   0.89   0.71
L3H0     0.26   0.29   0.17   0.10 |   0.17 |   0.29   0.21
L3H1     0.15   0.12   0.12   0.14 |   0.01 |   0.14   0.16
L3H2     0.09   0.04   0.04   0.07 |   0.02 |   0.10   0.07
L3H3     0.01   0.03   0.05   0.08 |  -0.07 |   0.01   0.02
L3H4     0.08   0.17   0.11   0.11 |  -0.03 |   0.08   0.10
L3H5     0.05   0.05   0.06   0.13 |  -0.07 |   0.06   0.04
L3H6     0.12   0.09   0.11   0.11 |   0.01 |   0.12   0.11
L3H7     0.01   0.02   0.03   0.09 |  -0.08 |   0.02   0.01
MLP3     0.91   1.08   1.25   1.37 |  -0.45 |   0.99   0.75
L4H0     0.18   0.16   0.18   0.18 |   0.00 |   0.19   0.17
L4H1     0.05   0.02   0.06   0.08 |  -0.03 |   0.06   0.03
L4H2     0.07   0.07   0.09   0.10 |  -0.03 |   0.06   0.09
L4H3     0.03   0.08   0.06   0.10 |  -0.07 |   0.02   0.04
L4H4     0.14   0.17   0.05   0.08 |   0.06 |   0.16   0.10
L4H5     0.44   0.34   0.15   0.06 |   0.38 |   0.46   0.41
L4H6     0.33   0.28   0.15   0.16 |   0.17 |   0.33   0.33
L4H7     0.01   0.02   0.04   0.12 |  -0.11 |   0.02  -0.00
MLP4     1.36   1.67   2.05   1.97 |  -0.60 |   1.36   1.37
L5H0     0.10   0.08   0.16   0.17 |  -0.08 |   0.11   0.07
L5H1     0.49   0.48   0.26   0.09 |   0.39 |   0.51   0.44
L5H2    -0.01   0.06  -0.04   0.01 |  -0.02 |   0.00  -0.04
L5H3     0.07   0.14   0.11   0.15 |  -0.08 |   0.07   0.07
L5H4     0.20   0.13   0.05   0.05 |   0.15 |   0.23   0.14
L5H5     0.03   0.04   0.08   0.08 |  -0.05 |   0.04   0.00
L5H6     0.85   0.61   0.30   0.10 |   0.74 |   0.91   0.71
L5H7     0.81   0.53   0.23   0.09 |   0.73 |   0.84   0.76
MLP5     1.43   1.23   1.45   1.47 |  -0.05 |   1.31   1.66

sorted by R4-N:
L5H6   R4=  0.85 N=  0.10 diff=  0.74
L5H7   R4=  0.81 N=  0.09 diff=  0.73
MLP0   R4=  2.78 N=  2.07 diff=  0.71
L5H1   R4=  0.49 N=  0.09 diff=  0.39
L4H5   R4=  0.44 N=  0.06 diff=  0.38
L4H6   R4=  0.33 N=  0.16 diff=  0.17
L3H0   R4=  0.26 N=  0.10 diff=  0.17
L5H4   R4=  0.20 N=  0.05 diff=  0.15
MLP1   R4=  0.78 N=  0.71 diff=  0.07
L4H4   R4=  0.14 N=  0.08 diff=  0.06
L1H5   R4=  0.10 N=  0.04 diff=  0.06
L2H7   R4=  0.08 N=  0.05 diff=  0.02
L3H2   R4=  0.09 N=  0.07 diff=  0.02
L1H2   R4=  0.01 N=  0.00 diff=  0.01
sum of all heads+MLPs:  {'R4': 13.57, 'R2': 13.26, 'R1': 12.45, 'N ': 11.98}

(b) attention mass, averaged over positions t>=8 where the target set is non-empty
head   induction(j: tok[j-1]==tok[t])   to-true-next-token(j: tok[j]==tok[t+1]) at R4 | at R1 ;  uniform baseline in brackets
L0H0  ind=0.039 [0.035]   next@R4=0.035  next@R1=0.041
L0H1  ind=0.035 [0.035]   next@R4=0.035  next@R1=0.036
L0H2  ind=0.048 [0.035]   next@R4=0.030  next@R1=0.035
L0H3  ind=0.037 [0.035]   next@R4=0.033  next@R1=0.047
L0H4  ind=0.039 [0.035]   next@R4=0.027  next@R1=0.035
L0H5  ind=0.069 [0.035]   next@R4=0.124  next@R1=0.048
L0H6  ind=0.061 [0.035]   next@R4=0.037  next@R1=0.040
L0H7  ind=0.031 [0.035]   next@R4=0.025  next@R1=0.025
L1H0  ind=0.033 [0.035]   next@R4=0.030  next@R1=0.033
L1H1  ind=0.032 [0.035]   next@R4=0.026  next@R1=0.033
L1H2  ind=0.038 [0.035]   next@R4=0.030  next@R1=0.045
L1H3  ind=0.035 [0.035]   next@R4=0.036  next@R1=0.041
L1H4  ind=0.032 [0.035]   next@R4=0.033  next@R1=0.037
L1H5  ind=0.033 [0.035]   next@R4=0.035  next@R1=0.036
L1H6  ind=0.053 [0.035]   next@R4=0.026  next@R1=0.028
L1H7  ind=0.031 [0.035]   next@R4=0.014  next@R1=0.020
L2H0  ind=0.026 [0.035]   next@R4=0.011  next@R1=0.013
L2H1  ind=0.031 [0.035]   next@R4=0.018  next@R1=0.025
L2H2  ind=0.026 [0.035]   next@R4=0.016  next@R1=0.022
L2H3  ind=0.038 [0.035]   next@R4=0.019  next@R1=0.031
L2H4  ind=0.028 [0.035]   next@R4=0.020  next@R1=0.027
L2H5  ind=0.066 [0.035]   next@R4=0.022  next@R1=0.028
L2H6  ind=0.043 [0.035]   next@R4=0.023  next@R1=0.035
L2H7  ind=0.047 [0.035]   next@R4=0.023  next@R1=0.033
L3H0  ind=0.109 [0.035]   next@R4=0.176  next@R1=0.091
L3H1  ind=0.037 [0.035]   next@R4=0.021  next@R1=0.021
L3H2  ind=0.035 [0.035]   next@R4=0.018  next@R1=0.027
L3H3  ind=0.064 [0.035]   next@R4=0.031  next@R1=0.033
L3H4  ind=0.090 [0.035]   next@R4=0.086  next@R1=0.069
L3H5  ind=0.031 [0.035]   next@R4=0.020  next@R1=0.023
L3H6  ind=0.039 [0.035]   next@R4=0.020  next@R1=0.023
L3H7  ind=0.041 [0.035]   next@R4=0.039  next@R1=0.035
L4H0  ind=0.034 [0.035]   next@R4=0.021  next@R1=0.024
L4H1  ind=0.051 [0.035]   next@R4=0.070  next@R1=0.037
L4H2  ind=0.047 [0.035]   next@R4=0.030  next@R1=0.030
L4H3  ind=0.082 [0.035]   next@R4=0.037  next@R1=0.021
L4H4  ind=0.133 [0.035]   next@R4=0.232  next@R1=0.093
L4H5  ind=0.146 [0.035]   next@R4=0.252  next@R1=0.096
L4H6  ind=0.077 [0.035]   next@R4=0.177  next@R1=0.076
L4H7  ind=0.036 [0.035]   next@R4=0.032  next@R1=0.042
L5H0  ind=0.047 [0.035]   next@R4=0.053  next@R1=0.024
L5H1  ind=0.119 [0.035]   next@R4=0.247  next@R1=0.112
L5H2  ind=0.070 [0.035]   next@R4=0.060  next@R1=0.034
L5H3  ind=0.069 [0.035]   next@R4=0.054  next@R1=0.059
L5H4  ind=0.068 [0.035]   next@R4=0.157  next@R1=0.069
L5H5  ind=0.063 [0.035]   next@R4=0.050  next@R1=0.044
L5H6  ind=0.126 [0.035]   next@R4=0.324  next@R1=0.129
L5H7  ind=0.144 [0.035]   next@R4=0.348  next@R1=0.123
top induction heads: [('L4H5', 0.146), ('L5H7', 0.144), ('L4H4', 0.133), ('L5H6', 0.126), ('L5H1', 0.119), ('L3H0', 0.109), ('L3H4', 0.09), ('L4H3', 0.082)]
top attend-to-next-token@R4 heads: [('L5H7', 0.348), ('L5H6', 0.324), ('L4H5', 0.252), ('L5H1', 0.247), ('L4H4', 0.232), ('L4H6', 0.177), ('L3H0', 0.176), ('L5H4', 0.157)]

(c) calibration: predicted probability mass on repeat-completing tokens vs how often the true next token is one
T=1          real 4gram: pred 0.0105 actual 0.0140 | bigram: pred 0.1282 actual 0.1346 | seen-token: pred 0.4625 actual 0.4709
T=1          gen  4gram: pred 0.0276 actual 0.0295 | bigram: pred 0.1654 actual 0.1834 | seen-token: pred 0.4828 actual 0.5244
T=0.8,top40  real 4gram: pred 0.0116 actual 0.0140 | bigram: pred 0.1429 actual 0.1346 | seen-token: pred 0.4994 actual 0.4709
T=0.8,top40  gen  4gram: pred 0.0301 actual 0.0295 | bigram: pred 0.1801 actual 0.1834 | seen-token: pred 0.5129 actual 0.5244
```

## analysis/08_out.txt

```
(1) OV copy score with extended embedding: z = (diag - offdiag mean)/std ; share of tokens whose own logit is top-1 / top-10
  L4H6 z= 3.20 top1=0.406 top10=0.677 diag_mean=3.80
  L5H7 z= 2.99 top1=0.559 top10=0.758 diag_mean=10.85
  L5H1 z= 2.66 top1=0.207 top10=0.435 diag_mean=4.45
  L5H6 z= 2.46 top1=0.421 top10=0.619 diag_mean=6.04
  L4H4 z= 2.43 top1=0.265 top10=0.490 diag_mean=2.20
  L3H0 z= 2.34 top1=0.168 top10=0.418 diag_mean=2.07
  L3H4 z= 1.97 top1=0.079 top10=0.280 diag_mean=0.80
  L5H4 z= 1.49 top1=0.038 top10=0.140 diag_mean=1.29
  L3H5 z= 1.38 top1=0.020 top10=0.124 diag_mean=0.56
  L4H5 z= 1.33 top1=0.061 top10=0.196 diag_mean=1.22
  L4H7 z= 0.99 top1=0.010 top10=0.057 diag_mean=0.53
  L1H7 z= 0.82 top1=0.016 top10=0.065 diag_mean=0.32
  ... lowest: [('L5H5', -1.09), ('L4H3', -0.59), ('L5H3', -0.28), ('L3H6', -0.25), ('L5H2', -0.24)]

(2) layer-0 positional attention from wpe alone (query position 100): softmax over j<=100, mass at offsets 0,-1,-2,-3, and the rest
  L0H0: off0=0.012 off-1=0.007 off-2=0.006 off-3=0.007 rest=0.968
  L0H1: off0=0.103 off-1=0.048 off-2=0.054 off-3=0.060 rest=0.735
  L0H2: off0=0.037 off-1=0.036 off-2=0.042 off-3=0.032 rest=0.854
  L0H3: off0=0.142 off-1=0.100 off-2=0.110 off-3=0.075 rest=0.574
  L0H4: off0=0.036 off-1=0.033 off-2=0.026 off-3=0.031 rest=0.874
  L0H5: off0=0.054 off-1=0.037 off-2=0.044 off-3=0.041 rest=0.823
  L0H6: off0=0.137 off-1=0.101 off-2=0.154 off-3=0.071 rest=0.536
  L0H7: off0=0.127 off-1=0.102 off-2=0.180 off-3=0.062 rest=0.529

(3) induction QK match. key side = L0 head OV applied to ln1_0(E[a]) (what an L0 head writes at a position when it attends to token a),
    then that layer's ln1; query side = ln1(Eext[a]). z of the diagonal of q K^T, per (late head, L0 writer head); 1024 random tokens
    head    L0H0  L0H1  L0H2  L0H3  L0H4  L0H5  L0H6  L0H7   | token-match (key = Eext[a] itself)
    L3H0    0.04   0.02   0.18   0.19   0.19   0.12   0.24   0.22   |  0.08
    L3H1    0.02  -0.06  -0.16   0.03  -0.25   0.22   0.17  -0.26   | -1.05
    L3H2    0.02  -0.05  -0.11   0.02  -0.16   0.04   0.13   0.02   | -1.28
    L3H3    0.04  -0.00   0.08   0.09  -0.21   0.00   0.18   0.03   | -0.68
    L3H4    0.03  -0.07  -0.08  -0.16   0.06   0.13   0.17   0.20   | -0.74
    L3H5    0.00  -0.08  -0.20  -0.14  -0.22  -0.05   0.10  -0.14   | -1.18
    L3H6    0.01  -0.02  -0.17   0.00  -0.22   0.11   0.04  -0.09   | -1.27
    L3H7   -0.01  -0.06  -0.11  -0.09  -0.24   0.12   0.19  -0.17   | -1.39
    L4H0   -0.01  -0.04  -0.09  -0.01  -0.11   0.01  -0.10  -0.12   | -1.09
    L4H1    0.03  -0.01   0.01  -0.04  -0.30  -0.02   0.26   0.13   | -1.42
    L4H2    0.00   0.02  -0.02   0.01  -0.00  -0.18   0.15   0.28   | -0.50
    L4H3   -0.00  -0.04  -0.19   0.10  -0.26   0.14   0.14   0.07   | -1.08
    L4H4    0.03  -0.03  -0.09   0.04   0.21   0.32   0.18   0.13   | -0.20
    L4H5    0.02   0.16   0.03   0.34   0.09   0.25   0.54   0.55   | -1.06
    L4H6   -0.02  -0.01  -0.11   0.20   0.04   0.20   0.27   0.37   | -0.50
    L4H7    0.01   0.02   0.04  -0.03  -0.10  -0.16   0.27   0.30   | -0.62
    L5H0    0.03   0.00   0.19  -0.15  -0.07  -0.11   0.13   0.07   | -0.62
    L5H1    0.03   0.03   0.13   0.24   0.18   0.16   0.22   0.19   | -0.34
    L5H2    0.04  -0.01   0.03   0.08   0.03  -0.08   0.23   0.25   |  0.17
    L5H3    0.03   0.00   0.01   0.11   0.10  -0.00   0.26   0.09   | -0.30
    L5H4    0.05   0.04   0.04   0.21   0.03   0.23   0.31   0.40   | -0.12
    L5H5    0.01   0.02  -0.16   0.08  -0.06   0.14   0.02   0.09   | -0.14
    L5H6    0.03   0.07   0.01   0.26   0.03   0.19   0.48   0.41   | -0.28
    L5H7    0.03   0.05   0.15   0.36   0.14   0.09   0.41   0.38   | -0.46
```

## analysis/09_out.txt

```
test_ablate_core4 tensors touched 2 weights changed 32768 relative L2 0.07954
test_ablate_control4 tensors touched 2 weights changed 32768 relative L2 0.05864
pass3 base loss {"call": 3, "weights_sha256": "6d9615059a7d499ad645f2f310fef70f91388a4804714becb008c72d90cb86ec", "mean_nll": 1.6499893557179008}
pass4 core4 ablated gen {"call": 4, "weights_sha256": "6f9826b3c53b2eed02a444c499ea50d517d2c5381fd7d7dd843b9abdeaa8f3c9", "generations_file": "work/outputs/generations_call004.json", "loop_rate": 0.109375, "rep4": 0.012382359795936358, "mean_length": 155.875, "eot_rate": 0.734375}
pass5 core4 ablated loss {"call": 5, "weights_sha256": "6f9826b3c53b2eed02a444c499ea50d517d2c5381fd7d7dd843b9abdeaa8f3c9", "mean_nll": 1.7958897593200214}
  delta vs base 0.14590040360212053
pass6 control4 ablated gen {"call": 6, "weights_sha256": "04c0d474115e31bba49ba74520dba115ab9ce99ca2c43c802329b572ba8ec49f", "generations_file": "work/outputs/generations_call006.json", "loop_rate": 0.234375, "rep4": 0.023716997395177228, "mean_length": 162.578125, "eot_rate": 0.671875}
pass7 control4 ablated loss {"call": 7, "weights_sha256": "04c0d474115e31bba49ba74520dba115ab9ce99ca2c43c802329b572ba8ec49f", "mean_nll": 1.6838265530049288}
  delta vs base 0.03383719728702794
used 7 remaining 43
```

## analysis/09_compare.txt

```
rows 64
base       loop=0.2344  long=0.0000  rep4=0.0255  rep2=0.1793  len=162.6250  eot=0.7188
core_ablated loop=0.1094  long=0.0000  rep4=0.0124  rep2=0.1218  len=155.8750  eot=0.7344
   loop  diff -0.1250  95% CI [-0.2656, +0.0156]   relative -53.3%
   rep4  diff -0.0131  95% CI [-0.0207, -0.0061]   relative -51.5%
   rep2  diff -0.0575  95% CI [-0.0742, -0.0408]   relative -32.1%
   len   diff -6.7500  95% CI [-18.5469, +4.3281]   relative -4.2%
control_ablated loop=0.2344  long=0.0000  rep4=0.0237  rep2=0.1789  len=162.5781  eot=0.6719
   loop  diff +0.0000  95% CI [-0.1250, +0.1250]   relative +0.0%
   rep4  diff -0.0018  95% CI [-0.0085, +0.0048]   relative -7.1%
   rep2  diff -0.0004  95% CI [-0.0176, +0.0167]   relative -0.2%
   len   diff -0.0469  95% CI [-10.0156, +9.9062]   relative -0.0%
```

## analysis/10_base_out.txt

```
base g1 {"call": 8, "weights_sha256": "6d9615059a7d499ad645f2f310fef70f91388a4804714becb008c72d90cb86ec", "generations_file": "work/outputs/generations_call008.json", "loop_rate": 0.296875, "rep4": 0.030249970851446296, "mean_length": 167.984375, "eot_rate": 0.671875}
base g2 {"call": 9, "weights_sha256": "6d9615059a7d499ad645f2f310fef70f91388a4804714becb008c72d90cb86ec", "generations_file": "work/outputs/generations_call009.json", "loop_rate": 0.234375, "rep4": 0.024416481440633984, "mean_length": 159.375, "eot_rate": 0.765625}
base g3 {"call": 10, "weights_sha256": "6d9615059a7d499ad645f2f310fef70f91388a4804714becb008c72d90cb86ec", "generations_file": "work/outputs/generations_call010.json", "loop_rate": 0.234375, "rep4": 0.027482076052802222, "mean_length": 159.046875, "eot_rate": 0.75}
base g4 {"call": 11, "weights_sha256": "6d9615059a7d499ad645f2f310fef70f91388a4804714becb008c72d90cb86ec", "generations_file": "work/outputs/generations_call011.json", "loop_rate": 0.2222222222222222, "rep4": 0.030132062736876217, "mean_length": 153.11111111111111, "eot_rate": 0.8148148148148148}
used 11 remaining 39
```

## analysis/11_v1_out.txt

```
v1 l0 {"call": 12, "weights_sha256": "39b9e4521cfc3df2f8ebfb5e71da1400eae96a734479472eed49f9c3a4839004", "mean_nll": 1.6538304252086118}
v1 g0 {"call": 13, "weights_sha256": "39b9e4521cfc3df2f8ebfb5e71da1400eae96a734479472eed49f9c3a4839004", "generations_file": "work/outputs/generations_call013.json", "loop_rate": 0.21875, "rep4": 0.02268959778358705, "mean_length": 160.34375, "eot_rate": 0.75}
v1 g1 {"call": 14, "weights_sha256": "39b9e4521cfc3df2f8ebfb5e71da1400eae96a734479472eed49f9c3a4839004", "generations_file": "work/outputs/generations_call014.json", "loop_rate": 0.25, "rep4": 0.027487807434383282, "mean_length": 171.859375, "eot_rate": 0.65625}
v1 g2 {"call": 15, "weights_sha256": "39b9e4521cfc3df2f8ebfb5e71da1400eae96a734479472eed49f9c3a4839004", "generations_file": "work/outputs/generations_call015.json", "loop_rate": 0.171875, "rep4": 0.019458796383589532, "mean_length": 163.59375, "eot_rate": 0.6875}
v1 g3 {"call": 16, "weights_sha256": "39b9e4521cfc3df2f8ebfb5e71da1400eae96a734479472eed49f9c3a4839004", "generations_file": "work/outputs/generations_call016.json", "loop_rate": 0.21875, "rep4": 0.023448848677234357, "mean_length": 155.109375, "eot_rate": 0.703125}
v1 g4 {"call": 17, "weights_sha256": "39b9e4521cfc3df2f8ebfb5e71da1400eae96a734479472eed49f9c3a4839004", "generations_file": "work/outputs/generations_call017.json", "loop_rate": 0.2962962962962963, "rep4": 0.02683762217099462, "mean_length": 158.22222222222223, "eot_rate": 0.7777777777777778}
used 17 remaining 33
```

## analysis/12_compare_v1.txt

```
rows 310
base       loop=0.2452  long=0.0065  rep4=0.0275  rep2=0.1798  len=160.6645  eot=0.7419
v1         loop=0.2290  long=0.0000  rep4=0.0239  rep2=0.1697  len=161.9419  eot=0.7129
   loop  diff -0.0161  95% CI [-0.0774, +0.0484]   relative -6.6%
   rep4  diff -0.0036  95% CI [-0.0075, +0.0003]   relative -13.1%
   rep2  diff -0.0101  95% CI [-0.0178, -0.0023]   relative -5.6%
   len   diff +1.2774  95% CI [-3.5226, +6.1806]   relative +0.8%
```

## analysis/13_out.txt

```
test_ablate_secondary4 tensors touched 3 weights changed 32768 relative L2 0.06531
secondary ablated g0 {"call": 18, "weights_sha256": "244974f1eec301b8ecf3b77c20dbb4e66f87e5e0f0edf3fdefa6ad8aa9e78b59", "generations_file": "work/outputs/generations_call018.json", "loop_rate": 0.125, "rep4": 0.01898055349425289, "mean_length": 163.703125, "eot_rate": 0.671875}
secondary ablated g1 {"call": 19, "weights_sha256": "244974f1eec301b8ecf3b77c20dbb4e66f87e5e0f0edf3fdefa6ad8aa9e78b59", "generations_file": "work/outputs/generations_call019.json", "loop_rate": 0.203125, "rep4": 0.026511666985218424, "mean_length": 171.078125, "eot_rate": 0.65625}
secondary ablated l0 {"call": 20, "weights_sha256": "244974f1eec301b8ecf3b77c20dbb4e66f87e5e0f0edf3fdefa6ad8aa9e78b59", "mean_nll": 1.7084322568808534}
  delta vs base (1.64999) 0.05844290116295259
used 20 remaining 30
```

## analysis/14_v2_out.txt

```
v2 l0 {"call": 21, "weights_sha256": "0bae6140301531717336d1e42ef29ab3bcc41336016eb0502455e830586153d6", "mean_nll": 1.6569293342919633}
v2 g0 {"call": 22, "weights_sha256": "0bae6140301531717336d1e42ef29ab3bcc41336016eb0502455e830586153d6", "generations_file": "work/outputs/generations_call022.json", "loop_rate": 0.15625, "rep4": 0.020604045565379082, "mean_length": 166.46875, "eot_rate": 0.75}
v2 g1 {"call": 23, "weights_sha256": "0bae6140301531717336d1e42ef29ab3bcc41336016eb0502455e830586153d6", "generations_file": "work/outputs/generations_call023.json", "loop_rate": 0.1875, "rep4": 0.017777807239718293, "mean_length": 157.140625, "eot_rate": 0.703125}
v2 g2 {"call": 24, "weights_sha256": "0bae6140301531717336d1e42ef29ab3bcc41336016eb0502455e830586153d6", "generations_file": "work/outputs/generations_call024.json", "loop_rate": 0.109375, "rep4": 0.020450048494688615, "mean_length": 167.703125, "eot_rate": 0.65625}
v2 g3 {"call": 25, "weights_sha256": "0bae6140301531717336d1e42ef29ab3bcc41336016eb0502455e830586153d6", "generations_file": "work/outputs/generations_call025.json", "loop_rate": 0.234375, "rep4": 0.02583520266397058, "mean_length": 168.4375, "eot_rate": 0.6875}
v2 g4 {"call": 26, "weights_sha256": "0bae6140301531717336d1e42ef29ab3bcc41336016eb0502455e830586153d6", "generations_file": "work/outputs/generations_call026.json", "loop_rate": 0.2037037037037037, "rep4": 0.02503127033682168, "mean_length": 161.74074074074073, "eot_rate": 0.7962962962962963}
used 26 remaining 24
```

## analysis/15_compare_v2.txt

```
rows 310
base       loop=0.2452  long=0.0065  rep4=0.0275  rep2=0.1798  len=160.6645  eot=0.7419
v1         loop=0.2290  long=0.0000  rep4=0.0239  rep2=0.1697  len=161.9419  eot=0.7129
   loop  diff -0.0161  95% CI [-0.0774, +0.0484]   relative -6.6%
   rep4  diff -0.0036  95% CI [-0.0075, +0.0003]   relative -13.1%
   rep2  diff -0.0101  95% CI [-0.0178, -0.0023]   relative -5.6%
   len   diff +1.2774  95% CI [-3.5226, +6.1806]   relative +0.8%
v2         loop=0.1774  long=0.0032  rep4=0.0218  rep2=0.1651  len=164.3806  eot=0.7161
   loop  diff -0.0677  95% CI [-0.1290, -0.0065]   relative -27.6%
   rep4  diff -0.0056  95% CI [-0.0095, -0.0017]   relative -20.5%
   rep2  diff -0.0147  95% CI [-0.0226, -0.0070]   relative -8.2%
   len   diff +3.7161  95% CI [-1.2067, +8.5420]   relative +2.3%
```

## analysis/16_base_seed1_out.txt

```
base l1 {"call": 27, "weights_sha256": "6d9615059a7d499ad645f2f310fef70f91388a4804714becb008c72d90cb86ec", "mean_nll": 1.5126434576300447}
base h0 {"call": 28, "weights_sha256": "6d9615059a7d499ad645f2f310fef70f91388a4804714becb008c72d90cb86ec", "generations_file": "work/outputs/generations_call028.json", "loop_rate": 0.3125, "rep4": 0.02669801046897418, "mean_length": 161.15625, "eot_rate": 0.765625}
base h1 {"call": 29, "weights_sha256": "6d9615059a7d499ad645f2f310fef70f91388a4804714becb008c72d90cb86ec", "generations_file": "work/outputs/generations_call029.json", "loop_rate": 0.1875, "rep4": 0.023531149329853097, "mean_length": 160.15625, "eot_rate": 0.75}
base h2 {"call": 30, "weights_sha256": "6d9615059a7d499ad645f2f310fef70f91388a4804714becb008c72d90cb86ec", "generations_file": "work/outputs/generations_call030.json", "loop_rate": 0.21875, "rep4": 0.029668866402159634, "mean_length": 160.734375, "eot_rate": 0.765625}
base h3 {"call": 31, "weights_sha256": "6d9615059a7d499ad645f2f310fef70f91388a4804714becb008c72d90cb86ec", "generations_file": "work/outputs/generations_call031.json", "loop_rate": 0.3125, "rep4": 0.03203585192205503, "mean_length": 156.28125, "eot_rate": 0.84375}
base h4 {"call": 32, "weights_sha256": "6d9615059a7d499ad645f2f310fef70f91388a4804714becb008c72d90cb86ec", "generations_file": "work/outputs/generations_call032.json", "loop_rate": 0.37037037037037035, "rep4": 0.03350064234543171, "mean_length": 152.22222222222223, "eot_rate": 0.7592592592592593}
used 32 remaining 18
```

## analysis/17_v2_seed1_out.txt

```
v2 l1 {"call": 33, "weights_sha256": "0bae6140301531717336d1e42ef29ab3bcc41336016eb0502455e830586153d6", "mean_nll": 1.520463605490634}
v2 h0 {"call": 34, "weights_sha256": "0bae6140301531717336d1e42ef29ab3bcc41336016eb0502455e830586153d6", "generations_file": "work/outputs/generations_call034.json", "loop_rate": 0.21875, "rep4": 0.019506940812154892, "mean_length": 160.515625, "eot_rate": 0.703125}
v2 h1 {"call": 35, "weights_sha256": "0bae6140301531717336d1e42ef29ab3bcc41336016eb0502455e830586153d6", "generations_file": "work/outputs/generations_call035.json", "loop_rate": 0.109375, "rep4": 0.02095037729949921, "mean_length": 158.4375, "eot_rate": 0.828125}
v2 h2 {"call": 36, "weights_sha256": "0bae6140301531717336d1e42ef29ab3bcc41336016eb0502455e830586153d6", "generations_file": "work/outputs/generations_call036.json", "loop_rate": 0.125, "rep4": 0.019629292227021478, "mean_length": 155.0, "eot_rate": 0.84375}
v2 h3 {"call": 37, "weights_sha256": "0bae6140301531717336d1e42ef29ab3bcc41336016eb0502455e830586153d6", "generations_file": "work/outputs/generations_call037.json", "loop_rate": 0.171875, "rep4": 0.018090124028557697, "mean_length": 166.484375, "eot_rate": 0.703125}
v2 h4 {"call": 38, "weights_sha256": "0bae6140301531717336d1e42ef29ab3bcc41336016eb0502455e830586153d6", "generations_file": "work/outputs/generations_call038.json", "loop_rate": 0.2037037037037037, "rep4": 0.026307432248273165, "mean_length": 155.40740740740742, "eot_rate": 0.7592592592592593}
used 38 remaining 12
```

## analysis/18_compare_seed1.txt

```
rows 310
base       loop=0.2774  long=0.0032  rep4=0.0289  rep2=0.1801  len=158.3000  eot=0.7774
v2         loop=0.1645  long=0.0032  rep4=0.0207  rep2=0.1595  len=159.2903  eot=0.7677
   loop  diff -0.1129  95% CI [-0.1742, -0.0548]   relative -40.7%
   rep4  diff -0.0082  95% CI [-0.0120, -0.0044]   relative -28.4%
   rep2  diff -0.0206  95% CI [-0.0287, -0.0126]   relative -11.4%
   len   diff +0.9903  95% CI [-3.9968, +5.8774]   relative +0.6%
--- both seeds pooled (620 rows)
rows 620
base       loop=0.2613  long=0.0048  rep4=0.0282  rep2=0.1800  len=159.4823  eot=0.7597
v2         loop=0.1710  long=0.0032  rep4=0.0213  rep2=0.1623  len=161.8355  eot=0.7419
   loop  diff -0.0903  95% CI [-0.1339, -0.0452]   relative -34.6%
   rep4  diff -0.0069  95% CI [-0.0097, -0.0042]   relative -24.6%
   rep2  diff -0.0177  95% CI [-0.0232, -0.0121]   relative -9.8%
   len   diff +2.3532  95% CI [-1.2000, +5.8404]   relative +1.5%
```

## analysis/20_out.txt

```
base greedy {"call": 39, "weights_sha256": "6d9615059a7d499ad645f2f310fef70f91388a4804714becb008c72d90cb86ec", "generations_file": "work/outputs/generations_call039.json", "loop_rate": 0.421875, "rep4": 0.11084694171649279, "mean_length": 165.09375, "eot_rate": 0.71875}
v2 greedy {"call": 40, "weights_sha256": "0bae6140301531717336d1e42ef29ab3bcc41336016eb0502455e830586153d6", "generations_file": "work/outputs/generations_call040.json", "loop_rate": 0.28125, "rep4": 0.0434206556275656, "mean_length": 158.34375, "eot_rate": 0.875}
used 40 remaining 10
rows 64
base       loop=0.4219  long=0.1406  rep4=0.1108  rep2=0.2673  len=165.0938  eot=0.7188
v2         loop=0.2812  long=0.0156  rep4=0.0434  rep2=0.2036  len=158.3438  eot=0.8750
   loop  diff -0.1406  95% CI [-0.2812, +0.0000]   relative -33.3%
   rep4  diff -0.0674  95% CI [-0.1074, -0.0338]   relative -60.8%
   rep2  diff -0.0637  95% CI [-0.1008, -0.0304]   relative -23.8%
   len   diff -6.7500  95% CI [-15.0625, +1.6875]   relative -4.1%
```

## analysis/predictions.md

# Predictions written before the passes that test them

## v1 = copy4_x080 (written before any pass on it)
Edit: output of L5H6, L5H7, L5H1, L4H5 multiplied by 0.8 (columns of attn.proj.weight).
Why 0.8: it is the decoding temperature. Pass 2 showed the model's repeat probability is calibrated on
real stories at T=1; decoding at T=0.8 multiplies every logit gap, the copy heads' contribution included,
by 1/0.8. Multiplying those heads' output by 0.8 gives their contribution under T=0.8 decoding the size
it has in the calibrated T=1 model. The number comes from the decoding rule, not from activations.
Predictions: loss on stories 1-64 rises by 0.003-0.012 nats (full ablation cost 0.146; a calibrated
component costs second order for a small change). Loop rate falls by roughly 2-3 points and rep-4 by
roughly 0.002-0.003 on 310 prompts (ablation removed 12.5 points / 0.013). Length unchanged.

## v1 outcome (passes 12-17) against the prediction
Loss +0.0038 (predicted 0.003-0.012). Loop rate -1.6 points (predicted 2-3; the paired CI is +-6 points,
so unresolved). rep-4 -0.0036 (predicted 0.002-0.003). Length +0.8%. Prediction held.

## Secondary copy heads (written before passes 18-20)
L4H6, L4H4, L3H0, L5H4 show the same three signatures as the core four, more weakly (pass 2: R4-N DLA
0.17, 0.06, 0.17, 0.15; attention to the true next token at R4 0.18, 0.23, 0.18, 0.16; weights: copy OV
z 3.2, 2.4, 2.3, 1.5). Hypothesis: they also carry in-context copying. Prediction: zeroing them lowers
rep-4 on prompts 0-127 seed 0 from 0.0279 by 0.003-0.008 (the core four gave about -0.013 on prompts 0-63),
loop rate by a few points, and raises loss on stories 1-64 by 0.03-0.08.
If rep-4 does not fall, the hypothesis is rejected and v1 stays as it is.

## Secondary ablation outcome (passes 18-20)
rep-4 -0.0051 on 128 rows (predicted 0.003-0.008), loop rate -10 points (CI -19.5 to -0.8), rep-2 -0.022
(CI excludes 0), loss +0.058 (predicted 0.03-0.08). Hypothesis confirmed: these four are also drivers.

## Revision 1: v2 = copy8_x080 (written before any pass on it)
Rationale: v1's reason (undo the decoder's 1/0.8 amplification of the copy signal) applies to every head
that carries the copy signal. v1 was limited to the four heads that had been causally tested; passes 18-20
have now tested the other four. v2 applies the same factor 0.8 to all eight: L5H6, L5H7, L5H1, L4H5,
L4H6, L4H4, L3H0, L5H4. The factor is unchanged; only the set of heads grows, on causal evidence.
Predictions: loss on stories 1-64 +0.005 to +0.010 (v1 +0.0038); rep-4 on the 310 prompts -0.004 to -0.006
(v1 -0.0036), loop rate -2 to -4 points, length unchanged (within +-3%).

## v2 outcome (passes 21-26) against the prediction
Loss +0.0069 (predicted 0.005-0.010). rep-4 -0.0056, CI [-0.0095, -0.0017] (predicted 0.004-0.006).
Loop rate -6.8 points, CI [-12.9, -0.7] (predicted 2-4: larger than predicted, but the interval is wide).
Length +2.3%. Prediction held, except that the loop-rate point estimate is larger than expected.

## Confirmation on fresh randomness and fresh stories (written before passes 27-38)
The loop-rate drop of v2 may be partly luck (the predicted size was 2-4 points). Replication: the same 310
prompts with seed 1 (new uniform numbers) for the unedited model and v2, and loss on dev stories 65-128.
Prediction: rep-4 again lower by 0.003-0.007; loop rate lower by 2-6 points; loss delta again 0.005-0.010.
If rep-4 is not lower on seed 1, v2's effect is not established and I will say so.

## Replication outcome (passes 27-38)
Seed 1: rep-4 -0.0082, CI [-0.0120, -0.0044]; loop rate -11.3 points, CI [-17.4, -5.5]; length +0.6%.
Loss on stories 65-128: +0.0078. Both seeds pooled (620 rows): loop rate 26.1% -> 17.1%, rep-4 0.0282 -> 0.0213.
The effect is real and larger than I predicted from the ablations (I expected 2-6 points of loop rate).

## Greedy decoding (written before passes 39-40)
The factor 0.8 was argued from temperature sampling; greedy decoding has no temperature. Hypothesis: the
copy heads also drive greedy loops, so v2 lowers greedy loop rate on prompts 0-63 (TASK.md: 40% overall),
but by less in relative terms than under sampling, because greedy only changes when the copied token's
lead over the runner-up is smaller than the removed contribution. Prediction: loop rate lower by 0-10 points,
rep-4 lower by 10-25% relative.
