# Diagnosis: what drives repetition, and the edit

## What "repetition" is in this model
Under the evaluation's sampling (T=0.8, top-k 40) the loops are not tight cycles. In the 64
baseline generations of pass 1, every looping story loops on a re-mentioned entity or phrase that is
already in its context: ' Velvet' (9 times), ' Bouncy.', 'The lawyer', ' the vacation',
' big, red ball', ' did not move.', ' wanted to play with'. Multi-token rare names form 4-grams by
themselves. So the thing to look for is the circuit that says "this was said before, say it again".

## Components
Seven attention heads, found from the weights and confirmed with one capture pass:
**5.7, 5.6, 5.1, 4.5, 4.6, 3.0, 5.4** (layer.head).

Evidence from weights alone (`analysis/06_copy_extended.py`): take the OV circuit of each head
(`proj[:, head cols] @ W_V[head rows]`), feed it each token's representation `wte + mlp0(wte)`, and
read the result through `ln_f` gain and the tied unembedding. For these heads the attended token's
own logit is among the 10 largest of 4096 for 76% (5.7), 68% (4.6), 62% (5.6), 43% (5.1), 42% (3.0),
20% (4.5), 14% (5.4) of the vocabulary; chance is 0.2%. They copy what they attend to. Most heads in
layers 1-3 do the opposite on raw embeddings (negative OV eigenvalue sums), i.e. they push attended
tokens down. (My first weights-only score, on raw embeddings, was only partly right: it flagged 5.0,
which is not a copy head, and missed 4.5, 3.0, 5.4. The capture pass exposed this and the extended
embedding fixed it.)

Evidence from pass 2 (32 baseline generations + 32 real stories, `analysis/05_attribution.py`): at
positions whose next token completes an already-seen 4-gram, these heads put 13-27% of their
attention on "induction sources" (an earlier copy of the next token that follows an earlier copy of
the current token), and their direct contribution to the next token's logit exceeds their
contribution at new-token positions by 0.59 (5.6), 0.57 (5.7), 0.40 (4.5), 0.34 (5.1), 0.18 (3.0),
0.17 (4.6), 0.17 (5.4). No other head exceeds 0.07. Heads 4.4, 3.4, 3.5 copy by their OV circuit but
show almost no repeat-specific contribution, so they were left alone.

Important limit of the diagnosis: the same heads behave the same way on real stories (gaps 0.61,
0.68, 0.36, 0.34 for the top four). This is the model's ordinary, useful mechanism for re-mentioning
characters and objects; there is no faulty component that fires only in generations. Real stories
also loop (19% by the same measure). The model's excess is small (24% vs 19%), and reducing the
copy circuit is a trade against predicting real text, not a repair.

## The edit (`edits/copy_heads_08.py`, final)
Multiply the 32 output-projection columns of each of the seven heads by **0.8**. Three tensors
(`blocks.{3,4,5}.attn.proj.weight`), 57,344 weights (0.98%), relative L2 change of the model 0.0196.

Why 0.8: sampling at temperature 0.8 multiplies every logit contribution by 1.25, including the
copy heads' "say it again" evidence. Scaling exactly that evidence by the temperature brings it
back to the strength the model was trained to have, under the decoding used in the evaluation. The
scalar is the decoding temperature; it was fixed before any edited-model pass and never changed.
Which heads to scale was decided by two criteria written down before the edit (weights-only copy
score > 0.1 and repeat-specific logit contribution > 0.15); no number from an activation or an
output was written into a weight.

## Measured effect (dev prompts 0-255, seeds 0 and 1, paired; 128 dev stories for loss)
| Measure | Unedited | Edited | Difference (95% CI, bootstrap over prompts / stories) |
|---|---|---|---|
| Loop rate | 0.254 | 0.207 | -0.047 [-0.092, -0.002] |
| rep-4 | 0.0275 | 0.0237 | -0.0037 [-0.0065, -0.0009] |
| Mean length | 160.9 | 162.0 | +1.05 [-2.4, +4.6] |
| Dev loss (nats/token) | 1.5949 | 1.6023 | +0.0074 [+0.0054, +0.0094] |

Teacher-forced check (pass 10): the edit lowers the loss of new tokens (-0.006 generated, -0.013
real) and raises the loss of already-seen tokens (+0.02), i.e. it shifts probability away from
re-use, on real text as well as generated text.

## What I am unsure of
- The loop-rate interval barely excludes zero, and both seeds were pooled to get there (seed 0
  alone was about 1.4 standard errors). The true effect could be anywhere from near zero to nine points.
- The loss cost is real and consistent (about a third of the 0.02 allowance on my 128 stories). I do
  not know the scoring convention of the sealed test (for example an end-of-text prefix or longer
  stories); my baseline 1.595 is close to, not equal to, the stated 1.605.
- Specificity is not shown: I ran no control edit on non-copy heads, so I cannot say from my own
  passes that the same scaling elsewhere would not do the same.
- The temperature argument for 0.8 justifies the scalar; it does not prove that 0.8 is the best
  strength. No other strength was tried.
- The attribution is direct-path only. Heads 3.0, 4.5, 4.6 also feed later layers, and that indirect
  effect was not measured.
- Greedy decoding, where long loops are common, was not tested.
