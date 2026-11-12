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
