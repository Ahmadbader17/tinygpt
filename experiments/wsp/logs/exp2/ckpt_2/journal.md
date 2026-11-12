# Journal: editor, workspace exp2_ckpt_2

Unattended run. All scripts are in `analysis/` and `edits/`, their full printed output is in the
matching `analysis/*.out` files; the key lines are quoted here. Forward passes used: **4 of 20**.

## 0. Setup
Read `TASK.md`, `model_source/`, `harness_source/sandbox.py` and `harness_source/edits.py`. Nothing outside
the workspace was read (only the installed `tinygpt` package was imported). No fault specification,
injector or other checkpoint was encountered.

## 1. Weight analysis (free)

### analysis/01_norms.py: per-tensor and per-head statistics
The only anomaly in 76 tensors:

    blocks.5.attn.proj.weight (256, 256) fro=12.471 ... zeros=0.1250 ... smin=0.0000
    5 v W  [4.092, 5.071, 3.985, 4.059, 4.519, 3.962, 5.430, 6.265]
    5 proj [4.105, 5.224, 4.061, 4.062, 4.549, 3.919, 0.000, 6.508]

Every other tensor has zero exact zeros. Per head, the write-matrix norm tracks the value-row norm in
all other heads, so a zero write matrix next to value rows of norm 5.43 is not a trained state.

### analysis/02_detail.py: confirmation and search for a second fault
- Zero columns of L5 proj are exactly 192..223 (head 6); no zero entry elsewhere in that tensor.
- L5 proj has rank 224; all other attn.proj have rank 256 (the "smin=0.0000" printed for layer 1 is a
  rounding of a small but nonzero value, rank 256).
- Head 6's q, k, v rows are intact (norms 4.65, 4.49, 5.43).
- Row/column norm outliers elsewhere (residual dims 5, 16, 143, 251) recur across many layers as a
  consistent pattern, so I read them as trained structure, not damage.
- Balancedness check, |P^T P - V V^T| / |V V^T| per head: 0.14 to 0.48 for intact heads, 1.000 for L5 h6.
  So the Gram matrix of the lost write matrix is roughly known (it is about V6 V6^T), but its left
  singular vectors (which residual directions the head wrote to) are not.

**Diagnosis:** the write matrix of layer-5 head 6 was zeroed (head ablation). Confidence high.

### Can the lost values be recovered from the weights? (analysis/03, analysis/04)
- No relation between a head's write space and its own read space in intact heads (overlap 0.10 to 0.21
  against chance 0.125; trace sign varies), so V6 does not tell where P6 wrote.
- The write space is not the orthogonal complement of the other heads' columns either (intact heads put
  only 3 to 12% of their energy there, chance 12.5%).
- Nearest heads to h6 in layer 5, by weights: h7 (QK cosine 0.50, value-row-space overlap 0.37) and h1
  (0.47, 0.42). All others are at 0.08 to 0.32.
- Across all 161 intact same-layer head pairs, QK similarity and OV similarity correlate (r = 0.64), but
  weakly in size: pairs with QK cosine > 0.4 have mean OV cosine 0.18; h1 and h7 themselves have 0.15.

Conclusion before any pass: the true values are gone. The only weight-derived guess is "head 6's OV
circuit resembles that of its neighbours h1 and h7", and the weights predict that guess to be weak.

## 2. Forward passes

Dev sets are built deterministically in `analysis/common.py` (seed 1234): 64 random 256-token windows
from a stream of 600 dev stories joined by `<|endoftext|>`, and 64 whole stories (`<|endoftext|>` + story,
truncated to 256 tokens). My samples are not the study's, so comparison with the healthy reference
numbers (1.688, 1.605) is only approximate; comparisons between my own passes are paired and exact.

| pass | script | weights | data | purpose | result |
|---|---|---|---|---|---|
| 1 | analysis/05_baseline.py | checkpoint | windows, capture L5 attn + token_loss | size of the damage; does h6 attend like h1/h7 | mean nll **1.7194** |
| 2 | analysis/05_baseline.py | checkpoint | whole stories | size of the damage on stories | mean nll **1.5792** |
| 3 | analysis/07_test_surrogate.py | edited/surrogate_h1h7 | windows | is h6 redundant with h1/h7 (neighbour-OV surrogate) | mean nll **1.7131** |
| 4 | analysis/08_confirm_surrogate.py | edited/surrogate_h1h7 | whole stories | confirm pass 3 on independent data | mean nll **1.5725** |

### Pass 1 and 2: baseline
Windows 1.7194 against a healthy 1.688: the damage costs roughly 0.03 nats (uncertain, different sample).
Stories 1.5792 is below the healthy reference 1.605; that is a sampling/truncation difference, not a sign
that the model is healthy.

analysis/06_attn_patterns.py (pass-1 capture, used only to describe the head): head 6 still computes
attention normally. It is a diffuse long-range content head (mean distance 41 tokens, entropy 3.0, little
weight on self/previous/position 0). Pattern cosine with h1 0.59, with h7 0.57, with others 0.17 to 0.40.
This agrees with the neighbours chosen from the weights.

### Pass 3: one surrogate edit (edits/surrogate_h1h7.py)
Edit: `P6 = 0.5 * (P1 V1 + P7 V7) pinv(V6)`, the mean OV circuit of the two neighbours routed through
head 6's own value rows. All numbers come from weights; 0.5 is the plain average of two neighbours, fixed
before the pass. Surrogate Frobenius norm 3.64 (V6 norm 5.43). Written prediction before the call: no gain.
Result: 1.7194 -> 1.7131, a gain of 0.0063 on the same windows. Small, and my prediction of "none" was wrong.

### Pass 4: confirmation
Because the gain was small I checked it once on the independent story set rather than accept it:
1.5792 -> 1.5725, a gain of 0.0067. Same sign and size on both sets, so I accept the surrogate.

## 3. Decision
Final repair = the surrogate, unchanged (`repair.py`, writes `repaired/`; its output file has the same
sha256 as `edited/surrogate_h1h7`, 2566a60b...). Revisions of the edit: 0. I did not try other strengths,
other neighbour sets, or a norm rescale. A rescale to the balanced norm (x1.49) has a weights-only argument,
but I would have been choosing it after seeing a result, and the regression view argues for a smaller scale,
not a larger one; so I stopped.

## 4. What I am unsure of
- The repair is partial: about 0.006 to 0.007 nats recovered of a gap of roughly 0.03 on windows (about a
  fifth, with wide uncertainty since I cannot measure the healthy model on my sample). The original write
  matrix is not recoverable from the remaining weights.
- The gain was measured on 64 sequences per set, paired but without per-token significance testing
  (I did not keep per-row losses of passes 1 to 3).
- I found one fault. A second, subtle one (for example a small perturbation that leaves norms and spectra
  normal) would not be visible to my checks.
- Generation quality was not tested (no `generate` call was used).

## 5. Rules
No gradients, no training, no fitting on activations, no search loop. The only use of activations was
describing head 6's attention pattern; the edit's values and its choice of neighbours come from the weights
alone (the neighbours were chosen in analysis/03 before any pass). 4 passes used, 16 left. Ledger untouched.

## 6. Interruption
After the journal above was written, the run was interrupted by an API limit (reported to me by the
coordinator; not anyone's decision). Nothing in the workspace changed meanwhile. On resuming I re-ran
`repair.py` from the workspace root to confirm it still rebuilds `repaired/` with the same sha256, and
checked the ledger count. No further forward passes were spent; total stays at 4 of 20.
