# Weight Space Programming on Tiny GPT: study plan

| Field | Value |
|---|---|
| Status | **Phase 2 running.** Progress and every deviation: `logs/phase2/journal.md` and the change log in section 12. |
| Branch | `exp/weight_space_programming` (from `main` at `a60bae9`) |
| Written | 2026-11-11, before any weight analysis and before any edit |
| Editor | Claude (`claude-fable-5-2`) in Claude Code; fresh subagents of the same model where section 7 says so |
| Original work | Takehi, Teoh, Zhang, Khattab, "Sparks of Weight-Space Programming", MIT CSAIL, 2 Nov 2026 |

This file is the preregistration. Hypotheses, metrics, thresholds, budgets and controls are fixed
here. Any later change is recorded in section 12 with its reason and its date, never made silently.
Section 12 lists the decisions I need from Ahmad.

## 1. What is replicated, and what is not

In the original work a general purpose LLM ("Astra") reads a network's weights, reasons about what
the parts compute, and writes weights directly, with no optimizer. Their eight experiments range from
naming the game a policy plays to editing Qwen3.5-0.8B so that it stops looping on maths problems.

| This study | Closest original experiment | Same | Different |
|---|---|---|---|
| Exp 1: read 48 heads from weights only | Exp 3: find copying heads in char-level transformers, no forward passes | weights only; ground truth from a scripted activation probe the editor never sees first | six head classes instead of one; BPE model; I score against chance explicitly |
| Exp 2: blind repair of planted faults | none (nearest: Exp 4, fixing a failing policy) | diagnose and repair by writing weights | synthetic faults with a known answer; new in this study |
| Exp 3: reduce repetition | Exp 8: remove a "looping direction" from two MLPs of Qwen3.5-0.8B at 10, 25, 50 forward passes | same three budgets; no gradients; editor chooses its own practice inputs | 5.8M model instead of 0.8B; repetition metrics instead of maths accuracy; sealed test set, random-edit control and a decoding baseline, none of which the original reports |

Two cautions carried over from the original. Their looping fix worked on a model where looping hid
an ability (93% of base answers hit the token limit) and their follow-up on Gemma 3 270M, which
rarely loops, found no gain. They conclude that editing "could remove an obstacle that hid an
ability, but it could not add an ability the model lacked". Section 2 shows Tiny GPT is closer to
the second case than the first.

I read the post through a fetch subagent's summary, not the page itself. Before the write-up every
number and quotation attributed to the original will be checked against the page again.

## 2. Baseline (measured, dev data only)

**Checkpoint.** `Ahmadbader17/tiny-gpt` at Hub revision `36ee8ed`, `model.safetensors` sha256
`6d961505...cb86ec`, byte-identical to `export/` and to `runs/overnight_long/checkpoints/best`.
Stored at `runs/wsp/original/` (gitignored, read-only for this study).

**Reported number reproduced.** `estimate_loss(model, val.bin, 200 batches, 32 x 256, VAL_SEED)`
gives **1.6877** (perplexity 5.407). The README reports 1.688 and `model_info.json` 1.6877.

**Baseline on this study's dev protocol** (`experiments/wsp/results/phase0_baseline.json`, produced
by `experiments/wsp/phase0_baseline.py`; 95% bootstrap intervals over stories or prompts):

| Dev measurement | Value |
|---|---|
| Per-story loss, 2,500 dev stories, 503,193 tokens | 1.605 [1.588, 1.622], perplexity 4.98 |

The per-story loss is lower than 1.688 because the README recipe scores random windows that start
mid-story with no context, while this protocol starts every story after `<|endoftext|>`.

Generation: 300 dev prompts, 200 new tokens at most, stop at `<|endoftext|>`.

| Dev generations | Loop rate | Long-loop rate | rep-2 | rep-3 | rep-4 | Mean length | Ended by itself |
|---|---|---|---|---|---|---|---|
| Real stories (same prompts, real continuation) | 0.187 [0.143, 0.230] | 0.003 [0.000, 0.010] | 0.154 | 0.051 | 0.0182 | 170.5 | n/a |
| Model, sampling (temp 0.8, top-k 40), 5 seeds | 0.243 [0.219, 0.268] | 0.003 [0.001, 0.007] | 0.180 | 0.066 | 0.0270 | 162.6 | 73% |
| Model, greedy | 0.400 [0.347, 0.457] | 0.117 [0.080, 0.153] | 0.251 | 0.136 | 0.0892 | 158.8 | 78% |
| Model, sampling + repetition penalty 1.2 | 0.035 [0.025, 0.045] | 0.000 [0.000, 0.000] | 0.070 | 0.015 | 0.0036 | 151.2 | 84% |

The long-loop column (8-gram three times) separates the two kinds of loop. Sampled generations
almost never contain a long loop (0.3%, the same as real text), so the sampling loops are short
phrase repeats. Greedy decoding produces real long loops (11.7%). The sampling loop-rate gap
over real text is therefore mostly short-phrase repetition, the kind that the repetition penalty
removes at a cost of 7% length.

Three things in this table shape the plan:

1. **Real stories trip the loop metric too.** 19% of real TinyStories continuations contain a
   4-gram three times (short stories reuse phrases such as ". She was very"). The model's excess
   over real text under the README sampling settings is about 6 points of loop rate and 0.009 of
   rep-4. The excess is large only under greedy decoding (40% against 19%).
2. **The conventional fix overshoots.** A repetition penalty of 1.2 cuts loop rate to 3.5%, far
   below real text, and shortens stories by 7%. "Less repetition" is not the same as "closer to
   real stories", so the report will show the real-text level next to every repetition number.
3. **Sampling noise.** Two disjoint seed sets of the same model differ by -0.006 in loop rate with
   interval [-0.036, +0.024], and by -0.0006 in rep-4 with interval [-0.0027, +0.0015]. With 300
   prompts, effects smaller than about 0.03 (loop rate) or 0.002 (rep-4) are invisible. The sealed
   test therefore uses 600 prompts (section 3), which shrinks these by about 1.4 times.

Disclosure: these numbers came from forward passes I ran as harness builder, outside any editor
budget. The script prints and saves aggregates only; I have not read any generated text. Every
editor in Exp 3 is given this table, so no attempt has an information advantage from it.

## 3. Data protocol

Only `train.bin` and `val.bin` exist. `val.bin` holds the first 5,000 stories of the official
`TinyStoriesV2-GPT4-valid.txt`. The whole file's story count is not measured: I did not stream the sealed range to count it, so the plan relies only on the sealed stories being past index 5,000.

| Split | Source | Who may use it |
|---|---|---|
| Dev stories | valid-file stories 0 to 4,999 (`data/raw/val.txt`, already on disk) | editors, harness |
| Dev loss subset | dev stories 0 to 2,499 | editors, harness |
| Dev prompts | first 8 words of dev stories 2,500 to 4,999, first 300 distinct, plus the 10 README prompts | editors, harness |
| **Sealed test stories** | valid-file stories 5,000 to 6,999 (2,000 stories) | final evaluation script only |
| **Sealed test prompts** | first 8 words of valid-file stories 7,000 onwards, first 600 distinct that are not dev or README prompts | final evaluation script only |
| Sealed audit block | valid-file stories 20,000 to 21,999 | only if section 9 triggers an audit |

**Why fresh stories instead of splitting `val.bin`.** I recommend the fresh range for three reasons.
(a) `val.bin` chose the `best` checkpoint and produced the headline 1.688, so it is the right split
to spend as dev. (b) The sealed stories are **not on this machine** during editing: the file is
pinned by dataset commit `f54c09f` and sha256 `6874bae9...585d`, recorded from Hub metadata without
downloading, and only `final_eval.py` fetches it. Sealing is then a fact about the disk, not a
promise. (c) Sealed prompts are defined by a rule over stories nobody has read, so nobody, including
me, can tune to them.

**Leakage risks and what is done about each.**

| Risk | Handling |
|---|---|
| Sealed stories were seen in training | Training used only the first 300,000 stories of the *train* file. The valid file is a separate file. |
| Near-duplicate stories across train and valid (TinyStories is templated) | Affects absolute loss equally for every model; all claims are paired differences between models on the same stories. Exact duplicates of any train or dev story are dropped at evaluation time and counted. |
| Identical short prompts in dev and test ("Once upon a time, there was a little") | Sealed prompts exclude every dev and README prompt. Cost: sealed prompts lean slightly towards rarer openings; identical for all models. |
| Editor reads sealed data | Not on disk until the final evaluation. After that, edits are frozen (section 8). |
| Editor learns from test results and edits again | All edits of Exp 2 and Exp 3 are committed and tagged before `final_eval.py` first runs. Every run of it is logged in `logs/sealed_access.jsonl`. |
| Exp 3 editor profits from Exp 1 verification passes on dev | Exp 3 editors are fresh subagents that are not given Exp 1 or Exp 2 material (section 7, decision D1). |
| Dataset file changes upstream | Pinned commit and sha256; the script refuses a mismatch. |
| The validation file's first story is a fragment (it starts mid-sentence) | It is dev story 0; harmless for loss; prompts come from stories 2,500 onwards. |

## 4. Metrics (exact definitions)

Code: `src/tinygpt/wsp/metrics.py`, `scoring.py`, `generation.py`, with tests.

**Loss.** A story is scored as `<|endoftext|> story <|endoftext|>`: every story token and the stop
token are predicted. Stories longer than 256 tokens are cut into consecutive non-overlapping
windows. Test loss is total negative log likelihood over total tokens (nats per token); perplexity
is its exponential. Interval: bootstrap over stories, 10,000 resamples, percentile, seed fixed.
Differences between two models are paired on the same stories.

**Generation.** Token ids of the continuation only (no prompt, no stop token), at most 200 new
tokens, stopping at `<|endoftext|>`. Two decoders: the README settings (temperature 0.8, top-k 40,
seeds 0 to 4) and greedy (one run per prompt). Sampling uses common random numbers: token t of
(prompt p, seed s) is drawn by inverse CDF from a uniform number that depends only on (p, s, t), so
two models see identical randomness.

| Metric | Definition |
|---|---|
| **Loop rate** (primary) | share of generations in which some 4-gram occurs 3 or more times |
| **rep-4** (primary) | repeated 4-gram fraction of a generation, 1 - unique 4-grams / all 4-grams, averaged over generations |
| rep-2, rep-3 | same with 2-grams and 3-grams |
| distinct-1, 2, 3 | unique n-grams / all n-grams pooled over all generations of a model (comparable only at equal sample counts) |
| Long-loop rate | share of generations in which some 8-gram occurs 3 or more times (rare in real text, so it isolates pathological loops) |
| Length, stop rate | mean new tokens; share of generations ending with `<|endoftext|>` |
| Judge loss | mean per-token loss of a model's sampled generations under the **unedited** model (rises if an edit produces text the original finds unlikely) |
| Real-text reference | the same repetition metrics on the real continuations of the same prompts |

Intervals for generation metrics: bootstrap over prompts (the seeds of one prompt stay together),
10,000 resamples. Comparisons are paired on prompt and seed.

## 5. Editor rules: how each is interpreted and enforced

| Rule | Enforcement |
|---|---|
| 1. No gradients, backward calls, optimizers, training loops | The sandbox disables autograd globally and loads weights with `requires_grad=False`. A test parses every file under `edits/` and every saved analysis script and fails on `torch.optim`, `.backward`, `requires_grad`, `enable_grad`, `set_grad_enabled(True)`, `autograd`. |
| 2. No fitting weights to data | The same test also fails on `lstsq`, `pinv`, `linalg.solve`, `sklearn`, `scipy.optimize`. See interpretation I1. |
| 3. No search over edits scored by evaluation; at most 3 revisions | Every forward call needs a written `purpose` stating the hypothesis it tests, logged before it runs. An attempt is one initial edit plus at most 3 revisions, each with a rationale in the journal; the ledger records which edit version each call ran. I audit every ledger for sweep patterns (the same edit at several strengths, or many candidate locations scored in turn). |
| 4. Forward passes rationed | One function, `Sandbox.forward`, with a hash-chained ledger on disk; call number budget + 1 raises `BudgetExceeded`. Section 6 defines one pass. |
| 5. Editor never sees sealed test data, sealed prompts, or fault specs | Sealed test data is not on disk during editing. Fault specs and the clean checkpoint are on disk during Exp 2, so that part rests on instructions, plus tripwires: access times of every copy are armed before each subagent starts and checked after (verified to work on this filesystem), and scripts are scanned for those paths. |
| 6. Original checkpoint and `main` untouched | Edits are written to `runs/wsp/`; the original's sha256 is checked at the start and end of every script. No pushes without Ahmad's approval. |

**Interpretations I will follow unless Ahmad says otherwise (decision D2).**

- **I1, what may be written into weights.** Every number written into a weight is a function of
  (a) existing weights and (b) scalars chosen by stated reasoning. It is never a function of
  activations or data. So "scale head 3.2's output by 0.5" and "remove the top singular direction
  of this matrix" are allowed. "Subtract the mean activation difference between looping and
  non-looping text" is **not**, even though it is not a regression, because the numbers written
  would be estimated from data. Forward passes are for testing hypotheses about which component
  matters, not for producing edit values. This is stricter than the original, whose Exp 8 direction
  may have come from activations. It will cost some effect size; I think it is what rule 2 means.
- **I2, what counts as free weight analysis.** Free: any computation on weights that takes no token
  sequence as input, including SVD, norms, cosines, products of matrices along one circuit path
  (for example `W_E^T W_Q^T W_K W_E`), LayerNorm folded in as a fixed gain, and a softmax over one
  such score matrix to read it. Not free: anything that chains two or more nonlinear sublayers or
  evaluates the network on a token sequence, such as a hand-written forward pass. Those go through
  the budgeted function.
- **I3, revisions.** "At most 3 revisions per experiment" is applied per attempt: per damaged
  checkpoint in Exp 2 and per budget in Exp 3. The editor may declare any of its versions final.

## 6. Forward pass unit and budgets

One forward pass is one call to `Sandbox.forward` on **one set of weights**, of one of two kinds:

| Kind | Limit per call | Returns |
|---|---|---|
| `read` | 64 sequences of up to 256 tokens | logits, per-token loss, and any requested activations (attention patterns, per-head outputs, residual stream, MLP hidden) |
| `generate` | 64 continuations (prompt and seed pairs), up to 200 new tokens, one decoder setting | tokens, text, and the repetition metrics of section 4 |

Comparing an edit with the original on the same inputs costs two passes. The original's unit was
larger (up to 1,024 texts read or 512 problems answered on a 0.8B model); this one is scaled to the
model and keeps the budgets binding: 64 generations estimate a 24% loop rate to about plus or minus
5 points, so an editor cannot rank small edits by evaluation.

| Experiment | Editor budget |
|---|---|
| Exp 1 predictions | 0 |
| Exp 2, per damaged checkpoint | 20 |
| Exp 3, three attempts | 10, 25, 50 |

Harness passes (baseline, fault calibration, Exp 1 verification, final evaluation) are not editor
passes. They are logged with their role, and their outputs are aggregates or sealed result files.

## 7. Experiments

### Exp 1: reading heads from weights

**Procedure.** With zero forward passes I assign each of the 48 heads a confidence in [0, 1] for
each of six classes, with a one-line reason from the weights. The file
`results/exp1_predictions.json` is committed before the verification script is run for the first
time; the script refuses to run otherwise and records that commit hash. The script and its
thresholds are written and committed in Phase 1, before I look at any weight.

**Ground truth (scripted, dev data).** NAT = 256 windows of 256 tokens, window j starting at the
`<|endoftext|>` before dev story 8j. REP = 64 sequences of `<|endoftext|>` followed by 100 random
token ids repeated twice (seed 0). A is a head's attention pattern.

| Class | Score | A head is in the class if |
|---|---|---|
| previous token | mean A[t, t-1] on NAT, t >= 1 | >= 0.40 |
| start anchor | mean A[t, 0] on NAT, t >= 16 (position 0 holds `<|endoftext|>`) | >= 0.40 |
| induction | mean A[t, t-99] on REP for t in the second copy (the token after the earlier occurrence) | >= 0.20 |
| duplicate token | mean A[t, t-100] on REP for t in the second copy | >= 0.20 |
| positional | mean cosine similarity between the attention patterns (rows t >= 16) of consecutive NAT windows, which hold different text | >= 0.80 |
| mostly inactive | absolute change in dev loss (500 stories) when the head's output is zeroed | < 0.002 nats |

Classes overlap (a previous-token head is also positional). A prediction is positive at
confidence >= 0.5.

**Reported.** Per class: precision, recall, F1, base rate, and an exact hypergeometric p-value;
micro-averages over classes with at least one positive; Spearman correlation between confidence and
the continuous score (defined even when a class is empty); Brier score against a base-rate
predictor. Chance is assessed with two permutation nulls of 10,000 shuffles: heads shuffled
globally, and heads shuffled **within their layer**. The second is the real test, because "early
layers are positional" is prior knowledge, not weight reading.

**H1 (success).** Micro-F1 exceeds the within-layer null at p < 0.05, and micro-precision >= 0.5.
**H1 fails** otherwise, and the report says so.

### Exp 2: blind repair of planted faults

**Injector.** `inject_faults.py` builds 5 damaged checkpoints from a seed, one per family, in a
seeded order so the checkpoint number says nothing about the family:

| Family | Fault |
|---|---|
| Head scaled up | one head's output slice of `attn.proj.weight` multiplied by a factor drawn from [3, 8] |
| Rank-one addition | `alpha * u v^T` added to one `mlp.proj.weight`, random unit u and v, alpha drawn from [1, 3] times the matrix's spectral norm |
| Head zeroed | one head's output slice set to zero |
| LayerNorm gain | a random quarter of one `ln1` or `ln2` gain vector multiplied by a factor drawn from [4, 8] |
| Noise | Gaussian noise with standard deviation drawn from [0.5, 1.5] times the matrix's own, added to one of `qkv`, `attn.proj`, `mlp.fc`, `mlp.proj` |

Layer, head, directions and magnitudes are drawn from the seed. **Calibration**, done by the
injector and not by any editor: dev loss on 500 stories must rise by between 0.05 and 1.0 nats,
with a paired interval excluding zero; otherwise the magnitude is multiplied or divided by 1.5 and
retried, at most 12 times. A zeroed head has no magnitude, so heads are tried in seeded order until
one costs at least 0.02 nats. Specs go to a sealed file that I do not open until the final
evaluation; the injector prints only "checkpoint k: damage confirmed".

**Editors.** One fresh subagent per damaged checkpoint, same model as me, each in its own workspace
outside the repo containing only: the damaged checkpoint, the tokenizer, the dev stories and
prompts, the model source for reference, the sandbox with a budget of 20, the editor rules, and the
healthy model's published validation loss. The task text is identical for all five and names no
fault family. It forbids reading the clean checkpoint (any copy, including the Hub), the injector
and the specs. Deliverables: `diagnosis.json` (tensor names, layer, component, head if any, fault
type, confidence), a deterministic `repair.py`, the journal, the ledger.

**Metrics** (sealed test stories, by script, no judgment calls).
- Recovery = (loss damaged - loss repaired) / (loss damaged - loss clean), with a paired bootstrap
  interval.
- Location correct at tensor level (named tensor equals the spec's), and at sub-location level
  (head index or gain vector) where that applies.
- Control: 5 random edits per checkpoint with the same support and the same Frobenius norm per
  tensor as the editor's repair.

**H2a (localisation).** The right tensor is named for at least 3 of 5 checkpoints.
**H2b (repair).** For at least 3 of 5 checkpoints, recovery has an interval above zero **and**
exceeds all 5 random controls. Mean recovery is reported with every individual value.
A zeroed head and added noise destroy information, so near-complete recovery on those would be a
warning sign, not a success (section 9).

### Exp 3: reducing repetition in the real model (headline)

**Attempts.** Three independent attempts at budgets of 10, 25 and 50 passes. I recommend each is a
fresh subagent with its own workspace (decision D1), because attempts run one after another in this
session would share knowledge: the "25" attempt would really have 35 passes of information. Each
editor gets the original checkpoint, the sandbox, dev stories and prompts, the README, the rules,
section 4, and the baseline table of section 2. It does not get the other attempts, Exp 1 or Exp 2.
The prompt text is committed in Phase 1, before any experiment, so I cannot leak hints into it later.
Deliverables: `edits/exp3_b{10,25,50}.py`, journal, ledger, the declared final version.

**Evaluation** on 600 sealed prompts x 5 seeds (sampling) and 600 greedy runs, and 2,000 sealed
stories, all paired with the unedited model.

**H3 (success for one attempt).** All of:
1. loop rate falls: interval of (edited - unedited) entirely below zero;
2. rep-4 falls: interval entirely below zero;
3. test loss rises by no more than the tolerance: upper end of the interval of the paired loss
   difference <= **+0.02 nats**;
4. guardrail: mean generation length >= 90% of the unedited model's, so that repetition is not
   "reduced" by ending stories early.

Criteria 1 and 2 use the README sampling settings. Greedy results are secondary and always shown.
**Headline claim.** "Weight Space Programming reduced repetition" is made only if at least one
attempt meets H3 with intervals at the 98.3% level (Bonferroni for three attempts). Each attempt is
also reported at 95%.

**Why 0.02 nats.** It is 1.2% of the loss and about what the last 1,300 of 9,500 training steps
bought (validation loss 1.727 at step 8,250 and 1.708 at step 9,500, same 20-batch recipe: 0.018
nats in 1,250 steps). An edit
that costs more than that has undone a visible share of training.

**Comparisons.**
- Unedited model.
- Random edit matched in location, shape and norm, 5 seeds per attempt: same tensors, same support
  (the entries the edit changed), Gaussian entries, same Frobenius norm per tensor. If the edit has
  rank 4 or less on its support, a second control with random factors of the same rank and norm is
  added. The edit is "specific" if its loop-rate reduction exceeds all 5 controls.
- Decoding-time repetition penalty 1.2 on the unedited model (the value recommended by Keskar et
  al. 2019; fixed now, not tuned). To compare cost fairly I also report the penalised decoder's
  own test loss, by applying the penalty to the next-token distribution on real sealed stories.
- Real-text reference level.

**Test loss of the edited model** is always reported. A loss lower than the unedited model's, with
an interval excluding zero, is not expected and triggers the audit of section 9 before it appears
anywhere.

## 8. Harness (Phase 1)

| Piece | Design |
|---|---|
| `tinygpt.wsp.sandbox` | `load_weights` (plain dict of tensors, no autograd); `Sandbox(workdir, budget).forward(kind, weights, ..., purpose)` with the hash-chained ledger and the hard budget; `write_edit` saves the edited checkpoint and `diff_summary.json` (tensors touched, count and fraction of weights changed, relative L2 change per tensor and overall, sha256 before and after). Refuses paths under the sealed directories. |
| `experiments/wsp/edits/*.py` | One standalone deterministic script per edit, rebuilding it from the original (or from the damaged checkpoint in Exp 2). A test rebuilds each twice and compares hashes. An edited checkpoint without a script does not count. |
| Static test | Parses `edits/` and saved analysis scripts and fails on the patterns in section 5. |
| `experiments/wsp/final_eval.py` | The only code that names the sealed URL. Requires a clean working tree and a committed manifest of frozen checkpoints (sha256 each), downloads and verifies the sealed file, writes result JSON under `results/`, appends to `logs/sealed_access.jsonl`. |
| Sanity tests | identity edit reproduces baseline exactly; evaluation is deterministic; call budget + 1 is refused; ledger tampering is detected; cached generation matches `GPT.forward` and greedy `generate` (already in). |
| Journal | `experiments/wsp/logs/<experiment>/`: `journal.md` (reasoning, code run, output, passes spent), the ledger, every analysis script and its output. Released as is. |
| Figures | `make_figures.py` reads `results/*.json` only. |

Already built and tested in Phase 0 because the baseline needed them: metrics, cached generation,
per-story scoring, the data protocol module (21 new tests, 141 passing in total, `make check` green).

**Order and freeze points.** Phase 1 harness and all editor prompts committed -> Exp 1 predictions
committed -> Exp 1 verification -> Exp 2 injection and five editors -> Exp 3 three editors -> tag
`wsp-freeze` -> `final_eval.py` once -> figures and report.

## 9. If a result looks too good

Each of these stops the write-up until a written audit is in the journal: edited test loss below
the unedited model's with an interval excluding zero; Exp 2 recovery above 100%, or above 90% for
the zeroed-head or noise fault; a repair that restores clean weights almost exactly (cosine above
0.99 between the repair and the true inverse of a fault that destroyed information); Exp 1
micro-F1 above 0.9; a loop-rate drop of more than half with no measurable loss cost; any tripwire
or ledger anomaly. The audit checks for leaks (script inputs, access times, paths), reruns from a
clean rebuild, and for loss claims repeats the measurement on the sealed audit block.

## 10. Runtime on this laptop (Core Ultra 9 185H, 12 threads, no GPU)

Measured in Phase 0: scoring 2,500 stories takes 61 s; 1,500 sampled generations 110 s; 300 greedy
generations 19 s.

| Step | Estimate |
|---|---|
| Phase 1 tests | under 1 minute |
| Exp 1 weight analysis | seconds per script; my reasoning time dominates |
| Exp 1 verification (attention probes, 48 head ablations x 500 stories) | about 12 minutes |
| Exp 2 fault calibration | 5 to 15 minutes |
| Exp 2 editors, 5 x 20 passes | compute under 5 minutes in total; about 20 to 40 minutes of agent time each |
| Exp 3 editors, 85 passes in total | compute under 15 minutes; about 30 to 60 minutes of agent time each |
| Final evaluation, Exp 3: 19 models (unedited, 3 edits, 15 random controls) at about 6 minutes each, plus the repetition penalty runs | about 2 hours |
| Final evaluation, Exp 2: 36 models, loss only, 48 s each | about 30 minutes |
| Figures | under 1 minute |

Total compute is about 3.5 hours, of which the final evaluation is one background run of about
2.5 hours. CLAUDE.md asks for approval before runs over 10 minutes (decision D4).

## 11. Honest prior

| Hypothesis | My probability it is met | Why |
|---|---|---|
| H1, reading beats within-layer chance | 55% | Positional and previous-token heads should be visible in the positional part of the QK circuit, especially in layers 0 and 1. Induction needs composition through earlier layers and LayerNorm, which is much harder to read. Several classes may turn out empty in a 6-layer TinyStories model. |
| H2a, right tensor for 3 of 5 | 65% | Scaled heads, corrupted gains, exact zeros and a large rank-one term are outliers against the other layers. Noise is the hard one. |
| H2b, real repair for 3 of 5 | 35% | Scaling, gain and rank-one faults are invertible in principle from peer statistics. A zeroed head and noise are not; expected recovery near zero for those two. |
| H3, any attempt meets the criteria | 15 to 20% | See below. |

For H3 the evidence so far points towards a null. The model's excess repetition over real text under
the README settings is small (about 6 points of loop rate), so there is little to remove before an
edit starts deleting repetition that real stories have, which costs loss. Copying recent tokens is
also how a small model gets names and objects right, so the components that drive repetition are
likely the ones that carry much of the loss; I expect a trade-off that breaks the 0.02 tolerance
before the repetition change becomes detectable. Interpretation I1 removes the most direct route
(a direction estimated from activations). And the original authors' own Gemma result suggests that
weight editing helps when one removable fault hides an ability, which is not this model's
situation. The likeliest outcomes, in order: no detectable change at any budget; a detectable drop
that fails the loss tolerance; a small real effect, clearest under greedy decoding. I expect the
repetition penalty to beat every edit on repetition, while overshooting the real-text level.

## 12. Decisions needed from Ahmad, and the change log

| # | Decision | My recommendation |
|---|---|---|
| D1 | Who edits in Exp 3: three fresh subagents (independent), or me in this session (sequential, sharing knowledge)? | Fresh subagents, same model. |
| D2 | Interpretations I1 (no activation-derived numbers in weights), I2 (what is free analysis), I3 (3 revisions per attempt). | Accept all three as written. I1 is the consequential one: relaxing it would make Exp 3 closer to the original and more likely to work, at the cost of sitting close to "fitting to data". |
| D3 | Sealed test from fresh validation stories not on disk until the end, instead of splitting `val.bin`. | Fresh stories. |
| D4 | Approval for the long runs in section 10, in the background. | Approve. |
| D5 | Thresholds: loss tolerance +0.02 nats; length guardrail 90%; repetition penalty 1.2; Exp 1 class thresholds; fault damage window 0.05 to 1.0 nats. | Accept; all were fixed before any weight analysis. |

Change log:

- 2026-11-12, D1 to D5 approved with the recommended defaults. Editor model for Exp 2 and Exp 3:
  Fable 5.2, as written in section 7.
- 2026-11-12, Phase 1: the sealed-data guard also refuses any file whose name contains
  `GPT4-valid`. Not in the original plan; it closes a path that section 5 would not otherwise
  catch.
- 2026-11-12, Phase 1: the penalised decoder's test loss (section 7) is not implemented yet.
  Listed as an open item in `logs/phase1/journal.md`, to be done before the freeze.
- 2026-11-12, start of Phase 2, **before any weight of the checkpoint was inspected**:
  - The Exp 1 verification script (`exp1_verify.py`) and its scoring module were written and
    committed now (commit `805f12d`), not in Phase 1 as section 7 said. The condition that
    matters holds: they were fixed before any weight analysis.
  - Exp 1 natural-text windows are j = 1..256 (the `<|endoftext|>` before dev story 8j), not
    j = 0..255: story 0 has no preceding `<|endoftext|>`.
  - Exp 1 also reports micro-averages over all six classes, where false positives in an empty
    class count. The H1 rule is unchanged (classes with at least one positive).
  - The editor's budget is no longer an argument the editor passes. The harness writes it as
    the first record of the workspace ledger (`init_budget`), and the sandbox reads it there.
  - The `read` call now returns what section 6 promised (per-token loss, logits, attention,
    per-head outputs, residual stream, MLP hidden); Phase 1 had only loss and attention.
  - The static rule scan also forbids constructing or running the model outside the sandbox,
    and it now covers the editors' archived scripts.
  - Exp 2 editors run one after another, and only one workspace exists at a time, because two
    damaged checkpoints side by side would reveal both faults by subtraction.
  - Exp 2: a zeroed head is chosen by trying all 48 heads in a seeded order (first one costing
    at least 0.02 nats); if none does, the costliest head is used and the spec records it.
  - Editor prompts were corrected before any experiment: the Exp 3 prompt no longer says
    "some stories never end" (not measured); both prompts gained run instructions and the full
    sandbox interface; the Exp 2 prompt gives both healthy-loss reference values (1.688, 1.605).
  - An editor's edit is the checkpoint rebuilt from its script in a clean folder that holds
    only the base checkpoint and the editor's Python files (`pipeline.py rebuild`). No dev data
    or saved activations exist there, so an edit whose values depend on data cannot be rebuilt.
  - The final evaluation flags the section 9 "too good" conditions automatically
    (`audit_triggers` in `final_eval.json`).
- 2026-11-12, before the Exp 2 injection: the access-time tripwires also watch the earlier
  editors' collected logs, archived edit scripts, rebuilt checkpoints and random controls
  (commit `fe28f14`). Without this a later Exp 2 editor could have compared its damaged
  checkpoint with an earlier rebuilt one unseen. Exp 3 attempts also run one after another,
  each workspace deleted before the next is built, so attempts cannot read each other.
- 2026-11-12, after the five Exp 2 editors: the static rule scan flagged two editors for
  `pinv` (`ckpt_2`, in its repair) and `lstsq` (`ckpt_3`, in one descriptive analysis script).
  Both are applied to weight matrices only, which interpretation I1 allows; the scan's name
  match is a proxy for fitting to data and is over-broad. The rule test now carries a
  per-file, per-pattern list of these audited cases instead of a relaxed scan. The report
  states H2 with and without `ckpt_2`. Ahmad may overrule this reading.
- 2026-11-12: `git config core.trustctime false` is set in the working clone so that arming
  the tripwires does not make git re-read the armed files (journal, false alarm at `ckpt_1`).
