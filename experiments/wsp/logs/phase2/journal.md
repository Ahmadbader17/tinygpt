# Phase 2 journal: experiments

Editor model: Claude Fable 5.2 (`claude-fable-5-2`), in the main session for Exp 1 and as fresh
subagents for Exp 2 and Exp 3. Entries are in order of work.

## 2026-11-12: preparation, before any weight was inspected

Phase 1 left open items (see `../phase1/journal.md`). All were closed before the first
experiment, and the code is committed:

| Item | Where |
|---|---|
| Exp 1 verification script and scoring, with thresholds from PLAN section 7 | `exp1_verify.py`, `src/tinygpt/wsp/headscore.py` |
| Penalised-decoder test loss | `scoring.penalise_seen`, `story_nll(..., repetition_penalty=)` |
| Judge loss (generations scored by the unedited model) | `scoring.continuation_nll`, `evaluate.add_judge` |
| Fault families and the injector | `src/tinygpt/wsp/faults.py`, `inject_faults.py` |
| Matched random controls (Gaussian, and rank-matched for low-rank edits) | `src/tinygpt/wsp/controls.py` |
| Workspace builder with a harness-issued budget | `make_workspace.py`, `sandbox.init_budget` |
| Collect, audit, clean rebuild, controls, manifest | `pipeline.py` |
| Access-time tripwires | `tripwire.py` |
| Final evaluation for both experiments, with automatic audit triggers | `final_eval.py` |

Checks: `make check` green. The verification code was smoke-tested on an **untrained** model of
the same shape (random weights), so that no behaviour of the real checkpoint was seen before
the predictions. On random weights every head attends uniformly: previous-token score 0.02,
positional score 0.99. That confirms the scores are wired correctly, and it is a reminder that
the "positional" class as defined also contains heads that simply average over the context.

The pipeline (workspace, two budgeted reads, a refused third, collect, clean rebuild, controls)
was dry-run with a placeholder edit that halves one positional-embedding row. The placeholder
was never evaluated and all its files were deleted. It cost 2 harness passes on the unedited
checkpoint, whose only output I read was the list of returned keys.

Deviations from the plan made here are in the PLAN change log (section 12).

## 2026-11-12 (evening): Exp 2 and Exp 3 procedure, fixed before the injection

Session resumed by the harness builder (Claude Fable 5.2, main session). `make check` green
(218 passed) at `9adb19e`.

Per editor, in this order, with nothing else touching the repository in between:

1. `make_workspace.py` builds the workspace (budget written as the ledger's first record).
2. `tripwire.py arm <profile>`.
3. One fresh general-purpose subagent on Fable 5.2 receives the committed launch prompt with
   only the workspace path filled in. No other text is added.
4. `tripwire.py check`, then `pipeline.py collect`, `rebuild`, `controls`.
5. The workspace is deleted, then the next one is built.

Exp 2 runs first and completely (the Exp 3 workspaces hold the healthy checkpoint, which no
Exp 2 editor may see). While an editor runs I do not run git, lint, tests or any read of the
watched files, since that would trip the wires myself.

## 2026-11-12: Exp 2, tripwire false alarm after the first editor (audited)

After the `ckpt_1` editor finished, `tripwire.py check` reported 6 of 33 watched files as
read: `PLAN.md`, `edits/identity.py`, `inject_faults.py`, this journal, `faults.py` and
`tests/test_wsp_faults_controls.py`. The sealed specs, the other damaged checkpoints and every
copy of the healthy checkpoint were **not** tripped. PLAN section 9 requires an audit of any
tripwire anomaly; this is it.

- The six files are exactly the watched files that git tracks. Arming changes a file's
  ctime, so the next `git status` re-reads every armed tracked file to re-hash it. Claude
  Code runs `git status` on its own when an agent starts. Reproduced: arm, then one
  `git status`, then check, trips the same six files and no others.
- The editor's transcript lists 11 tool calls. Every one starts with `cd` into its workspace
  and touches only workspace files (listed by a script that prints tool names and command
  heads only). It never named the repository, the injector or the specs.
- Fix: `git config core.trustctime false` in this clone (local setting, not committed). With
  it, arm, `git status`, check reports nothing. Applied from `ckpt_2` on.
- From `ckpt_2` on I also list every editor's tool calls from its transcript after it ends,
  as a second check that does not depend on access times.

Conclusion: a harness artefact, not a read by the editor. The `ckpt_1` attempt stands. The
access-time record for `ckpt_1` is therefore uninformative for those six files; the
transcript is the evidence for them.

## 2026-11-12: Exp 2, `ckpt_2` editor: interruption, one tripped wire, one scan flag (audited)

- **Interruption.** The editor's run was cut off by an account usage limit after it had
  written its diagnosis, repair script and journal (4 passes used). After the limit was reset
  the same agent was resumed with one message that said only that the interruption was an API
  limit and that it should continue under the same rules. It appended a note to its journal,
  re-ran `repair.py` and reported. No pass was spent after the resume.
- **Tripwire.** One file tripped: this journal, read at 19:53:48, the second it was armed
  and before the editor's first tool call. Cause: I had appended to it and committed within
  the same second, which leaves git unsure whether the file changed ("racily clean"), so
  the next `git status` re-reads it. The editor's 18 tool calls all ran inside its
  workspace. From `ckpt_3` on the index is refreshed before arming and the wires are checked
  once right after a `git status` of my own, before the editor starts.
- **Static scan flag.** `repair.py` uses `torch.linalg.pinv`, which the rule-2 scan forbids
  by name. Audit: the pseudo-inverse is taken of head 6's own value rows, a weight matrix,
  and the repair is `0.5 * (P1 V1 + P7 V7) pinv(V6)`: a function of existing weights and one
  stated scalar, which is what interpretation I1 allows. No activation or text enters; the
  clean rebuild, which has no data in its folder, reproduces the same sha256. The scan
  pattern was a proxy for "least squares on data" and is over-broad here. I count the repair
  and disclose the flag; whether to count it is put to Ahmad in the final summary, and the
  report gives H2 both ways.

## 2026-11-12: Exp 2, `ckpt_3` editor (collected)

8 of 20 passes, tripwires clean (142 files), all 14 tool calls inside the workspace, clean
rebuild matches. Static scan flag: `analysis/03_gram.py` calls `lstsq`, to fit a line
through two spectra computed from weight matrices (a description of the weights, labelled
as such in the script). No text or activation enters and `repair.py` does not use it.
Same judgement as the `pinv` case of `ckpt_2`: within I1, flagged and disclosed.

## 2026-11-12: Exp 2, `ckpt_4` editor (collected)

4 of 20 passes, tripwires clean (200 files), every tool call inside the workspace, clean
rebuild matches, static scan clean.

## 2026-11-12: Exp 2, `ckpt_5` editor (collected), and Exp 2 closed for editing

5 of 20 passes, tripwires clean (265 files), every tool call inside the workspace, clean
rebuild matches, static scan clean. The last workspace is deleted.

| Checkpoint | Passes | Tensor named by the editor | Tripwires | Scan |
|---|---|---|---|---|
| ckpt_1 | 4 | `blocks.5.ln2.weight` | false alarm (git), audited | clean |
| ckpt_2 | 4 | `blocks.5.attn.proj.weight` (head 6) | one false alarm (git), audited | `pinv` on weights, audited |
| ckpt_3 | 8 | `blocks.3.attn.proj.weight` | clean | `lstsq` on weights, audited |
| ckpt_4 | 4 | `blocks.2.mlp.proj.weight` | clean | clean |
| ckpt_5 | 5 | `blocks.5.attn.proj.weight` (head 5) | clean | clean |

I have not opened the sealed specs. Whether these names are right, and what the repairs
recover on sealed stories, is decided by `final_eval.py` after the freeze. The editors' own
dev numbers (in their journals) are on 64-row samples and are not results of this study.
No Exp 2 edit is changed after this point.

## 2026-11-12: Exp 3, attempt `b10` (collected)

10 of 10 passes, none refused, tripwires clean (328 files), clean rebuild matches the
workspace checkpoint. Declared final: `copyheads_075` (heads 4.5, 4.4, 5.7, 5.6 scaled by
0.75; 2 tensors, 32,768 weights, relative L2 change 0.019).

My audit of the ledger against rule 3 (no search):

- Weights run: unedited (4 passes), the four heads zeroed (2, a causal probe), factor 0.5
  (1), factor 0.75 (3). So one location at three strengths. It is not a sweep in the sense
  of the plan: the zeroing was declared a probe of "do these heads matter" before it ran,
  0.5 was the first edit, and 0.75 is one revision with a written reason (0.5 cost
  +0.0225 nats on 64 dev stories, over the tolerance; the step was halved, not solved for).
  One of three revisions used. The factor was however chosen after seeing a dev loss, which
  the editor states itself. I accept it as a revision and report it as such.
- The four heads were chosen from activations captured in budgeted passes, which the rules
  allow (passes test which component matters). The numbers written into the weights are the
  existing weights times 0.75.
- Static scan flag: `analysis/07_copy_mode_split.py` uses `pinv` on a head's output matrix
  (weights) to express captured head outputs in the head's own basis. Analysis only; the
  editor rejected the idea it was testing and no edit uses it. Listed with the audited cases.
- The transcript's tool calls outside the workspace are reads of the agent's own tool-output
  files and one `free`/`df`/`nproc`. Nothing of the repository.
- Unused script `edits/final_gain_090.py` was never run through the model and is not final.
