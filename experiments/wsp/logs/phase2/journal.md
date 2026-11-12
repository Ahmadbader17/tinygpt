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
