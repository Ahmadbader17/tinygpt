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
