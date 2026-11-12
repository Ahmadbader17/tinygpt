# Phase 1 journal: harness

**Date:** 2026-11-12. **Commit at start:** `ed7e750` (Phase 0 plan). **Branch:**
`exp/weight_space_programming`. **Editor passes spent:** 0. Sandbox passes: 2, both for the budget test (listed at the end).

## Approval

Ahmad approved Phase 1 with the recommended defaults for D1 to D5 (PLAN.md section 12). He chose
**Fable 5.2** as the editor model for Exp 2 and Exp 3, as the plan says.

## What was built

| Piece | File | Notes |
|---|---|---|
| Budgeted sandbox | `src/tinygpt/wsp/sandbox.py` | `read` and `generate` calls, refused past the limit, ledger written before each call runs |
| Hash-chained ledger | same | verified on open; any edited, deleted or reordered record raises `LedgerError` |
| Sealed-path guard | same | refuses `data/sealed/`, `runs/wsp/sealed/`, and any file whose name contains `GPT4-valid` |
| Attention capture | `src/tinygpt/wsp/activations.py` | explicit softmax; matches `GPT.forward` to 1e-5 (tested) |
| Edit writer and diff summary | `src/tinygpt/wsp/edits.py` | checks names, shapes, dtype, finiteness; reloads the saved file and compares bit for bit |
| Story and sequence losses | `src/tinygpt/wsp/scoring.py` | `sequence_nll` added for the editor's `read` call |
| Pure evaluation and success rule | `src/tinygpt/wsp/evaluate.py` | paired intervals; the four preregistered criteria; specificity against controls |
| Final evaluation | `experiments/wsp/final_eval.py` | the only sealed reader; refuses unless clean, at `wsp-freeze`, run once, manifest hashes match; logs access first |
| Identity edit | `experiments/wsp/edits/identity.py` | the pipeline's null edit |
| Static rule test | `tests/test_wsp_edit_rules.py` | scans `experiments/wsp/edits/` and `analysis/` for optimizers, gradients, least squares, sealed names |
| Editor prompts | `experiments/wsp/prompts/` | shared rules and sandbox API; Exp 2 (blind, no fault names) and Exp 3 (with the budget placeholder) |
| Sanity script | `experiments/wsp/phase1_sanity.py` | run on the real checkpoint with dev data only |

New tests: 36 (sandbox, edits, static rules, evaluation, final-script computation). Total: 177
passing, 1 skipped (the tiktoken test needs network). `make check` is green.

## Checks and results

From `experiments/wsp/logs/phase1/sanity.json` (all passed):

- Identity edit rebuilds the original checkpoint **byte for byte** (both sha256
  `6d961505...cb86ec`). Zero weights changed.
- Dev loss of the identity edit equals the original's exactly (max difference 0.0 nats per story).
- Story loss and sampled generations are repeated exactly on a second run.
- Budget: with a limit of 2, the third call is refused. The ledger records it as `refused`,
  and it is not counted. A new process reads the same count.
- The ledger verifies after reopening.
- The sandbox's `read` matches the direct computation exactly.
- The static rule test passes on all ruled files.

Unit tests cover the cases the sanity run cannot: tampering with a record, deleting a record,
refusing sealed paths, refusing a checkpoint whose hash differs from the manifest, dropping
sealed stories that duplicate train or dev, rejecting NaN and wrong shapes, and the success
rule's failure modes (including that an identical model can never pass).

## Sealed-data check

- `data/sealed/` and `runs/wsp/sealed/` do not exist.
- No file whose name contains `GPT4-valid` exists anywhere on this machine (`find /` over the
  root file system).
- `~/.cache/huggingface` is 23 MB: it holds the 23 MB model download from Phase 0, not the
  dataset.
- The sealed file has not been downloaded. Nothing in Phase 1 reads it.

## Decisions taken while building (for the record)

- The sandbox's default tokenizer path points at `data/tokenizer/`, not at the original
  checkpoint folder. In Exp 2 the editor must not reach `runs/wsp/original/`. The two tokenizer
  files are byte-identical (sha256 `f229f2e9...`).
- Attention patterns are returned as a file, not inline, to keep call output small.
- `generate` writes full token ids and text to `outputs/`. Editors can read dev text; that is
  intended.
- The sealed-data guard checks paths, not reads. The editor is also blinded by workspace
  layout and instructions (PLAN section 5, rule 5). Access-time tripwires are still to be
  added for Exp 2, before the first editor starts.

## Open items before any editor starts (Phase 2)

1. **Repetition-penalty loss.** PLAN section 7 requires the penalised decoder's test loss,
   computed by applying the penalty to next-token distributions on real stories. It is not
   implemented yet. It belongs in `final_eval.py` before the freeze.
2. **Exp 2 workspace builder and fault injector.** Not written. The injector's seed and specs
   go to a sealed location that `guard_path` refuses.
3. **Tripwires.** Access-time checks on the clean checkpoint, the injector, and the specs, armed
   before each Exp 2 editor starts. Verified in Phase 0 that access times update on this
   filesystem.
4. **Manifest.** `experiments/wsp/manifest.json` is written at freeze time, not before.
5. **Editor identity.** Fable 5.2 is chosen. Fresh subagents must be launched on that model.
   The current session runs on Haiku 5.5, so the editor agents must be configured explicitly.
6. **Activations in the Exp 2 editor's workspace.** The editor can capture attention through
   `read`, but it can read the output only from `outputs/`. Confirm that is enough for the
   diagnosis work.

## Harness passes spent in Phase 1

- Sandbox passes (counted in the ledger): 2. Both are `read` calls on two dev prompts, made to
  test the budget. The third call was refused and is logged as `refused`, not counted.
- Direct harness computations (not through the sandbox, not counted): the identity comparison,
  the repeated dev-loss and generation runs for the determinism checks, and the direct
  `sequence_nll` used to compare with the sandbox. All on dev data.
- Editor passes: 0.
