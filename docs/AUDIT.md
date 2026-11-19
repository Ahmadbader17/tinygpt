# Simplification audit (2026-11-19, branch `simplify`)

Written before any code change. The baseline numbers are at the bottom.

## Overall
The core library (`src/tinygpt` without `wsp/`, about 2,000 lines) is already lean: typed, tested (120 tests),
one config object per concern, and comments that say why. Most of its weight is deliberate: the naive attention
path is the reference for the fused path, and the `# LEARN:` markers belong to the rewrite plan. This audit found
a few real bugs, some duplication and stale documentation. It found no structural rot.

Most of the repo by volume is the Weight Space Programming study (342 of 425 tracked files, 5,900 lines of
experiment scripts plus 1,900 lines in `src/tinygpt/wsp`). Nothing in the core imports it.

## Bugs
1. **Divergence overwrites the last good checkpoint** (`training/trainer.py` `fit`). When validation loss is NaN,
   `_after_eval` raises, and the `finally` block then saves the diverged weights as `checkpoints/last`, so
   `--resume` restarts from a broken model. Reproduced: NaN eval at step 20 leaves `last` at step 20 instead of
   step 10.
2. **tok/s is under-reported on the step after each eval** (`fit`). The throughput window is reset before
   checkpoint saving and sample generation, so their time is charged to the next window. In
   `runs/overnight_long/metrics.csv` step 260 reads 2,157 tok/s against about 2,430 for its neighbours. This affects
   logging only, not training.
3. **pre-commit runs ruff 0.6.9, `make check` runs ruff 0.17.1.** The two formatters can disagree, so a commit hook
   could rewrite code that `make check` then flags.

## Checked and correct
Next-token targets (`y` is `x` shifted by one, last window ends exactly at EOF); causal mask (naive `tril` and
`is_causal=True`, both tested); weight tying (shared tensor, counted once, saved once by `save_model`); dropout off
in eval and in sampling; loss averaging over gradient accumulation; tokenizer trained on the training split only;
**no train/val overlap** (0 of 5,000 validation stories appear verbatim among the 299,987 training stories);
checkpoint save/load (atomic swap, `weights_only=True`, exact resume tested); seeding (validation and sampling use
their own generators, so they do not perturb training randomness).

## Dead code, duplication, over-engineering
- `SlidingWindowDataset`: no caller outside one test; its docstring wrongly says evaluation uses it.
- Sampling defaults (120 tokens, 0.8, top-k 40) are written in three places: `SampleConfig`, `CompletionRequest`
  and the pydantic `CompletionIn`. `CompletionRequest` is `SampleConfig` plus a prompt.
- `finish_reason` is "stop" in the service and translated to "eot" in the route.
- `bench.py` calls `replace(cfg)` with no changes, which is a no-op copy.
- `load_config` repeats the seed default (1337) already set on `Config`; `run_export` hard-codes 200 x 32 eval batches.
- `utils/` is a package of four 10 to 25 line modules that could be one file.

## Documentation
- README has no install, train or reproduce steps for the model itself (only Hub generation and Docker).
- `docs/RESULTS.md` still says Space latency "will be added after Phase 8".
- `docs/LEARNING.md` and `DESIGN.md` mention `SlidingWindowDataset`.

## Risky spots (not bugs)
- Unknown config keys are rejected, and `generate --run` / `export` re-read `runs/<run>/config.toml`. So
  **removing any config field would break loading existing runs**. Config fields are therefore left alone.
- `generate()` stops a batch only when every row has emitted `<|endoftext|>`, and finished rows keep sampling.
  All callers use batch size 1, so this has no effect today.
- `.gitignore` does not cover `*.pt` (optimiser state) outside `runs/`.
- Tag `wsp-freeze` exists locally only; it is not on `origin`.

## Left alone on purpose
- Naive attention path: the readable reference that the fused path is tested against, and rewrite item 3.
- tiktoken adapter, `train-tokenizer`, `notebooks/solutions/03_solutions.py`: kept at Ahmad's request.
- Mixed precision: training is CPU-first. bf16 autocast on CPU changes numerics and is often slower on this chip,
  so it would break the "same results" rule.
- Dependency lower bounds in `pyproject.toml`: `uv.lock` pins exact versions; lower bounds are the norm for the
  manifest of a locked app.

## Baseline (before changes)
Command: `tinygpt train --config configs/small.toml --max-steps 100` (seed 1337, 12 threads, CPU), run twice.
With 200 warmup steps, all 100 steps are in warmup; this is a regression check, not a quality run.

| Step | 10 | 30 | 50 | 70 | 90 | 100 |
|---|---|---|---|---|---|---|
| train loss | 8.0998 | 7.2643 | 6.2917 | 5.3740 | 4.8215 | 4.6268 |

- Final val loss (20 x 32 x 256 tokens): **4.502402**, identical in both runs.
- Run-to-run noise: train loss differs in the 7th significant digit (multithreaded float reductions).
- Throughput: about 2,500 tok/s (per-window range 2,115 to 2,746). Wall time about 6:00 for 100 steps plus eval and samples.
- Peak RSS: 3.46 GB.
- Real checkpoints (`runs/overnight_long` best/last, `export/`): same state-dict SHA256 `5a7dc51d…`, recomputed
  val loss 1.708412 (equal to `best_val` in `state.json`), and greedy completions recorded for three prompts.

## After (end of the pass)
Same seeded 100-step command on the simplified code:

| | Before | After |
|---|---|---|
| Final val loss | 4.502402 | 4.502402 |
| Train loss at step 100 | 4.6267878 | 4.6267878 (differs in the 8th digit, same as run-to-run noise) |
| Samples at step 100 | | byte-identical |
| Mean tok/s (10 windows) | 2,517 | 2,545 |
| Peak RSS | 3.46 GB | 3.42 GB |
| Real checkpoints fingerprint | | identical (weights hash, logits, val loss, greedy and seeded outputs) |
| Tracked files | 425 | 79 (WSP study moved to its branch) |
| Core `src` lines (without `wsp/`) | 1,991 | 1,973 |
| Tests | 120 core (+198 WSP) | 119 (2 regression tests added, 4 cases for removed code dropped) |
| Runtime dependencies | 9 | 9 |
