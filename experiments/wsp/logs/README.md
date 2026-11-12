# Study journal and logs

Released as part of the study, following the approach of the original paper: every phase
has a journal, and every run that the report uses leaves its inputs, outputs and call ledger
here.

| Path | What it holds |
|---|---|
| `phase1/journal.md` | Phase 1 (harness): what was built, what was checked, what was run, and open items |
| `phase1/sanity.json` | Output of `experiments/wsp/phase1_sanity.py` |
| `sealed_access.jsonl` | One line per access to the sealed test data (written by `final_eval.py` before reading) |
| `exp1/`, `exp2/`, `exp3/` | Per-experiment journals, ledgers, analysis scripts and outputs (created in Phase 2) |

Ledgers (`ledger.jsonl`) are hash-chained records of every forward pass, including refused
calls. Editor and harness roles are separate.
