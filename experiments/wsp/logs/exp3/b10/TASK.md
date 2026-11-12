# Editor task: reduce repetition in a small language model

You are the editor. The GPT checkpoint in your workspace (`checkpoint/`) is a 5.8M-parameter
model trained on children's stories (TinyStories). Its README lists as a limitation that the
output "repeats itself". Your task is to find out, from the weights and a few forward passes,
which components drive that repetition, and to write an edit to the weights that reduces it
without making the model worse at predicting stories.

You have a budget of **10 forward passes** through the sandbox. Use them to test
hypotheses about which components drive repetition. Do not use them to search.

## What counts as success (for your information)

The final evaluation runs once, on sealed data you never see. It uses these measures:

- **Loop rate:** share of generations in which some 4-token chunk occurs 3 or more times.
- **rep-4:** share of 4-token chunks that have already appeared earlier in the same story.
- **Test loss:** how well the model predicts held-out stories. It must not rise by more than
  0.02 nats.
- **Length:** mean story length must stay at least 90% of the unedited model's.

Baseline on dev data, sampled with temperature 0.8 and top-k 40, 200 tokens at most, 300
prompts, 5 seeds:

| Measure | Unedited model | Real stories, same prompts |
|---|---|---|
| Loop rate | 24% | 19% |
| rep-4 | 0.027 | 0.018 |
| Mean length (tokens) | 163 | 170 |
| Dev story loss (nats per token) | 1.605 | not applicable |

A "long loop" is an 8-token chunk occurring 3 or more times. Under sampling long loops are
rare (0.3% of generations, the same as in real stories). Greedy decoding loops much more: loop
rate 40%, long-loop rate 12%. The final evaluation also compares your edit with a decoding-time
repetition penalty and with random edits of the same size and location; you do not need to
produce either.

## Your workspace

- `checkpoint/`: the unedited checkpoint, and the tokenizer.
- `tokenizer.json`: the tokenizer.
- `dev/stories.txt`: dev stories, blank-line separated.
- `dev/prompts.json`: dev prompts, as token id lists.
- `model_source/`, `harness_source/`: the model's and the sandbox's Python source, for reference.
- `work/`: the sandbox's ledger and outputs.

Do not read the other attempts of this experiment, the results of other experiments, or any
sealed data. The baseline table above is all the other information you have.

## Deliverables (in your workspace)

1. `edits/<name>.py`: one or more edit scripts, run from the workspace root. Each rebuilds its
   checkpoint from `checkpoint/` with `write_edit` and writes it to `edited/<name>/`.
2. `final.txt`: the `<name>` of the version you declare final (one line), or the single word
   `unedited` if you conclude that no edit should be made.
3. `diagnosis.md`: which components you believe drive repetition, your evidence from the passes,
   and the reasoning that led to each edit.
4. `journal.md`: each call and its purpose, what it returned, and each revision with its
   rationale.

You may revise an edit at most 3 times. Each revision needs a written rationale and costs
passes. Stop when you have a version you can defend, or when the budget is spent. Your final
message should say what you changed, what you found, what you are unsure of, and how many
passes you used.

## Rules (binding)

1. No gradients. Do not call `backward`, `torch.optim`, or enable autograd. Do not train anything.
2. No fitting. Do not estimate new weight values from data: no least squares, regression, or
   values computed from activations on text. Weight values come from stated reasoning about
   what the weights compute, using scalars you choose (a factor, a rank, a direction you
   derived from the weights themselves). Analysis that uses only the weights (SVD, norms,
   cosine similarities, products of matrices) is free.
3. No search. Do not try many edit strengths or locations scored by evaluation, and do not
   run a loop that keeps the best of several candidates. Each edit comes from a written
   reason. You may revise an edit at most 3 times, and every revision needs a written rationale.
4. Forward passes are rationed. Every model run goes through the sandbox (below). It counts,
   logs, and refuses calls beyond the budget. Each call needs a `purpose` that states the
   hypothesis it tests. Write the purpose before you call.
5. You never read sealed data: nothing under any folder named `sealed`, and no file whose name
   contains `GPT4-valid`. Only the data provided in your workspace is available.
6. Do not touch any original or other checkpoint except where your task names it.

Interpretation I1 (binding): a number you write into a weight must not be computed from
activations or from model outputs on any text. Forward passes are for testing hypotheses about
which component matters, not for producing edit values.

## How to run things

Work only inside your workspace folder. Run Python from the workspace root with
`/home/baderowski/code/tiny-gpt/.venv/bin/python`. The package `tinygpt` is already importable there. Its source for the pieces you
use is copied into `harness_source/` and `model_source/` for reference; you do not need to
read anything else on this machine, and you must not read the study's repository
(`/home/baderowski/code/tiny-gpt`), its `experiments/`, `runs/`, `export/`, `docs/` or `tests/` folders, or any other
workspace next to yours.

The tokenizer: `from tinygpt.tokenizer import BPETokenizer`, then
`tok = BPETokenizer.load(Path("tokenizer.json"))`, `tok.encode(text)`, `tok.decode(ids)`,
`tok.eot_id` (the `<|endoftext|>` id, 0). Stories in training were separated by that token.

Save every analysis script you run as a file in your workspace (for example
`analysis/01_norms.py`) and keep its printed output in your journal. Scripts must use paths
relative to the workspace root, never absolute paths, so that they can be re-run elsewhere.
Do not construct or run the model yourself (no `GPT(...)`, no hand-written forward pass on
token sequences): the sandbox is the only way to run it.

## Sandbox (your only way to run the model)

```python
from pathlib import Path
from tinygpt.wsp.sandbox import Sandbox
from tinygpt.wsp.generation import DecodeConfig
from tinygpt.wsp.edits import load_base_params, write_edit, diff_summary

ws = Path(".")  # the workspace root; run your scripts from there
sb = Sandbox.open(ws / "work")  # your budget of 10 passes is already recorded there

# read: loss of token sequences (each 2 to 256 tokens, at most 64 rows per call).
# Optionally capture activations (equal-length rows needed); they are saved to a .npz file.
out = sb.read(
    ws / "checkpoint",
    sequences,
    purpose="the hypothesis this call tests",
    capture=("attn", "head_out", "resid_post", "token_loss"),
    layers=(0, 1),
)
# capture kinds, each taken at the layers you name:
#   attn (rows, heads, T, T)      head_out (rows, heads, T, 256)   attn_out, mlp_out (rows, T, 256)
#   mlp_hidden (rows, T, 1024)    resid_pre, resid_mid, resid_post (rows, T, 256)
# and, without a layer: "logits" (rows, T, 4096), "token_loss" (rows, T-1)
# out["nll"], out["tokens"], out["mean_nll"], out["capture_file"]

# generate: continuations of prompts (token id lists), at most 64 continuations per call
out = sb.generate(
    ws / "checkpoint",
    prompts,
    seeds=(0, 1),  # prompts x seeds rows; seeds=None with temperature=0 is greedy
    decode=DecodeConfig(200, temperature=0.8, top_k=40),
    purpose="...",
)
# out["loop_rate"], out["rep4"], out["mean_length"], out["eot_rate"], out["generations_file"]

sb.used, sb.remaining  # passes spent and left
```

One call runs one set of weights: comparing an edited checkpoint with the original on the same
inputs costs two passes. Every call, including refused ones, is recorded in `work/ledger.jsonl`
before it runs; do not edit or delete that file. A call past the budget raises
`BudgetExceeded`; do not retry it. Invalid arguments raise before a pass is spent.

## Making an edit

```python
params = load_base_params(ws / "checkpoint")  # dict: parameter name -> tensor (76 tensors)
new = {name: value.clone() for name, value in params.items()}
# ... change entries of `new` ...
write_edit(ws / "checkpoint", new, ws / "edited" / "my_edit", "my_edit", Path(__file__))
```

`write_edit` saves the checkpoint and `diff_summary.json` (tensors touched, weights changed,
relative L2 change). An edit counts only if its script rebuilds it from `checkpoint/` alone,
deterministically. In `blocks.L.attn.proj.weight` (256 x 256), head `h` owns input columns
`32*h` to `32*h + 31`. In `blocks.L.attn.qkv.weight` (768 x 256), rows 0-255 are queries,
256-511 keys, 512-767 values, and head `h` owns rows `32*h` to `32*h + 31` of each third.
