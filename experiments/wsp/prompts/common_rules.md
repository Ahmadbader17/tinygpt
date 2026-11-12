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
`{PYTHON}`. The package `tinygpt` is already importable there. Its source for the pieces you
use is copied into `harness_source/` and `model_source/` for reference; you do not need to
read anything else on this machine, and you must not read the study's repository
(`{REPO}`), its `experiments/`, `runs/`, `export/`, `docs/` or `tests/` folders, or any other
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
sb = Sandbox.open(ws / "work")  # your budget of {BUDGET} passes is already recorded there

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
