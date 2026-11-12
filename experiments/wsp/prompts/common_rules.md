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

## Sandbox (your only way to run the model)

```python
from pathlib import Path
from tinygpt.wsp.sandbox import Sandbox
from tinygpt.wsp.generation import DecodeConfig
from tinygpt.wsp.edits import load_base_params, write_edit, diff_summary

BUDGET = 0  # replace with the budget given in your task
ws = Path(".").resolve()  # your workspace root
sb = Sandbox(ws / "work", role="editor", limit=BUDGET, tokenizer_path=ws / "tokenizer.json")

# read: per-token loss of token sequences (each 2 to 256 tokens, at most 64 rows per call)
sb.read(checkpoint_dir, sequences, purpose="...", attention_layers=(0, 1))
# generate: continuations of prompts (token id lists), at most 64 rows per call
sb.generate(
    checkpoint_dir,
    prompts,
    seeds=(0, 1),
    decode=DecodeConfig(200, temperature=0.8, top_k=40),
    purpose="...",
)
sb.generate(
    checkpoint_dir, prompts, seeds=None, decode=DecodeConfig(200, temperature=0), purpose="..."
)
sb.used, sb.remaining  # passes spent and left
```

Calls return a dict with a `call` number, the checkpoint hash, and summary numbers. Full text
and attention arrays are written under `work/outputs/`. A call past the budget raises
`BudgetExceeded`, so do not retry it.

To make an edit, write a script that returns new values for the named parameters and calls
`write_edit(base_dir, new, out_dir, name, script_path)`. That writes a checkpoint and
`diff_summary.json` (tensors touched, weights changed, relative L2). Rebuild the edited
checkpoint from the base each time, using the script alone. A checkpoint without its script
does not count. Parameter names come from `load_base_params(base_dir)`. Nothing else in the repo
is needed to rebuild an edit.
