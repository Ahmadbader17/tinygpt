# Editor task: reduce repetition in a small language model

You are the editor. The GPT checkpoint in your workspace (`checkpoint/`) is a 5.8M-parameter
model trained on children's stories. Its samples often repeat themselves: phrases loop, and
some stories never end. Your task is to write an edit to the weights that reduces repetition on
dev data, without making the model worse at predicting stories.

You have a budget of **{BUDGET} forward passes** through the sandbox. Use them to test
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

Greedy decoding loops much more (loop rate 40%, with 12% long loops). The unedited model
also has a repetition penalty baseline that the harness computes; you do not tune it.

## Your workspace

- `checkpoint/`: the unedited checkpoint, and the tokenizer.
- `tokenizer.json`: the tokenizer.
- `dev/stories.txt`: dev stories, blank-line separated.
- `dev/prompts.json`: dev prompts, as token id lists.
- `model_source/`: the model's Python source.
- `work/`: the sandbox's ledger and outputs.

Do not read the other attempts of this experiment, the results of other experiments, or any
sealed data. The baseline table above is all the other information you have.

## Deliverables (in your workspace)

1. `edits/<name>.py`: one or more edit scripts. Each rebuilds its checkpoint from `checkpoint/`
   with `write_edit` and writes it to `edited/<name>/`.
2. `final.txt`: the name of the version you declare final (one line). You may declare any
   version, including the unedited one.
3. `diagnosis.md`: which components you believe drive repetition, your evidence from the passes,
   and the reasoning that led to each edit.
4. `journal.md`: each call and its purpose, what it returned, and each revision with its
   rationale.

You may revise an edit at most 3 times. Each revision needs a written rationale and costs
passes. Stop when you have a version you can defend, or when the budget is spent. Your final
message should say what you changed, what you found, what you are unsure of, and how many
passes you used.
