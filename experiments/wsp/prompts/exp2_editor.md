# Editor task: diagnose and repair a damaged language model

You are the editor. A small GPT checkpoint in your workspace (`checkpoint/`) has been damaged.
The architecture is in `model_source/`: a GPT with 6 pre-norm blocks, 8 attention heads per
block, d_model 256, context 256, tied token embeddings, and a vocabulary of 4096.

Your task: find what is wrong and repair it, so that the checkpoint's loss on dev data goes
back towards its healthy level. The healthy model of this architecture had validation loss
1.6877 on the official validation set. You may measure the damaged checkpoint's loss on the
dev data.

You have a budget of **20 forward passes** through the sandbox. Use them for hypotheses, not
for searching. Name the component you suspect, say why from the weights, test it, and decide.

You are not told what kind of damage happened. Work from the weights and what the sandbox
returns.

## Your workspace

- `checkpoint/`: the damaged checkpoint (`model.safetensors`, `model_config.json`,
  `tokenizer.json`).
- `tokenizer.json`: the tokenizer (byte-level BPE, vocabulary 4096).
- `dev/stories.txt`: dev stories, blank-line separated.
- `dev/prompts.json`: dev prompts, as token id lists.
- `model_source/`: the model's Python source.
- `work/`: the sandbox's ledger and outputs.

Do not read any other checkpoint or any copy of the healthy model, including the Hugging Face
Hub, any folder named `original`, `clean`, `healthy` or `reference`, and any folder outside
this workspace named after the study. Do not read anything whose name suggests a fault
specification or an injector. If you come across one by accident, stop reading it and say so
in your journal.

## Deliverables (in your workspace)

1. `diagnosis.json`: `{"tensor": "<parameter name>", "layer": <int or null>, "component":
   "<attn|mlp|ln|other>", "head": <int or null>, "fault": "<your best description>",
   "confidence": <0 to 1>, "evidence": "<what the passes showed>"}`. Name the tensor you
   believe is damaged, even if your repair is partial.
2. `repair.py`: a deterministic script that rebuilds the repaired checkpoint from `checkpoint/`
   and writes it to `repaired/` with `write_edit`. It must run from scratch with no other state.
3. `journal.md`: your reasoning, each call and its purpose, what it returned, and each
   revision with its rationale.

Stop when you have a repair you can defend, or when the budget is spent. Your final message
should say what you found, what you repaired, what you are unsure of, and how many passes you
used.

