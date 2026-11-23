# Learning plan: independent rewrite of the core

The working code serves as the answer key. For each exercise, the reader studies the listed material, deletes the listed bodies (keeping signatures and the `# LEARN` header), rewrites them,
and runs the listed tests until they pass. The tests are the oracle and are written to be read. The exercises build on one another and are taken in order.
Time estimates assume familiarity with Python and PyTorch basics; the first attempt at attention typically takes longest.

## 1. Data windows and memmap sampler
- **Rewrite:** `src/tinygpt/data/dataset.py`: `MemmapTokenSampler.__init__`, `__len__`, `batch`.
- **Read first:** `docs/LEARNING.md` Phase 2; the notebook's `GPTDatasetV1`; the `np.memmap` documentation.
- **Tests:** `uv run pytest tests/test_dataset.py` (target is the input shifted by one; shapes `(B, T)`; dtype int64; start positions never exceed the end of the data).
- **Time:** 1 to 2 hours.
- **Explain it back:** (1) Why is the last valid start `len(data) - T` and not `len(data) - T - 1`? (2) What does a memmap provide over a list of tensors, and what does the OS do when it is sliced?

## 2. BPE training algorithm in pure Python
- **Write (new file):** `src/tinygpt/tokenizer/my_bpe.py` with `train_merges(text, num_merges)`, `encode(text, merges)` and `decode(ids, merges)`. Merge `i` creates token `256 + i`. The existing tokenizer is left untouched.
- **Read first:** `docs/LEARNING.md` Phase 1 and the notebook's tokenizer section. The notebook used only a pretrained BPE, so this exercise covers how merges are *learned*.
- **Tests:** `uv run pytest tests/test_my_bpe.py` (skipped until the file exists): the first merge is the most frequent pair (`"aaabdaaabac"` gives `(97, 97)`), lossless round trips including unicode and emoji, merges shorten the sequence, training is deterministic, and ids stay below `256 + num_merges`.
- **Time:** 3 to 5 hours. The main work is making training fast enough for the fixture text (count pairs, merge in place); speed on a full corpus is not the goal.
- **Explain it back:** (1) Why does decoding need only the merge list and not the training text? (2) Why must tie-breaking be deterministic, and what happens to a trained model if two runs learn merges in different orders?

## 3. Causal self-attention, naive path
- **Rewrite:** `src/tinygpt/model/attention.py`: `CausalSelfAttention.__init__` and the `naive` branch of `forward` (the `sdpa` branch is kept as the reference).
- **Read first:** `docs/LEARNING.md` Phase 3 (every tensor shape is written out); DESIGN D3.
- **Tests:** `uv run pytest tests/test_attention.py` (naive equals sdpa to 1e-5; changing future tokens never changes earlier outputs; output shape equals input shape).
- **Time:** 2 to 3 hours.
- **Explain it back:** (1) Draw the `(T, T)` score matrix for T = 4 and mark which entries the mask sets to `-inf`, and why. (2) Why divide by `sqrt(head_dim)`, and how would the softmax behave without it?

## 4. Block, MLP, GPT forward, initialisation, weight tying
- **Rewrite:** `src/tinygpt/model/layers.py` (`MLP`, `Block`) and `src/tinygpt/model/gpt.py` (`GPT.__init__`, `_init_weights`, `forward`, `num_parameters`).
- **Read first:** `docs/LEARNING.md` Phase 3; DESIGN D3 (initialisation and pre-norm); the notebook's embedding layers.
- **Tests:** `uv run pytest tests/test_model.py` (logits shape; parameter count equals the hand formula; tied head is the same tensor; initial loss within 0.3 of `ln(vocab)`; no leakage from the future; gradients reach every parameter; save and load give identical logits).
- **Time:** 3 to 4 hours.
- **Explain it back:** (1) Why is the untrained loss `ln(V)`, and what would a value of 12 or 2 indicate? (2) What does weight tying save in this model, and why can a tied head and embedding share one matrix?

## 5. One training step and the LR schedule
- **Rewrite:** `src/tinygpt/training/schedule.py`: `lr_at`; `src/tinygpt/training/trainer.py`: `Trainer.train_step`.
- **Read first:** `docs/LEARNING.md` Phase 4; DESIGN D4.
- **Tests:** `uv run pytest tests/test_schedule.py tests/test_trainer.py` (warmup, peak, cosine midpoint and floor; overfit one fixed batch below loss 0.1; resume reproduces the next five losses; loss falls on a learnable pattern).
- **Time:** 2 to 3 hours.
- **Explain it back:** (1) Why divide the loss by `grad_accum_steps` before `backward()`? (2) Why does warmup help AdamW, and what would the loss curve show if the peak learning rate were 10 times too high?

## 6. Sampler
- **Rewrite:** `src/tinygpt/inference/sampler.py`: `generate`.
- **Read first:** `docs/LEARNING.md` Phase 6 and the temperature table in `docs/RESULTS.md`.
- **Tests:** `uv run pytest tests/test_sampler.py tests/test_service.py` (temperature 0 equals argmax; same seed gives the same tokens; stops at `<|endoftext|>`; top-k 1 equals greedy; context longer than the window is cropped).
- **Time:** 1 to 2 hours.
- **Explain it back:** (1) Why does temperature 0.2 look almost like greedy while 2.5 produces noise? (2) Where does a KV cache save work in this loop, and what must it store?

## Understand, do not rewrite
- **FastAPI plumbing (`serving/`):** the app factory builds the app from injected parts, the lifespan loads the model in a background thread, and routes only translate HTTP to `CompletionService`. *Interview question:* "How is a web framework kept out of model code while keeping the API quick to test?"
- **Docker (`Dockerfile`):** pinned uv, `--frozen --no-dev --no-cache`, CPU-only torch from the lock file, non-root user and a health check. *Question:* "How is the image that was tested the image that ships, and how is it kept small?"
- **CI (`.github/workflows/ci.yml`):** a clean machine runs lint, format check, mypy and pytest on every push. It found a real bug (a `.gitignore` rule that hid the `data` package). *Question:* "What does CI catch that a developer's machine does not?"
- **Hub upload (`publish.py`):** a dry run by default, one atomic commit per repository, and the token read only from the environment. *Question:* "How is publishing made safe and verifiable?"
- **CLI (`cli.py`):** thin argparse commands that call library code. *Question:* "Why keep the CLI thin?"
- **Logging and metrics (`utils/logging.py`, `training/metrics.py`):** a log file and a CSV per run, so a night-long job can be diagnosed afterwards. *Question:* "How is the health of a long job known while unattended?"

## Interview drill
Each item is to be whiteboarded without notes, then checked against `docs/LEARNING.md`:
1. Attention shapes from `(B, T, C)` through heads, scores `(B, H, T, T)`, and back to `(B, T, C)`.
2. Why the initial loss is `ln(V)`, and what a different starting value would indicate.
3. Why scores are scaled by `sqrt(d_head)`.
4. Pre-norm versus post-norm: where the LayerNorm sits and its effect on gradients.
5. Why weight tying saves parameters, and how many it saves here (1,048,576).
6. How a KV cache changes generation cost, from growing with context length to constant per token.
7. How this training would change under data-parallel training across machines: one replica per process, gradient all-reduce each step, a larger effective batch, a sampler that gives each rank different windows, and a learning rate scaled with the batch.
