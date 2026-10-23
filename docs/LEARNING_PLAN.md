# Learning plan: rewrite the core yourself

The working code is the answer key. For each exercise: read first, delete the listed bodies (keep signatures and the `# LEARN` header), rewrite them,
and run the listed tests until green. The tests are the oracle; they were written to be read. Do them in order, because each builds on the one before.
Time estimates are honest guesses for someone comfortable in Python and PyTorch basics; the first attempt at attention usually takes longest.

## 1. Data windows and memmap sampler
- **Rewrite:** `src/tinygpt/data/dataset.py`: `MemmapTokenSampler.__init__`, `__len__`, `batch`.
- **Read first:** `docs/LEARNING.md` Phase 2; your notebook's `GPTDatasetV1`; `np.memmap` docs.
- **Tests:** `uv run pytest tests/test_dataset.py` (target is the input shifted by one; shapes `(B, T)`; dtype int64; start positions never run off the end).
- **Time:** 1 to 2 hours.
- **Explain it back:** (1) Why is the last valid start `len(data) - T` and not `len(data) - T - 1`? (2) What does a memmap give you over a list of tensors, and what does the OS do when you slice it?

## 2. BPE training algorithm in pure Python
- **Write (new file):** `src/tinygpt/tokenizer/my_bpe.py` with `train_merges(text, num_merges)`, `encode(text, merges)`, `decode(ids, merges)`. Merge `i` creates token `256 + i`. Do not touch the existing tokenizer.
- **Read first:** `docs/LEARNING.md` Phase 1 and the notebook's tokenizer section. The notebook only used a pretrained BPE, so this is the gap in your understanding: how merges are *learned*.
- **Tests:** `uv run pytest tests/test_my_bpe.py` (skipped until your file exists): the first merge is the most frequent pair (`"aaabdaaabac"` gives `(97, 97)`), lossless round trips including unicode and emoji, merges shorten the sequence, training is deterministic, ids stay below `256 + num_merges`.
- **Time:** 3 to 5 hours. Making training fast enough for the fixture text (count pairs, merge in place) is the real work; speed on a full corpus is not the goal.
- **Explain it back:** (1) Why does decoding need only the merge list, not the text it was trained on? (2) Why must tie-breaking be deterministic, and what happens to a trained model if two runs learn merges in different orders?

## 3. Causal self-attention, naive path
- **Rewrite:** `src/tinygpt/model/attention.py`: `CausalSelfAttention.__init__` and the `naive` branch of `forward` (keep the `sdpa` branch as your reference).
- **Read first:** `docs/LEARNING.md` Phase 3 (every tensor shape is written out); DESIGN D3.
- **Tests:** `uv run pytest tests/test_attention.py` (naive equals sdpa to 1e-5; changing future tokens never changes earlier outputs; shape in equals shape out).
- **Time:** 2 to 3 hours.
- **Explain it back:** (1) Draw the `(T, T)` score matrix for T = 4 and mark which entries the mask sets to `-inf` and why. (2) Why divide by `sqrt(head_dim)`, and what would the softmax look like without it?

## 4. Block, MLP, GPT forward, init, weight tying
- **Rewrite:** `src/tinygpt/model/layers.py` (`MLP`, `Block`) and `src/tinygpt/model/gpt.py` (`GPT.__init__`, `_init_weights`, `forward`, `num_parameters`).
- **Read first:** `docs/LEARNING.md` Phase 3; DESIGN D3 (init and pre-norm); the notebook's embedding layers.
- **Tests:** `uv run pytest tests/test_model.py` (logits shape; parameter count equals the hand formula; tied head is the same tensor; initial loss within 0.3 of `ln(vocab)`; no leakage from the future; gradients reach every parameter; save and load give identical logits).
- **Time:** 3 to 4 hours.
- **Explain it back:** (1) Why is the untrained loss `ln(V)`? What would 12 or 2 tell you? (2) What does weight tying save in this model, and why can a tied head and embedding share one matrix?

## 5. One training step and the LR schedule
- **Rewrite:** `src/tinygpt/training/schedule.py`: `lr_at`; `src/tinygpt/training/trainer.py`: `Trainer.train_step`.
- **Read first:** `docs/LEARNING.md` Phase 4; DESIGN D4.
- **Tests:** `uv run pytest tests/test_schedule.py tests/test_trainer.py` (warmup, peak, cosine midpoint and floor; overfit one fixed batch below loss 0.1; resume reproduces the next five losses; loss falls on a learnable pattern).
- **Time:** 2 to 3 hours.
- **Explain it back:** (1) Why divide the loss by `grad_accum_steps` before `backward()`? (2) Why does warmup help AdamW, and what do you expect to see in the loss curve if the peak learning rate is 10 times too high?

## 6. Sampler
- **Rewrite:** `src/tinygpt/inference/sampler.py`: `generate`.
- **Read first:** `docs/LEARNING.md` Phase 6 and the temperature table in `docs/RESULTS.md`.
- **Tests:** `uv run pytest tests/test_sampler.py tests/test_service.py` (temperature 0 equals argmax; same seed gives the same tokens; stops at `<|endoftext|>`; top-k 1 equals greedy; context longer than the window is cropped).
- **Time:** 1 to 2 hours.
- **Explain it back:** (1) Why does temperature 0.2 look almost like greedy while 2.5 produces noise? (2) Where does a KV cache save work in this loop, and what must it store?

## Understand, do not rewrite
- **FastAPI plumbing (`serving/`):** the app factory builds the app from injected parts, the lifespan loads the model in a background thread, routes only translate HTTP to `CompletionService`. *Interview question it answers:* "How do you keep a web framework out of your model code and still test the API quickly?"
- **Docker (`Dockerfile`):** pinned uv, `--frozen --no-dev --no-cache`, CPU-only torch from the lock file, non-root user, health check. *Question:* "How do you make the image you tested the image you ship, and keep it small?"
- **CI (`.github/workflows/ci.yml`):** a clean machine runs lint, format check, mypy and pytest on every push, which found a real bug (a `.gitignore` rule that hid the `data` package). *Question:* "What does CI catch that your laptop does not?"
- **Hub upload (`publish.py`):** a dry run by default, one atomic commit per repo, the token only from the environment. *Question:* "How do you publish safely and verify what was published?"
- **CLI (`cli.py`):** thin argparse commands that call library code. *Question:* "Why keep the CLI thin?"
- **Logging and metrics (`utils/logging.py`, `training/metrics.py`):** a log file and a CSV per run, so a night-long job can be diagnosed afterwards. *Question:* "How do you know a long job is healthy while you sleep?"

## Interview drill
Whiteboard each of these without notes, then check against `docs/LEARNING.md`:
1. Attention shapes from `(B, T, C)` through heads, scores `(B, H, T, T)`, back to `(B, T, C)`.
2. Why the initial loss is `ln(V)`, and what a different starting value would mean.
3. Why scale by `sqrt(d_head)`.
4. Pre-norm versus post-norm: where the LayerNorm sits and what that does to gradients.
5. Why weight tying saves parameters, and how many here (1,048,576).
6. How a KV cache changes generation cost, from growing with the context length to constant per token.
7. How this training would change with data-parallel training across machines: one replica per process, gradient all-reduce each step, a larger effective batch, a sampler that gives each rank different windows, and the learning rate scaled with the batch.
