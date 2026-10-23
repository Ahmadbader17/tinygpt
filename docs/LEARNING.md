# Learning log

## Phase 0: scaffold, config, environment

**Concept.** A `src/` layout puts the package in `src/tinygpt/`, so `import tinygpt` only works
once it is installed (`uv sync` does an editable install). Tests therefore exercise the installed
package, and a missing file in the packaging shows up immediately instead of hiding behind the
current directory. Configs are frozen dataclasses loaded from TOML: a run reads its settings once
and cannot change them mid-flight; unknown keys raise, so a typo like `drop_out` fails loudly.

**Shapes preview.** `ModelConfig` checks `d_model % n_head == 0` because attention later reshapes
`(B, T, C) -> (B, n_head, T, C // n_head)`; with `small`, `C=256`, `n_head=8`, head dim 32.

**Predict-first answer.** Tied embedding matrix = 4096 x 256 = 1,048,576 parameters, about 18% of
the ~5.8M total (blocks: ~12 x 256^2 x 6 = 4.7M; positions: 256 x 256 = 65k). With GPT-2's 50257
vocab it would be 12.9M, which is why we trained our own 4096 BPE.

**Link to your notebook.** The notebook has no config: vocab size and context length are loose
variables. Here they live in one place and every run saves its resolved config.

<details><summary>Check questions</summary>

1. *Why does `src/` layout catch packaging bugs?* Tests import the installed package, not the
   working directory.
2. *Why frozen dataclasses?* Immutability: no hidden mutation during a run, hashable, safe to share.
3. *Why must `d_model` divide by `n_head`?* Heads split the channel dimension evenly.
</details>

**Interview angle.** "I kept configuration declarative (TOML into frozen, validated dataclasses) so
every run is reproducible from one saved file." Mention strict typing and `make check` as the single
gate used locally and in CI.

## Phase 1: tokenizer (byte-level BPE)

**Concept.** Text must become integers. Your notebook split on words and built a dictionary, so
unseen words needed `<|unk|>`. Byte-level BPE starts from the 256 byte values and *learns* merges:
count adjacent pairs in the training text, fuse the most frequent into a new token, repeat until the
vocab reaches 4096 (256 bytes + 1 special + 3839 merges). Encoding replays the merges in the order
they were learned. Because every string is bytes, nothing is ever unknown.

**Shapes.** `encode: str -> list[int]` (length L, each id in `[0, vocab)`); later the pipeline
turns these into `(B, T)` int tensors. Decoding concatenates bytes and decodes UTF-8, which is why
a single id may print as `�`: it holds a partial multi-byte character.

**Worked example (tiny vocab 512, fixture stories).** `"the"` -> `[84, 257]` = `t` + `he`
(not one token: only 255 merges to spend and "the" is usually preceded by a space); `" the"` ->
one token `[261]`; `"Pip"` -> `Pi` + `p`; `"🦊"` -> 4 byte tokens (the emoji is 4 UTF-8 bytes
and never appeared in training, so no merges cover it). With vocab 4096 on real TinyStories, common
words become single tokens.

**Design choices.** `add_prefix_space=False` keeps `" the"` and `"the"` distinct, like GPT-2.
Train the tokenizer on the training split only. `Tokenizer` is a `Protocol`: model and trainer
depend on the interface, so the tiktoken adapter slots in (used in one comparison test).

<details><summary>Check questions</summary>

1. *Why can byte-level BPE never produce an unknown token?* The base alphabet is all 256 bytes,
   so any input decomposes into known symbols.
2. *Why does `" the"` get its own token while `"the"` may not?* Pre-tokenization keeps the leading
   space attached to the word, and `" the"` is far more frequent mid-sentence.
3. *What does a bigger vocab trade off?* Fewer tokens per text (cheaper attention) versus a larger
   embedding/output matrix (vocab x d_model) and sparser statistics per token.
</details>

**Interview angle.** "I trained a 4096-token byte-level BPE rather than reusing GPT-2's 50k vocab,
because at d_model=256 the embedding table alone would have been over twice the rest of the model.
Byte-level means no unknown tokens, and I verified lossless round trips on unicode and emoji."

## Phase 2: data pipeline (clean, encode, memmap)

**Concept.** Training should never touch text. We turn the corpus into numbers once, then training just reads
them. `make data` runs `tinygpt prepare-data`: stream stories, clean them, train BPE on the training split only,
encode each story plus `<|endoftext|>`, and write one flat `uint16` file per split (`train.bin`, `val.bin`) and a
`meta.json`.

**Shapes.** `train.bin` is a flat `(N,)` array of uint16 (N = 60,838,396, about 2 bytes per token). A training batch picks
`B` random starts `s` and slices `x = tokens[s : s+T]`, `y = tokens[s+1 : s+T+1]`, giving `x, y` of shape `(B, T)`,
cast to int64 for the embedding lookup. `y` is `x` shifted left by one, so at every position the target is the
next token. One window therefore gives `T` training examples at once.

**Versus the notebook.** The notebook's `GPTDatasetV1` builds a Python list holding a tensor for every window, so memory grows
as `num_windows x context_length`. The memmap maps the file and the OS pages in only what we touch: RAM stays tiny and
random starts replace `stride`. We keep `SlidingWindowDataset` for deterministic evaluation and tests.

**Design choices.** Validation is held out, and the tokenizer never sees it. Cleaning is light (whitespace, control characters) so
the model still sees natural text. `meta.json` stores the tokenizer SHA256 so a checkpoint can be checked against its tokenizer.
Measured: 202.8 tokens per story, 13 of 300,000 stories dropped as empty.

<details><summary>Check questions</summary>

1. *Why is uint16 enough, and what breaks past 65,535?* ids are below 4096, so 2 bytes hold them. A vocab of 65,536 or more would
   overflow, and `encode_split` raises instead of silently wrapping.
2. *Why are `x` and `y` offset by one token?* The model is trained to predict the next token at every position, so the target
   at position `i` is the input at `i+1`.
3. *Why train the tokenizer on the training split only?* If it saw validation text, merges would be tuned to it and validation
   loss would be slightly optimistic. Held-out data should be untouched.
</details>

**Interview angle.** "I tokenised 300k stories once into a 122 MB uint16 memmap (60.8M tokens), so batches are random slices with
almost no RAM, instead of materialising every window as a tensor. The tokenizer is fit on train only, and `meta.json` records its
hash so checkpoints can be matched to the right vocabulary."

## Phase 3: the model (attention, MLP, GPT)

**Concept.** A GPT turns token ids into next-token scores. Each token id becomes a vector, a position vector is added,
and the sum passes through a stack of identical blocks. Each block lets tokens look at earlier tokens (attention)
and then transforms each token on its own (MLP). A final LayerNorm and a linear head give one score per vocabulary entry.

**Shapes, small config (B=batch, T=256, C=256, H=8 heads, head_dim=32, V=4096).**
```
idx                (B, T)            int64 token ids
wte(idx) + wpe(pos) (B, T, C)        token vector + position vector
-- per block (x6) --
 ln1(x)            (B, T, C)
 qkv(x)            (B, T, 3C)        -> split into q, k, v, each (B, T, C)
 q, k, v           (B, H, T, 32)     view + transpose: heads become a batch dimension
 q @ k^T / sqrt(32)(B, H, T, T)      score of every query position against every key position
 mask + softmax    (B, H, T, T)      future positions get -inf, so weight 0; rows sum to 1
 att @ v           (B, H, T, 32)
 concat heads+proj (B, T, C)         added back onto x (residual)
 mlp(ln2(x))       (B, T, C)         C -> 4C -> C, added back onto x (residual)
-- end --
ln_f, lm_head      (B, T, V)         logits; loss = cross_entropy over B*T rows
```

**The causal mask.** `scores[t, s]` is how much position t cares about position s. Setting it to `-inf` for s > t makes the
softmax weight exactly 0, so token t cannot read the future. That is what lets one forward pass train all `T` next-token
predictions at once without cheating. Test: change a later token and earlier logits stay identical.

**Why divide by sqrt(head_dim).** A dot product of two 32-dim vectors with unit-variance entries has variance 32. Without scaling, the
scores get large, softmax saturates to near one-hot, and gradients vanish. Dividing by sqrt(32) restores variance ~1.

**Pre-norm.** `x = x + attn(ln1(x))` keeps an un-normalised identity path from the output back to the input, so gradients reach early
layers directly. Post-norm puts LayerNorm on that path and is harder to train without careful warmup.

**Weight tying.** `lm_head.weight is wte.weight`: one (4096 x 256) matrix reads tokens in and scores tokens out. It saves
1,048,576 parameters (18% of the model) and ties the two meanings together. Counted once: **5,853,184 parameters**
(4,739,072 without embeddings). The test recomputes this by hand from the design.

**Init and initial loss.** All weights ~ N(0, 0.02), so the logits are tiny and the softmax is almost uniform over V tokens. Cross-entropy
of a uniform guess is `ln(V)`: ln(4096) = 8.318, and the measured untrained loss was 8.371 (tiny: 6.243 vs ln(512) = 6.238).
If your first loss is far from `ln(V)`, something is wrong, which is why this is a test. The residual output projections get an extra `1/sqrt(2*n_layer)` so variance does not grow with depth.

**Versus the notebook.** The notebook stopped at embeddings. Here the embedding layer is the front of a model, and the same `wte` doubles as the output layer.

<details><summary>Check questions</summary>

1. *Why is the untrained loss ln(V)?* Near-zero logits give a uniform distribution, so every token has probability 1/V and
   the cross-entropy is `-ln(1/V) = ln V`.
2. *What would happen without the causal mask?* Position t could read token t+1, which is its own training target.
   Loss would drop fast in training, but the model would be useless at generation time, when the future does not exist.
3. *Why are the naive and sdpa attention paths both kept?* The naive one is readable and serves as a reference; sdpa is faster.
   A test checks they agree to 1e-5, so the optimised path is proven correct.
</details>

**Interview angle.** "I implemented GPT-2 style attention twice: an explicit softmax(QK^T/sqrt(d))V with a triangular mask, and the
fused PyTorch kernel, and a test asserts they agree and that the future never leaks into the past. I also check that the parameter count
equals a hand-derived formula and that the untrained loss is ln(vocab), which catches wiring and init bugs before any training."

## Phase 4: the training loop

**Concept.** Training is the same five lines repeated: get a batch, forward to a loss, backward to gradients, clip, update. Everything else
(schedule, evaluation, checkpoints) exists to make those lines safe to run for hours.

**Anatomy of one step (`Trainer.train_step`), small config, B=32, T=256.**
```
x, y = sampler.batch(32, rng)        (32, 256) int64 each; y is x shifted left by one
logits, loss = model(x, y)           logits (32, 256, 4096) -> loss: one scalar (mean over 8192 positions)
(loss / grad_accum).backward()       fills p.grad for all 5.85M parameters
clip_grad_norm_(params, 1.0)         if the global gradient norm exceeds 1, rescale it to 1
optimizer.step()                     AdamW update using the current learning rate
```
**Learning-rate schedule.** `lr_at(step)` rises linearly for 200 steps (Adam's moment estimates are noisy at the start, so big early steps
are risky), then follows a cosine from 1e-3 down to 1e-4 at the last step. It is a pure function, which is why it is easy to test.

**AdamW detail.** Weight decay 0.1 applies only to tensors with `ndim >= 2` (matrices and embeddings). Biases and LayerNorm gains are
excluded: shrinking them toward zero has no regularising benefit.

**Checkpoint and resume.** A checkpoint is the weights (safetensors), the optimiser state (Adam's two moment estimates per parameter, which is why the
checkpoint is ~3x the weights), the step, and both RNG states. The test trains 5 steps, saves, then a new trainer with a different init resumes
and produces the *same next 5 losses exactly*. Saves go to a temp directory and are swapped in, so a crash cannot leave half a checkpoint.
`SIGINT` and `SIGTERM` finish the current step, save `last`, and exit.

**Evaluation.** Every 250 steps the same validation windows (fixed seed) are scored, so the curve is comparable between points. `best`
is kept by validation loss; `last` is for resuming. `samples.md` repeats five fixed prompts with fixed seeds so you can watch gibberish turn into English.

**Versus the notebook.** The notebook had a DataLoader over a list of tensors and no notion of a run. Here a run is a directory (config, environment, git commit, metrics, plot, samples, checkpoints).

<details><summary>Check questions</summary>

1. *Why divide the loss by `grad_accum_steps`?* Gradients add across `backward()` calls, so dividing makes the accumulated gradient the mean over all micro batches, the same as one big batch.
2. *Why save optimiser state, not just weights?* Without Adam's moment estimates a resumed run restarts with cold statistics and a different trajectory; the resume test would fail.
3. *Why is the test "overfit one fixed batch" a good gradient check?* A model that cannot drive loss near zero on 64 tokens has a bug in the forward pass, the loss, the optimiser or the data path.
</details>

**Interview angle.** "I wrote the training loop with warmup plus cosine decay, global-norm clipping, decay only on matrices, and exact resume: a test checks that a resumed run
reproduces the next five losses bit for bit. Checkpoints are written atomically and the run saves on SIGTERM, so a night-long CPU run survives interruptions."

## Phase 5: benchmark, budget, baselines

**Compute budgeting.** Throughput decides what is possible. `tinygpt bench` times real training steps for several thread counts on this laptop (Core Ultra 9 185H, 22 threads):

| threads | 4 | 6 | 8 | 10 | 12 | 16 | 22 |
|---|---|---|---|---|---|---|---|
| tokens/s | 2.1k | 2.1k | 2.3k | 2.9k | 2.8k | 0.9k | 0.6k |

Using every thread is 4 to 5 times *slower* than 12. Likely cause: the chip mixes performance and efficiency cores, and PyTorch's parallel loops wait for the slowest
thread. So the config uses 12. The real run sustained ~2.5k tok/s (laptop thermals), so one pass over the 61M training tokens = 7,500 steps = about 6.8 hours.
Rule of thumb: ~20 tokens per parameter is "compute optimal" (Chinchilla); we get ~10 for one epoch, a fair trade for a laptop budget.

**Baselines.** Predict the validation tokens using only counts, no neural network:

| model | perplexity | loss (nats) |
|---|---|---|
| uniform guess | 4096 | 8.32 |
| unigram (token frequencies) | 357 | 5.88 |
| bigram (previous token) | 38.4 | 3.65 |

Perplexity = exp(loss) = "the model is as unsure as a fair die with this many sides". The trained GPT must beat the bigram table clearly; otherwise it has learned no more than which word follows which.

**Reading loss curves.** Train and val falling together = healthy. Val flat or rising while train falls = overfitting (not expected in one epoch). Sudden spikes = learning rate too high or bad batch (clipping helps).
Loss stuck near 7 to 8 = the model has only learned unigram frequencies.

<details><summary>Check questions</summary>

1. *Why can more threads be slower?* Synchronisation across unequal cores and memory-bandwidth limits; the slowest thread gates each parallel operation.
2. *Why compare to a bigram baseline?* It is the best you can do looking one token back, so beating it proves the model uses longer context.
3. *Why one epoch rather than several?* No data repeats, so val loss measures generalisation, and the budget fits the sleep window.
</details>

**Interview angle.** "I benchmarked thread counts before committing compute and found that using all 22 threads was 4x slower than 12 on a hybrid CPU, then sized the run from measured tokens per second.
I judged the model against unigram and bigram perplexity baselines I computed on the same tokenisation, not just against its own loss."

## Phase 6: sampling and the CompletionService

**Concept.** The model never outputs text. For a context of `T` tokens it outputs scores `(T, V)`; only the last row matters for generation.
Sampling turns that row into one token, appends it, and repeats. That loop is `generate()`; `CompletionService` wraps it with tokenising, seeding, stop handling and a concurrency cap.

**Shapes (one generation step, small config).**
```
ctx          (1, T)       last <= 256 tokens (older ones are cropped away)
logits       (1, T, 4096) -> keep the last position only -> (1, 4096)
/ temperature             one score per token, sharpened (<1) or flattened (>1)
top-k                     everything below the k-th best score becomes -inf (probability 0)
softmax      (1, 4096)    probabilities summing to 1
multinomial  (1, 1)       the next token id; concatenate -> ctx becomes (1, T+1)
```
**Temperature.** Dividing scores by `t` before softmax: `t -> 0` makes the top token take all the probability (greedy), `t = 1` is the model's own belief,
`t > 1` flattens it so rare tokens get picked. **Top-k** removes the long tail of nonsense tokens so a flat distribution cannot pick them. See the side-by-side table in `docs/RESULTS.md`:
greedy repeats one story, 0.8 varies and stays fluent, 1.5 invents words, 2.5 is noise, and 1.5 without top-k is worse than 1.5 with it.

**Determinism.** A `torch.Generator` seeded per request means the same seed gives the same tokens, and the service returns the seed it used, so any sample you liked can be reproduced.
**Stopping.** Generation stops at `<|endoftext|>` (`finish_reason = "stop"`) or at `max_new_tokens` (`"length"`). The stop token is trimmed from the text. An empty prompt starts from `<|endoftext|>`, the context every training story began with.
**Cost.** No KV cache: every new token re-runs the whole context, so cost per token grows with the prompt. 100 tokens took about 0.5 s here, fine for 256 context.

**Versus the notebook.** The notebook ended at embeddings. Here the same tokenizer that encoded training data encodes the prompt, and the trained tied embedding matrix both reads the prompt and scores the next token.

<details><summary>Check questions</summary>

1. *Why does temperature 0.2 look nearly identical to greedy?* Dividing logits by 0.2 multiplies the gaps by 5 before softmax, so the top token gets almost all the probability.
2. *Why combine top-k with temperature above 1?* High temperature flattens the distribution so unlikely tokens become plausible picks; top-k removes those tokens before sampling.
3. *What does a KV cache change?* It stores each layer's past keys and values, so each new token only computes attention for itself instead of re-running all `T` positions: per-token cost stops growing with the context.
</details>

**Interview angle.** "Generation is a loop over one forward pass: take the last position's logits, apply temperature and top-k, sample, append. I seeded a per-request generator, so any output is reproducible,
and I returned the seed and a finish reason from one service that both the CLI and the API call. I also know what it costs: without a KV cache each token re-runs the whole context."

## Phase 7: the API (app factory, lifespan, dependency boundaries)

**Concept.** The API adds no ML. It translates HTTP into a `CompletionRequest`, calls `CompletionService`, and translates the result back. All the model code stays importable and testable
without a web server: `serving` is the only package that imports FastAPI, and it depends on `inference`, never the reverse.

**The pieces.**
```
Settings.from_env()          frozen dataclass: model dir, max concurrency, queue timeout, threads
create_app(settings, loader) builds the FastAPI app; `loader` is injectable
lifespan                     starts a thread that loads the model, so /health answers while weights load
routes                       /health /v1/model /v1/completions  (+ static page at /, OpenAPI at /docs)
schemas (pydantic)           prompt 1-500 chars, max_new_tokens 1-256, temperature 0-2, top_k >= 1, seed >= 0
```
**Why a factory.** `create_app` returns a fresh app each call, so each test builds one with a tiny random model (`loader=lambda _: service`) and no files on disk. A module-level `app = FastAPI()` that loads weights on import cannot be tested this way.
**Why a background load.** While the model loads, `/health` is 200 (the process is alive) but `/v1/*` is 503 ("model is loading"). A host such as a Space can tell "starting" from "dead", and if loading fails the 503 message says why instead of the process crashing.
**Status codes.** 422 bad input (pydantic, plus `top_k` above the vocabulary, which is only known once the model is loaded), 503 not ready, 429 all slots busy. The answer to the predict-first question: a third simultaneous request waits up to `queue_timeout` (2 s) for a slot, then gets **429**, not a hang and not a crash. The test blocks a real generation, sends a second request, and checks the 429 and that the first one still returns 200.
**Why `def`, not `async def`, for the completion route.** Generation is CPU-bound and blocking. FastAPI runs a plain `def` in a worker thread, so the event loop keeps serving `/health` meanwhile. An `async def` would freeze the whole server for the duration of each generation.
**Concurrency.** A CPU forward pass already uses many threads, so two at a time is the sweet spot; more would only make each request slower.
**Safety detail.** The demo page writes model output with `textContent`, never `innerHTML`, so generated text can never be interpreted as HTML.

**Versus the notebook.** None of this existed. It is the part that turns a trained file into something a stranger can use.

<details><summary>Check questions</summary>

1. *Why inject the loader into `create_app`?* So tests can serve a tiny random model without files, and a failing load can be simulated; the app code is identical in both cases.
2. *Why is `/health` separate from readiness?* Liveness says the process is up; the model may still be loading. Mixing them makes an orchestrator kill a healthy-but-slow-starting container.
3. *What breaks if the completion route is `async def`?* The blocking generation runs on the event loop, so every other request, including `/health`, stalls until it finishes.
</details>

**Interview angle.** "I kept FastAPI out of the core library: the CLI and the API call the same `CompletionService`. The app is built by a factory with an injectable model loader, so my API tests run against a tiny random model in milliseconds,
including the 503-while-loading and 429-when-busy paths. Generation runs in the threadpool under a bounded semaphore, so a small CPU host degrades with a clear 429 instead of falling over."
