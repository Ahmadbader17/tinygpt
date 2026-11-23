# Learning log

## Phase 0: scaffold, config, environment

**Concept.** A `src/` layout places the package in `src/tinygpt/`, so `import tinygpt` works only
once the package is installed (`uv sync` performs an editable install). Tests therefore exercise the installed
package, and a file missing from the packaging surfaces immediately instead of being masked by the
current directory. Configs are frozen dataclasses loaded from TOML: a run reads its settings once
and cannot change them mid-flight, and unknown keys raise an error, so a typo such as `drop_out` fails loudly.

**Shapes preview.** `ModelConfig` checks `d_model % n_head == 0` because attention later reshapes
`(B, T, C) -> (B, n_head, T, C // n_head)`. With `small`, `C=256` and `n_head=8`, giving head dimension 32.

**Predict-first answer.** The tied embedding matrix has 4096 × 256 = 1,048,576 parameters, about 18% of
the roughly 5.8M total (blocks: ~12 × 256² × 6 = 4.7M; positions: 256 × 256 = 65k). With GPT-2's 50257-token
vocabulary it would be 12.9M, which is why a 4096-token BPE was trained.

**Link to the notebook.** The notebook has no config: vocabulary size and context length are loose
variables. Here they live in one place, and every run saves its resolved config.

<details><summary>Check questions</summary>

1. *Why does the `src/` layout catch packaging bugs?* Tests import the installed package, not the
   working directory.
2. *Why frozen dataclasses?* Immutability prevents hidden mutation during a run; instances are hashable and safe to share.
3. *Why must `d_model` be divisible by `n_head`?* The heads split the channel dimension evenly.
</details>

**Interview angle.** Configuration is declarative (TOML into frozen, validated dataclasses), so
every run is reproducible from one saved file. Strict typing and `make check` form the single
gate used locally and in CI.

## Phase 1: tokenizer (byte-level BPE)

**Concept.** Text must become integers. The notebook split on words and built a dictionary, so
unseen words required `<|unk|>`. Byte-level BPE starts from the 256 byte values and *learns* merges:
it counts adjacent pairs in the training text, fuses the most frequent into a new token, and repeats until the
vocabulary reaches 4096 (256 bytes + 1 special token + 3839 merges). Encoding replays the merges in the order
they were learned. Because every string is a sequence of bytes, no input is ever unknown.

**Shapes.** `encode: str -> list[int]` (length L, each id in `[0, vocab)`); the pipeline later
turns these into `(B, T)` int tensors. Decoding concatenates bytes and decodes UTF-8, so
a single id may print as `�` when it holds a partial multi-byte character.

**Worked example (tiny vocabulary 512, fixture stories).** `"the"` -> `[84, 257]` = `t` + `he`
(not one token: only 255 merges are available and "the" is usually preceded by a space); `" the"` ->
one token `[261]`; `"Pip"` -> `Pi` + `p`; `"🦊"` -> 4 byte tokens (the emoji is 4 UTF-8 bytes
and never appeared in training, so no merges cover it). With vocabulary 4096 on real TinyStories, common
words become single tokens.

**Design choices.** `add_prefix_space=False` keeps `" the"` and `"the"` distinct, as in GPT-2.
The tokenizer is trained on the training split only. `Tokenizer` is a `Protocol`, so the model and trainer
depend on the interface and the tiktoken adapter can be substituted (used in one comparison test).

<details><summary>Check questions</summary>

1. *Why can byte-level BPE never produce an unknown token?* The base alphabet is all 256 bytes,
   so any input decomposes into known symbols.
2. *Why does `" the"` have its own token while `"the"` may not?* Pre-tokenization keeps the leading
   space attached to the word, and `" the"` is far more frequent mid-sentence.
3. *What does a larger vocabulary trade off?* Fewer tokens per text (cheaper attention) against a larger
   embedding and output matrix (vocab × d_model) and sparser statistics per token.
</details>

**Interview angle.** A 4096-token byte-level BPE was trained instead of reusing GPT-2's 50k vocabulary,
because at d_model=256 the embedding table alone would be more than twice the size of the rest of the model.
Byte-level encoding yields no unknown tokens, and lossless round trips were verified on unicode and emoji.

## Phase 2: data pipeline (clean, encode, memmap)

**Concept.** Training should never touch raw text. The corpus is converted to numbers once, and training reads
only those. `make data` runs `tinygpt prepare-data`: it streams stories, cleans them, trains BPE on the training split only,
encodes each story plus `<|endoftext|>`, and writes one flat `uint16` file per split (`train.bin`, `val.bin`) and a
`meta.json`.

**Shapes.** `train.bin` is a flat `(N,)` array of uint16 (N = 60,838,396, about 2 bytes per token). A training batch picks
`B` random starts `s` and slices `x = tokens[s : s+T]`, `y = tokens[s+1 : s+T+1]`, giving `x, y` of shape `(B, T)`,
cast to int64 for the embedding lookup. `y` is `x` shifted left by one, so at every position the target is the
next token, and one window yields `T` training examples at once.

**Comparison with the notebook.** The notebook's `GPTDatasetV1` builds a Python list holding a tensor for every window, so memory grows
as `num_windows × context_length`. The memmap maps the file and the OS pages in only what is touched, so RAM use stays small and
random starts replace `stride`. `SlidingWindowDataset` is retained for deterministic evaluation and tests.

**Design choices.** Validation data is held out and the tokenizer never sees it. Cleaning is light (whitespace, control characters) so
the model still sees natural text. `meta.json` stores the tokenizer SHA256 so a checkpoint can be checked against its tokenizer.
Measured: 202.8 tokens per story, with 13 of 300,000 stories dropped as empty.

<details><summary>Check questions</summary>

1. *Why is uint16 sufficient, and what breaks past 65,535?* Ids are below 4096, so 2 bytes hold them. A vocabulary of 65,536 or more would
   overflow, and `encode_split` raises an error instead of silently wrapping.
2. *Why are `x` and `y` offset by one token?* The model is trained to predict the next token at every position, so the target
   at position `i` is the input at `i+1`.
3. *Why train the tokenizer on the training split only?* If it saw validation text, the merges would be tuned to it and validation
   loss would be slightly optimistic. Held-out data must remain untouched.
</details>

**Interview angle.** 300k stories were tokenised once into a 122 MB uint16 memmap (60.8M tokens), so batches are random slices with
almost no RAM, instead of materialising every window as a tensor. The tokenizer is fit on the training split only, and `meta.json` records its
hash so checkpoints can be matched to the correct vocabulary.

## Phase 3: the model (attention, MLP, GPT)

**Concept.** A GPT turns token ids into next-token scores. Each token id becomes a vector, a position vector is added,
and the sum passes through a stack of identical blocks. Each block lets tokens look at earlier tokens (attention)
and then transforms each token independently (MLP). A final LayerNorm and a linear head produce one score per vocabulary entry.

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

**The causal mask.** `scores[t, s]` is how much position t attends to position s. Setting it to `-inf` for s > t makes the
softmax weight exactly 0, so token t cannot read the future. This allows one forward pass to train all `T` next-token
predictions at once without leakage. Test: changing a later token leaves earlier logits identical.

**Why divide by sqrt(head_dim).** A dot product of two 32-dimensional vectors with unit-variance entries has variance 32. Without scaling, the
scores grow large, softmax saturates to near one-hot, and gradients vanish. Dividing by sqrt(32) restores a variance of about 1.

**Pre-norm.** `x = x + attn(ln1(x))` keeps an un-normalised identity path from the output back to the input, so gradients reach early
layers directly. Post-norm places LayerNorm on that path and is harder to train without careful warmup.

**Weight tying.** `lm_head.weight is wte.weight`: one (4096 × 256) matrix reads tokens in and scores tokens out. It saves
1,048,576 parameters (18% of the model) and links the two roles. Counted once, the model has **5,853,184 parameters**
(4,739,072 without embeddings). A test recomputes this by hand from the design.

**Initialisation and initial loss.** All weights are drawn from N(0, 0.02), so the logits are tiny and the softmax is almost uniform over V tokens. The cross-entropy
of a uniform guess is `ln(V)`: ln(4096) = 8.318, and the measured untrained loss was 8.371 (tiny: 6.243 against ln(512) = 6.238).
A first loss far from `ln(V)` indicates a defect, which is why this is a test. The residual output projections receive an extra `1/sqrt(2*n_layer)` factor so that variance does not grow with depth.

**Comparison with the notebook.** The notebook stopped at embeddings. Here the embedding layer is the front of a model, and the same `wte` doubles as the output layer.

<details><summary>Check questions</summary>

1. *Why is the untrained loss ln(V)?* Near-zero logits give a uniform distribution, so every token has probability 1/V and
   the cross-entropy is `-ln(1/V) = ln V`.
2. *What would happen without the causal mask?* Position t could read token t+1, which is its own training target.
   Training loss would drop quickly, but the model would be useless at generation time, when the future does not exist.
3. *Why are both the naive and sdpa attention paths kept?* The naive path is readable and serves as a reference; sdpa is faster.
   A test checks that they agree to 1e-5, which verifies the optimised path.
</details>

**Interview angle.** GPT-2 style attention was implemented twice: an explicit softmax(QK^T/sqrt(d))V with a triangular mask, and the
fused PyTorch kernel. A test asserts that they agree and that the future never leaks into the past. The parameter count is checked
against a hand-derived formula and the untrained loss against ln(vocab), which catches wiring and initialisation bugs before any training.

## Phase 4: the training loop

**Concept.** Training is the same five steps repeated: get a batch, run forward to a loss, run backward to gradients, clip, update. Everything else
(schedule, evaluation, checkpoints) exists to make those steps safe to run for hours.

**Anatomy of one step (`Trainer.train_step`), small config, B=32, T=256.**
```
x, y = sampler.batch(32, rng)        (32, 256) int64 each; y is x shifted left by one
logits, loss = model(x, y)           logits (32, 256, 4096) -> loss: one scalar (mean over 8192 positions)
(loss / grad_accum).backward()       fills p.grad for all 5.85M parameters
clip_grad_norm_(params, 1.0)         if the global gradient norm exceeds 1, rescale it to 1
optimizer.step()                     AdamW update using the current learning rate
```
**Learning-rate schedule.** `lr_at(step)` rises linearly for 200 steps (Adam's moment estimates are noisy at the start, so large early steps
are risky), then follows a cosine from 1e-3 down to 1e-4 at the last step. It is a pure function, which makes it easy to test.

**AdamW detail.** Weight decay of 0.1 applies only to tensors with `ndim >= 2` (matrices and embeddings). Biases and LayerNorm gains are
excluded, because shrinking them toward zero has no regularising benefit.

**Checkpoint and resume.** A checkpoint contains the weights (safetensors), the optimiser state (Adam's two moment estimates per parameter, which is why the
checkpoint is about 3× the size of the weights), the step, and both RNG states. The test trains 5 steps and saves, then a new trainer with a different initialisation resumes
and produces the *same next 5 losses exactly*. Saves go to a temporary directory and are swapped in, so a crash cannot leave a partial checkpoint.
`SIGINT` and `SIGTERM` finish the current step, save `last`, and exit.

**Evaluation.** Every 250 steps the same validation windows (fixed seed) are scored, so the curve is comparable between points. `best`
is kept by validation loss, and `last` is used for resuming. `samples.md` repeats five fixed prompts with fixed seeds, making the progression from gibberish to English visible.

**Comparison with the notebook.** The notebook had a DataLoader over a list of tensors and no notion of a run. Here a run is a directory (config, environment, git commit, metrics, plot, samples, checkpoints).

<details><summary>Check questions</summary>

1. *Why divide the loss by `grad_accum_steps`?* Gradients add across `backward()` calls, so dividing makes the accumulated gradient the mean over all micro-batches, equivalent to one large batch.
2. *Why save optimiser state, not just weights?* Without Adam's moment estimates, a resumed run restarts with cold statistics and a different trajectory, and the resume test would fail.
3. *Why is "overfit one fixed batch" a good gradient check?* A model that cannot drive loss near zero on 64 tokens has a defect in the forward pass, the loss, the optimiser or the data path.
</details>

**Interview angle.** The training loop uses warmup plus cosine decay, global-norm clipping, weight decay on matrices only, and exact resume: a test checks that a resumed run
reproduces the next five losses bit for bit. Checkpoints are written atomically and the run saves on SIGTERM, so a night-long CPU run survives interruptions.

## Phase 5: benchmark, budget, baselines

**Compute budgeting.** Throughput determines what is feasible. `tinygpt bench` times real training steps for several thread counts on the development laptop (Core Ultra 9 185H, 22 threads):

| threads | 4 | 6 | 8 | 10 | 12 | 16 | 22 |
|---|---|---|---|---|---|---|---|
| tokens/s | 2.1k | 2.1k | 2.3k | 2.9k | 2.8k | 0.9k | 0.6k |

Using every thread is 4 to 5 times *slower* than using 12. The likely cause is that the chip mixes performance and efficiency cores, and PyTorch's parallel loops wait for the slowest
thread. The config therefore uses 12. The real run sustained about 2.5k tok/s (laptop thermals), so one pass over the 61M training tokens (7,500 steps) takes about 6.8 hours.
As a rule of thumb, about 20 tokens per parameter is "compute optimal" (Chinchilla); one epoch here gives about 10, a reasonable trade-off for a laptop budget.

**Baselines.** Predictions for the validation tokens that use only counts, with no neural network:

| model | perplexity | loss (nats) |
|---|---|---|
| uniform guess | 4096 | 8.32 |
| unigram (token frequencies) | 357 | 5.88 |
| bigram (previous token) | 38.4 | 3.65 |

Perplexity is exp(loss): the model is as uncertain as a fair die with that many sides. The trained GPT must beat the bigram table clearly; otherwise it has learned no more than which word tends to follow which.

**Reading loss curves.** Training and validation loss falling together indicates healthy training. Validation flat or rising while training falls indicates overfitting (not expected in one epoch). Sudden spikes indicate a learning rate that is too high or a bad batch (clipping helps).
Loss stuck near 7 to 8 indicates the model has learned only unigram frequencies.

<details><summary>Check questions</summary>

1. *Why can more threads be slower?* Synchronisation across unequal cores and memory-bandwidth limits; the slowest thread gates each parallel operation.
2. *Why compare against a bigram baseline?* It is the best achievable looking one token back, so beating it shows that the model uses longer context.
3. *Why one epoch rather than several?* No data repeats, so validation loss measures generalisation, and the budget fits the available time window.
</details>

**Interview angle.** Thread counts were benchmarked before committing compute, and using all 22 threads proved 4× slower than 12 on a hybrid CPU, so the run was sized from measured tokens per second.
The model was judged against unigram and bigram perplexity baselines computed on the same tokenisation, not only against its own loss.

## Phase 6: sampling and the CompletionService

**Concept.** The model never outputs text. For a context of `T` tokens it outputs scores `(T, V)`, and only the last row matters for generation.
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
**Temperature.** Scores are divided by `t` before softmax: `t -> 0` gives the top token all the probability (greedy), `t = 1` is the model's own belief,
and `t > 1` flattens the distribution so rare tokens are picked. **Top-k** removes the long tail of nonsense tokens so a flat distribution cannot select them. The side-by-side table in `docs/RESULTS.md` shows the effect:
greedy repeats one story, 0.8 varies and stays fluent, 1.5 invents words, 2.5 is noise, and 1.5 without top-k is worse than 1.5 with it.

**Determinism.** A `torch.Generator` seeded per request means the same seed gives the same tokens, and the service returns the seed it used, so any sample can be reproduced.
**Stopping.** Generation stops at `<|endoftext|>` (`finish_reason = "stop"`) or at `max_new_tokens` (`"length"`). The stop token is trimmed from the text. An empty prompt starts from `<|endoftext|>`, the context every training story began with.
**Cost.** There is no KV cache: every new token re-runs the whole context, so cost per token grows with the prompt. 100 tokens took about 0.5 s on the development machine, which is acceptable for a 256-token context.

**Comparison with the notebook.** The notebook ended at embeddings. Here the tokenizer that encoded the training data also encodes the prompt, and the trained tied embedding matrix both reads the prompt and scores the next token.

<details><summary>Check questions</summary>

1. *Why does temperature 0.2 look nearly identical to greedy?* Dividing logits by 0.2 multiplies the gaps by 5 before softmax, so the top token receives almost all the probability.
2. *Why combine top-k with a temperature above 1?* High temperature flattens the distribution so unlikely tokens become plausible picks; top-k removes those tokens before sampling.
3. *What does a KV cache change?* It stores each layer's past keys and values, so each new token computes attention only for itself instead of re-running all `T` positions, and per-token cost stops growing with the context.
</details>

**Interview angle.** Generation is a loop over one forward pass: take the last position's logits, apply temperature and top-k, sample, and append. A per-request seeded generator makes any output reproducible,
and a single service, shared by the CLI and the API, returns the seed and a finish reason. The cost is known: without a KV cache each token re-runs the whole context.

## Phase 7: the API (app factory, lifespan, dependency boundaries)

**Concept.** The API adds no ML. It translates HTTP into a `CompletionRequest`, calls `CompletionService`, and translates the result back. All model code stays importable and testable
without a web server: `serving` is the only package that imports FastAPI, and it depends on `inference`, never the reverse.

**The pieces.**
```
Settings.from_env()          frozen dataclass: model dir, max concurrency, queue timeout, threads
create_app(settings, loader) builds the FastAPI app; `loader` is injectable
lifespan                     starts a thread that loads the model, so /health answers while weights load
routes                       /health /v1/model /v1/completions  (+ static page at /, OpenAPI at /docs)
schemas (pydantic)           prompt 1-500 chars, max_new_tokens 1-256, temperature 0-2, top_k >= 1, seed >= 0
```
**Why a factory.** `create_app` returns a fresh app on each call, so each test builds one with a tiny random model (`loader=lambda _: service`) and no files on disk. A module-level `app = FastAPI()` that loads weights on import cannot be tested this way.
**Why a background load.** While the model loads, `/health` returns 200 (the process is alive) but `/v1/*` returns 503 ("model is loading"). A host such as a Space can distinguish "starting" from "dead", and if loading fails the 503 message states why instead of the process crashing.
**Status codes.** 422 for bad input (pydantic, plus `top_k` above the vocabulary, which is known only once the model is loaded), 503 when not ready, and 429 when all slots are busy. A third simultaneous request waits up to `queue_timeout` (2 s) for a slot, then receives **429**, not a hang or a crash. The test blocks a real generation, sends a second request, and checks the 429 and that the first request still returns 200.
**Why `def`, not `async def`, for the completion route.** Generation is CPU-bound and blocking. FastAPI runs a plain `def` in a worker thread, so the event loop keeps serving `/health` meanwhile. An `async def` would freeze the whole server for the duration of each generation.
**Concurrency.** A CPU forward pass already uses many threads, so two at a time is the practical optimum; more would only slow each request.
**Safety detail.** The demo page writes model output with `textContent`, never `innerHTML`, so generated text cannot be interpreted as HTML.

**Comparison with the notebook.** None of this existed in the notebook. This stage turns a trained file into something an outside user can use.

<details><summary>Check questions</summary>

1. *Why inject the loader into `create_app`?* Tests can then serve a tiny random model without files and simulate a failing load; the app code is identical in both cases.
2. *Why is `/health` separate from readiness?* Liveness shows that the process is up, while the model may still be loading. Combining them would cause an orchestrator to kill a healthy but slow-starting container.
3. *What breaks if the completion route is `async def`?* The blocking generation runs on the event loop, so every other request, including `/health`, stalls until it finishes.
</details>

**Interview angle.** FastAPI is kept out of the core library: the CLI and the API call the same `CompletionService`. The app is built by a factory with an injectable model loader, so API tests run against a tiny random model in milliseconds,
including the 503-while-loading and 429-when-busy paths. Generation runs in the threadpool under a bounded semaphore, so a small CPU host degrades with a clear 429 instead of failing.

## Phase 8: packaging and publishing (model card, Docker, Hub)

**Concept.** Training produces files on a developer machine; publishing makes them usable by others. This requires three artefacts: a **bundle** (weights, config and tokenizer that load without `runs/` or `data/`),
a **container** (the same server, with pinned dependencies, running as a non-root user), and a **model card** (an honest description of what the model is and is not).

**The pieces.**
```
tinygpt export        -> export/ : model.safetensors, model_config.json, tokenizer.json, model_info.json
Dockerfile            python:3.12-slim + pinned uv; `uv sync --frozen --no-dev` (CPU torch from uv.lock); user uid 1000; port 7860
Space                 same image; at startup downloads the bundle from TINYGPT_HF_REPO into TINYGPT_MODEL_DIR
tinygpt publish       dry run by default; --upload makes one atomic commit per repo; token only from HF_TOKEN
```
**Why `--frozen --no-dev`.** `--frozen` installs exactly the versions in `uv.lock` (the image that was tested is the image that ships), and `--no-dev` omits pytest, ruff and mypy. This was checked without Docker by installing into a clean folder: it resolved `torch 2.14.1+cpu`, included no pytest,
and served the same completion for seed 42 as the training environment.
**Why a non-root user and a health check.** A bug in the app then cannot write system files, and Spaces expect uid 1000 in any case. `HEALTHCHECK` queries `/health`, which responds while the weights are still downloading.
**Why weights are downloaded rather than baked in.** The image stays reusable and small, and a new model version is a Hub upload rather than a rebuild.

**Contents of an honest model card.** What the model is, how it was trained (data, tokens, hardware, time), real evaluation figures *with their baselines and caveats* (perplexity depends on the tokenizer), intended use, and limitations. Here these are a narrow domain, repetition and lost plots,
and synthetic training data that carries its generator's biases. The card avoids overselling: it states that the model is not an assistant and has not been tested for bias.

**Status at the time of this phase.** Nothing had been uploaded, because that requires the owner's Hugging Face token and approval. `docker build` had not run because Docker was not installed in the development environment.

<details><summary>Check questions</summary>

1. *Why bundle the tokenizer with the weights?* The model is meaningful only with the exact tokenizer it was trained with; `model_info.json` also stores its SHA256 so a mismatch can be detected.
2. *Why is the upload a dry run by default?* Publishing is outward-facing and hard to undo (copies may be cached or indexed), so the safe default lists what would be sent and the real upload requires an explicit flag.
3. *Why does a perplexity of 5.4 not show that this model beats GPT-2?* Perplexity is per token of a specific tokenizer on a specific dataset; this model uses a 4096-token BPE on simple children's stories, so it is comparable only to its own baselines.
</details>

**Interview angle.** The model bundle is separated from the training run so the server loads from one self-contained directory. The container is pinned, CPU-only and non-root, and the model card reports real baselines and explicit limitations. The publish command is a dry run
unless instructed otherwise and reads the token only from the environment, and the upload logic was tested against a fake Hub client.
