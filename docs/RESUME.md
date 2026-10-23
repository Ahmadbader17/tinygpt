# Resume bullets

Use only numbers that appear in `docs/RESULTS.md`. Links: GitHub `https://github.com/Ahmadbader17/tinygpt`, model `https://huggingface.co/Ahmadbader17/tiny-gpt`.

**Project line:** Tiny GPT: 5.8M-parameter language model built from scratch in PyTorch | GitHub | Hugging Face

- Built a 5.8M-parameter GPT from scratch in PyTorch (causal multi-head attention, pre-norm blocks, weight tying, custom training loop, no prebuilt model classes), trained on a laptop CPU on 77.8M TinyStories tokens in 7.9 hours, reaching validation perplexity 5.4 against 38.4 for a bigram baseline.
- Engineered a reproducible training pipeline: memory-mapped uint16 data, byte-level BPE tokenizer, warmup plus cosine LR schedule, gradient clipping, and atomic checkpoints with exact resume (a test shows a resumed run reproduces the next five losses bit for bit). A thread benchmark showed all 22 threads were 4 to 5 times slower than 12 on a hybrid CPU, so the run was sized from measured throughput.
- Wrote 120 tests, including equivalence of the readable and fused attention paths (1e-5), causality of the model, parameter count against a hand-derived formula, and initial loss near ln(vocab).
- Shipped a typed FastAPI service with input validation, a bounded concurrency queue (429 when busy, 503 while loading), and a 1.7 GB non-root Docker image; published the model and a model card with explicit limitations on the Hugging Face Hub.

**Say only what is true:** the model, attention and training loop are written from scratch, but the tokenizer is trained with the Hugging Face `tokenizers` library (rewrite exercise 2 in `LEARNING_PLAN.md` closes that gap).
Add "GitHub Actions CI" only while the badge in the README is green. The model is a story generator, not an assistant.
