---
license: mit
library_name: pytorch
tags:
  - text-generation
  - gpt
  - from-scratch
  - tinystories
datasets:
  - roneneldan/TinyStories
language:
  - en
metrics:
  - perplexity
---

# Tiny GPT (5.8M parameters)

A small GPT-2-style decoder-only transformer, written from scratch in PyTorch (no `transformers` classes), trained on a
laptop CPU on a subset of [TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories). It continues short children's stories.
Code, training logs and the full write-up are in the GitHub repository.

## Model details
| | |
|---|---|
| Architecture | pre-norm GPT, 6 layers, d_model 256, 8 heads, context 256, learned positions, tied embeddings |
| Parameters | 5,853,184 |
| Tokenizer | byte-level BPE, vocabulary 4096, trained on the training split only (`tokenizer.json`) |
| Files | `model.safetensors`, `model_config.json` (our `ModelConfig`), `tokenizer.json`, `model_info.json` |

## Training
- **Data:** first 300,000 stories (60.8M tokens) of TinyStories V2 (GPT-4 generated); validation on 5,000 held-out stories.
- **Run:** 9,500 steps x 8,192 tokens = 77.8M tokens (about 1.28 passes over the data), AdamW, peak lr 1e-3 with warmup and cosine decay, 12 CPU threads, 7.9 hours of running time on one laptop (Intel Core Ultra 9 185H).

## Evaluation
| Model | Validation loss | Perplexity |
|---|---|---|
| Uniform guess | 8.32 | 4096 |
| Unigram counts | 5.88 | 357 |
| Bigram counts | 3.65 | 38.4 |
| **This model** | **1.688** | **5.41** |

Perplexity is per BPE token with this 4096-token tokenizer, on TinyStories text, so it is not comparable to numbers from other tokenizers or datasets.

## Intended use
Learning, demos and experiments with small language models; generating simple children's-story-style text.

## Limitations and risks
- **Narrow domain.** It only knows the simple vocabulary and plots of TinyStories. It cannot follow instructions, answer questions or hold a conversation.
- **Quality.** Stories are fluent at sentence level but repeat phrases, lose objects and plot threads, and sometimes contradict themselves. Samples in the repository's `docs/RESULTS.md` include failures, not just good examples.
- **Synthetic data.** The training stories were written by a large language model, so the model inherits that model's stylistic and social biases (names, gender roles, settings). It has not been evaluated for bias or safety.
- **Not for** factual content, advice, or anything where wrong or biased text could cause harm.

## Data and licence
Code and weights in this repository are released under the MIT licence. The training data, TinyStories by Ronen Eldan and Yuanzhi Li, is
licensed CDLA-Sharing-1.0 on the dataset card; please credit it and check the dataset card for current terms.

## Usage
```bash
git clone <github repo> && cd tiny-gpt && make setup
TINYGPT_HF_REPO=<this repo id> TINYGPT_MODEL_DIR=./model uv run uvicorn tinygpt.serving.app:create_app --factory
```
