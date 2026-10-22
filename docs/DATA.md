# Data

## Source
- **Dataset:** TinyStories, `roneneldan/TinyStories` on the Hugging Face Hub (GPT-4 generated V2 files).
- **Files:** `TinyStoriesV2-GPT4-train.txt` (training) and `TinyStoriesV2-GPT4-valid.txt` (validation).
  The dataset card lists only the train file by name; the valid file resolves at the same path and was used for the v1 run.
- **Licence:** CDLA-Sharing-1.0 (`cdla-sharing-1.0`, read from the dataset card on 2026-10-22). Repeat it in the model card.

## Subset
The files are streamed, not downloaded whole. We keep the first 300,000 training stories and the first 5,000 validation stories (`[data]` in `configs/small.toml`).

## Cleaning rules (`clean_story`)
1. Normalise `\r\n` and `\r` to `\n`.
2. Turn runs of spaces and tabs into one space.
3. Strip control characters (everything below `0x20` except `\n`, plus `0x7f`).
4. Strip each line and drop blank lines, so every run of newlines becomes a single `\n`.
5. Drop stories that are empty after cleaning.

Stories are stored blank-line separated in `data/raw/{train,val}.txt`. Cleaning leaves no blank lines inside a story, so a blank line is a safe separator.

## Tokenisation and format
- Byte-level BPE, vocab 4096, trained on the **training split only** (`data/tokenizer/tokenizer.json`).
- `<|endoftext|>` is appended after every story. All tokens are joined into one flat `numpy.uint16` file (valid because vocab < 65536).
- `data/processed/{train,val}.bin` hold the tokens; `meta.json` holds counts, the tokenizer SHA256, source URLs and the config.

## v1 numbers (`make data`, ~52 s on this machine)
| split | stories read | kept | dropped | tokens | tokens/story | file size |
|-------|-------------:|-----:|--------:|-------:|-------------:|----------:|
| train | 300,000 | 299,987 | 13 | 60,838,396 | 202.8 | 121.7 MB |
| val | 5,000 | 5,000 | 0 | 1,005,673 | 201.1 | 2.0 MB |

The training set holds about 60.8M tokens. DESIGN.md D4 plans a budget of about 35M to 40M tokens, so one epoch has headroom.

## Reproduce
`make data` (needs network). For offline use: `tinygpt prepare-data --config ... --train-source PATH --val-source PATH`.
`data/` is gitignored.
