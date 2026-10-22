"""Command line entry point. Subcommands are added phase by phase."""

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from tinygpt.config import load_config
from tinygpt.data.prepare import prepare_data
from tinygpt.tokenizer import train_bpe
from tinygpt.utils.env import probe_environment
from tinygpt.utils.logging import setup_logging


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="tinygpt", description="Tiny GPT")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("env", help="print the environment probe as JSON")
    show = sub.add_parser("show-config", help="print a resolved config")
    show.add_argument("--config", required=True)
    prep = sub.add_parser("prepare-data", help="stream, clean, tokenize and write train/val .bin")
    prep.add_argument("--config", required=True)
    prep.add_argument("--data-dir", default="data")
    prep.add_argument("--train-source", help="local file or URL; default from config")
    prep.add_argument("--val-source", help="local file or URL; default from config")
    tok = sub.add_parser("train-tokenizer", help="train the byte-level BPE")
    tok.add_argument("--config", required=True)
    tok.add_argument(
        "--input", default="data/raw/train.txt", help="stories separated by blank lines"
    )
    tok.add_argument("--out", default="data/tokenizer/tokenizer.json")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "env":
        print(json.dumps(probe_environment(), indent=2))
    elif args.command == "show-config":
        print(load_config(args.config))
    elif args.command == "prepare-data":
        setup_logging()
        meta = prepare_data(
            load_config(args.config), Path(args.data_dir), args.train_source, args.val_source
        )
        print(json.dumps({k: meta[k] for k in ("train", "val", "vocab_size")}, indent=2))
    elif args.command == "train-tokenizer":
        cfg = load_config(args.config)
        text = Path(args.input).read_text(encoding="utf-8")
        tokenizer = train_bpe(text.split("\n\n"), cfg.data.tokenizer_vocab_size)
        tokenizer.save(Path(args.out))
        print(f"saved {tokenizer.vocab_size}-token tokenizer to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
