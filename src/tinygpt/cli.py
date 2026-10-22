"""Command line entry point. Subcommands are added phase by phase."""

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from tinygpt.config import load_config
from tinygpt.tokenizer import train_bpe
from tinygpt.utils.env import probe_environment


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="tinygpt", description="Tiny GPT")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("env", help="print the environment probe as JSON")
    show = sub.add_parser("show-config", help="print a resolved config")
    show.add_argument("--config", required=True)
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
    elif args.command == "train-tokenizer":
        cfg = load_config(args.config)
        text = Path(args.input).read_text(encoding="utf-8")
        tokenizer = train_bpe(text.split("\n\n"), cfg.data.tokenizer_vocab_size)
        tokenizer.save(Path(args.out))
        print(f"saved {tokenizer.vocab_size}-token tokenizer to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
