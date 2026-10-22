"""Command line entry point. Subcommands are added phase by phase."""

import argparse
import json
import sys
from collections.abc import Sequence

from tinygpt.config import load_config
from tinygpt.utils.env import probe_environment


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="tinygpt", description="Tiny GPT")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("env", help="print the environment probe as JSON")
    show = sub.add_parser("show-config", help="print a resolved config")
    show.add_argument("--config", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "env":
        print(json.dumps(probe_environment(), indent=2))
    elif args.command == "show-config":
        print(load_config(args.config))
    return 0


if __name__ == "__main__":
    sys.exit(main())
