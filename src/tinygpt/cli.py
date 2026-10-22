"""Command line entry point. Subcommands are added phase by phase."""

import argparse
import dataclasses
import json
import logging
import sys
import time
from collections.abc import Callable, Sequence
from pathlib import Path

import torch

from tinygpt.baselines import compute_baselines
from tinygpt.config import SampleConfig, load_config
from tinygpt.data.dataset import MemmapTokenSampler
from tinygpt.data.prepare import prepare_data
from tinygpt.inference import generate
from tinygpt.model import GPT
from tinygpt.tokenizer import BPETokenizer, train_bpe
from tinygpt.training.bench import bench_threads
from tinygpt.training.trainer import Trainer
from tinygpt.utils.device import pick_device
from tinygpt.utils.env import probe_environment
from tinygpt.utils.logging import setup_logging
from tinygpt.utils.seed import seed_everything


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
    train = sub.add_parser("train", help="train a model; re-run with --resume to continue")
    train.add_argument("--config", required=True)
    train.add_argument("--data-dir", default="data")
    train.add_argument("--run-dir", help="default: runs/<timestamp>_<config name>")
    train.add_argument(
        "--resume", action="store_true", help="continue from <run-dir>/checkpoints/last"
    )
    train.add_argument("--max-steps", type=int, help="override [train] max_steps")
    train.add_argument("--max-hours", type=float, help="stop and save after this much wall time")
    bench = sub.add_parser("bench", help="measure training tokens/s for several thread counts")
    bench.add_argument("--config", required=True)
    bench.add_argument("--threads", default="6,8,10,12,14,16,22")
    bench.add_argument("--steps", type=int, default=6)
    bench.add_argument("--token-budget", type=int, default=100_000_000)
    base = sub.add_parser("baselines", help="unigram and bigram perplexity on the validation split")
    base.add_argument("--config", required=True)
    base.add_argument("--data-dir", default="data")
    return parser


PROMPTS = [
    "Once upon a time",
    "One day, a little girl named",
    "The dog was very",
    "Tom and his mom went to the",
    "The sun was shining and",
]


def make_sample_fn(tokenizer: BPETokenizer, device: torch.device) -> Callable[[GPT], str]:
    """Same five prompts and seeds at every evaluation, so progress is visible in samples.md."""

    def sample(model: GPT) -> str:
        out = []
        for i, prompt in enumerate(PROMPTS):
            idx = torch.tensor([tokenizer.encode(prompt)], device=device)
            cfg = SampleConfig(max_new_tokens=60, temperature=0.8, top_k=40, seed=i)
            ids = generate(model, idx, cfg, tokenizer.eot_id)[0].tolist()
            out.append(f"**{prompt}** -> {tokenizer.decode(ids)}")
        return "\n\n".join(out)

    return sample


def run_train(args: argparse.Namespace) -> None:
    cfg = load_config(args.config)
    if args.max_steps:
        cfg = dataclasses.replace(
            cfg, train=dataclasses.replace(cfg.train, max_steps=args.max_steps)
        )
    run_dir = Path(args.run_dir or f"runs/{time.strftime('%Y%m%d_%H%M')}_{cfg.name}")
    setup_logging(log_file=run_dir / "train.log")
    torch.set_num_threads(cfg.train.num_threads)
    seed_everything(cfg.seed)
    data_dir = Path(args.data_dir)
    tokenizer = BPETokenizer.load(data_dir / "tokenizer" / "tokenizer.json")
    if tokenizer.vocab_size != cfg.model.vocab_size:
        raise SystemExit(
            f"tokenizer vocab {tokenizer.vocab_size} != model vocab {cfg.model.vocab_size}; "
            "run prepare-data with the same config"
        )
    t = cfg.model.context_length
    device = pick_device()
    trainer = Trainer(
        GPT(cfg.model),
        cfg,
        MemmapTokenSampler(data_dir / "processed" / "train.bin", t),
        MemmapTokenSampler(data_dir / "processed" / "val.bin", t),
        run_dir,
        device,
        make_sample_fn(tokenizer, device),
    )
    log = logging.getLogger("tinygpt.train")
    log.info(
        "run dir %s, %d parameters, %d threads",
        run_dir,
        trainer.model.num_parameters(),
        cfg.train.num_threads,
    )
    if args.resume:
        trainer.resume()
    trainer.fit(max_hours=args.max_hours)


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
    elif args.command == "train":
        run_train(args)
    elif args.command == "bench":
        cfg = load_config(args.config)
        counts = [int(n) for n in args.threads.split(",")]
        print(f"{'threads':>8} {'tokens/s':>10} {'hours for ' + str(args.token_budget):>26}")
        for r in bench_threads(cfg, counts, args.steps, args.token_budget):
            print(
                f"{r.threads:>8} {r.tokens_per_sec:>10.0f} {r.hours_for_budget:>26.2f}", flush=True
            )
    elif args.command == "baselines":
        cfg = load_config(args.config)
        processed = Path(args.data_dir) / "processed"
        result = compute_baselines(
            processed / "train.bin", processed / "val.bin", cfg.model.vocab_size
        )
        print(json.dumps(result, indent=2))
    elif args.command == "train-tokenizer":
        cfg = load_config(args.config)
        text = Path(args.input).read_text(encoding="utf-8")
        tokenizer = train_bpe(text.split("\n\n"), cfg.data.tokenizer_vocab_size)
        tokenizer.save(Path(args.out))
        print(f"saved {tokenizer.vocab_size}-token tokenizer to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
