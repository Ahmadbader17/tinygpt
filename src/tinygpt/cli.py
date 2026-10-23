"""Command line entry point. Subcommands are added phase by phase."""

import argparse
import dataclasses
import json
import logging
import math
import sys
import time
from collections.abc import Callable, Sequence
from pathlib import Path

import torch

from tinygpt.baselines import compute_baselines
from tinygpt.config import SampleConfig, load_config
from tinygpt.data.dataset import MemmapTokenSampler
from tinygpt.data.prepare import prepare_data
from tinygpt.inference import CompletionRequest, CompletionService, export_bundle, generate
from tinygpt.model import GPT
from tinygpt.publish import plan_model_upload, plan_space_upload, upload
from tinygpt.tokenizer import BPETokenizer, train_bpe
from tinygpt.training.bench import bench_threads
from tinygpt.training.trainer import Trainer, estimate_loss
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
    gen = sub.add_parser("generate", help="complete a prompt with a trained run")
    gen.add_argument("--run", required=True, help="run directory, e.g. runs/overnight_long")
    gen.add_argument("--prompt", default="Once upon a time")
    gen.add_argument("--checkpoint", default="best", choices=["best", "last"])
    gen.add_argument("--tokenizer", default="data/tokenizer/tokenizer.json")
    gen.add_argument("--max-new-tokens", type=int, default=SampleConfig.max_new_tokens)
    gen.add_argument("--temperature", type=float, default=SampleConfig.temperature)
    gen.add_argument("--top-k", type=int, default=SampleConfig.top_k, help="0 disables top-k")
    gen.add_argument("--seed", type=int, help="fix for a reproducible sample")
    exp = sub.add_parser("export", help="bundle a run's weights, config and tokenizer for serving")
    exp.add_argument("--run", required=True)
    exp.add_argument("--out", default="export")
    exp.add_argument("--checkpoint", default="best", choices=["best", "last"])
    exp.add_argument("--data-dir", default="data")
    pub = sub.add_parser("publish", help="plan (default) or perform the Hugging Face upload")
    pub.add_argument("--bundle", default="export", help="directory made by `tinygpt export`")
    pub.add_argument("--model-repo", required=True, help="e.g. your-username/tiny-gpt")
    pub.add_argument("--space-repo", required=True, help="e.g. your-username/tiny-gpt-demo")
    pub.add_argument("--upload", action="store_true", help="really upload; needs HF_TOKEN in env")
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


def run_export(args: argparse.Namespace) -> None:
    """Bundle a checkpoint with its tokenizer and a model_info.json the API serves at /v1/model."""
    run_dir, data_dir = Path(args.run), Path(args.data_dir)
    cfg = load_config(run_dir / "config.toml")
    ckpt = run_dir / "checkpoints" / args.checkpoint
    state = json.loads((ckpt / "state.json").read_text(encoding="utf-8"))
    model = GPT.from_pretrained(ckpt)
    val = MemmapTokenSampler(data_dir / "processed" / "val.bin", cfg.model.context_length)
    loss = estimate_loss(model, val, 200, 32, torch.device("cpu"))  # same recipe as docs/RESULTS.md
    data_meta = json.loads((data_dir / "processed" / "meta.json").read_text(encoding="utf-8"))
    info = {
        "run": run_dir.name,
        "checkpoint": args.checkpoint,
        "training_steps": state["step"],
        "training_tokens": state["tokens_seen"],
        "val_loss": round(loss, 4),
        "val_perplexity": round(math.exp(loss), 3),
        "tokenizer_sha256": data_meta["tokenizer_sha256"],
    }
    export_bundle(
        run_dir, data_dir / "tokenizer" / "tokenizer.json", Path(args.out), info, args.checkpoint
    )
    print(json.dumps(info, indent=2))


def run_publish(args: argparse.Namespace) -> None:
    root = Path(".")
    plans = [
        plan_model_upload(Path(args.bundle), root / "deploy" / "MODEL_CARD.md", args.model_repo),
        plan_space_upload(root, root / "deploy" / "space_README.md", args.space_repo),
    ]
    for plan in plans:
        size = sum(p.stat().st_size for p in plan.files.values()) / 1e6
        print(f"{plan.repo_type} {plan.repo_id}: {len(plan.files)} files, {size:.1f} MB")
        for dest in plan.files:
            print(f"  {dest}")
    if not args.upload:
        print("\ndry run: nothing uploaded. Add --upload (with HF_TOKEN set) to publish.")
        return
    from huggingface_hub import HfApi

    api = HfApi()  # reads HF_TOKEN from the environment; the token is never a CLI argument
    variables = {"TINYGPT_HF_REPO": args.model_repo}
    for plan in plans:
        print(upload(plan, api, variables if plan.repo_type == "space" else None))


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
    elif args.command == "generate":
        run_dir = Path(args.run)
        torch.set_num_threads(load_config(run_dir / "config.toml").train.num_threads)
        service = CompletionService.from_run(run_dir, Path(args.tokenizer), args.checkpoint)
        res = service.complete(
            CompletionRequest(
                args.prompt,
                args.max_new_tokens,
                args.temperature,
                args.top_k or None,
                args.seed,
            )
        )
        print(args.prompt + res.text)
        print(
            f"\n[{res.completion_tokens} tokens, {res.finish_reason}, seed {res.seed}, "
            f"{res.latency_s:.1f}s]",
            file=sys.stderr,
        )
    elif args.command == "export":
        run_export(args)
    elif args.command == "publish":
        run_publish(args)
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
