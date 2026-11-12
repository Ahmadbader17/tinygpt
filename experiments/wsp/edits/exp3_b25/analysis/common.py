"""Shared helpers for the sandbox scripts (paths relative to the workspace root)."""
import json
from pathlib import Path
from tinygpt.wsp.sandbox import Sandbox
from tinygpt.wsp.generation import DecodeConfig
from tinygpt.tokenizer import BPETokenizer

ws = Path(".")
tok = BPETokenizer.load(ws / "tokenizer.json")
PROMPTS = [p["tokens"] for p in json.load(open(ws / "dev" / "prompts.json"))]
SAMPLE = DecodeConfig(200, temperature=0.8, top_k=40)


def sandbox():
    return Sandbox.open(ws / "work")


def gen(ckpt, lo, hi, purpose, seeds=(0,)):
    sb = sandbox()
    out = sb.generate(ws / ckpt, PROMPTS[lo:hi], seeds=seeds, decode=SAMPLE, purpose=purpose)
    print({k: v for k, v in out.items()}, "used", sb.used, "remaining", sb.remaining)
    return out
