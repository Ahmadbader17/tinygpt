"""Shared helpers: workspace paths, prompts, stories, sandbox."""
import json
from pathlib import Path

from tinygpt.tokenizer import BPETokenizer
from tinygpt.wsp.generation import DecodeConfig
from tinygpt.wsp.sandbox import Sandbox

WS = Path(".")
tok = BPETokenizer.load(WS / "tokenizer.json")
SAMPLED = DecodeConfig(200, temperature=0.8, top_k=40)
GREEDY = DecodeConfig(200, temperature=0.0, top_k=None)


def prompts():
    return [p["tokens"] for p in json.loads((WS / "dev" / "prompts.json").read_text())]


def stories():
    return [s.strip() for s in (WS / "dev" / "stories.txt").read_text().split("\n\n") if s.strip()]


def story_rows(start, n, length=256):
    """Dev stories start..start+n as token rows: <eot> + story (+ <eot> if it fits), cut at `length`."""
    out = []
    for s in stories()[start : start + n]:
        ids = [tok.eot_id] + tok.encode(s) + [tok.eot_id]
        out.append(ids[:length])
    return out


def sandbox():
    return Sandbox.open(WS / "work")


def log(name, out):
    keep = {k: v for k, v in out.items() if k not in ("nll", "tokens")}
    print(name, json.dumps(keep))
