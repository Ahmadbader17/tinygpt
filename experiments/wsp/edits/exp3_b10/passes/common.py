"""Shared helpers for sandbox calls (paths relative to the workspace root)."""
import json
from pathlib import Path
from tinygpt.wsp.sandbox import Sandbox
from tinygpt.wsp.generation import DecodeConfig
from tinygpt.tokenizer import BPETokenizer

WS = Path(".")
tok = BPETokenizer.load(WS / "tokenizer.json")
PROMPTS = [x["tokens"] for x in json.loads((WS / "dev/prompts.json").read_text())]
GEN_PROMPTS = PROMPTS[:64]          # fixed once, before any model output was seen
SAMPLE = DecodeConfig(200, temperature=0.8, top_k=40)

def sandbox():
    return Sandbox.open(WS / "work")

def dev_story_rows(n=64, length=256, skip=0):
    """First n dev stories (after `skip`) with >= length-1 tokens, as [EOT] + first length-1 tokens."""
    stories = [s.strip() for s in (WS / "dev/stories.txt").read_text().split("\n\n") if s.strip()]
    rows = []
    for s in stories[skip:]:
        ids = tok.encode(s)
        if len(ids) >= length - 1:
            rows.append([tok.eot_id] + ids[: length - 1])
        if len(rows) == n:
            break
    return rows
