"""Shared, deterministic construction of the dev evaluation sets (no model involved)."""
from pathlib import Path
import random
from tinygpt.tokenizer import BPETokenizer


def dev_sets(n_windows=64, n_stories=64, seed=1234):
    tok = BPETokenizer.load(Path("tokenizer.json"))
    text = Path("dev/stories.txt").read_text(encoding="utf-8")
    stories = [s.strip() for s in text.split("\n\n") if s.strip()]
    rng = random.Random(seed)
    # windows: a stream of 600 randomly chosen stories joined by <|endoftext|>, cut at random offsets
    pick = rng.sample(range(len(stories)), 600)
    stream = []
    for i in pick:
        stream.append(tok.eot_id)
        stream.extend(tok.encode(stories[i]))
    starts = sorted(rng.sample(range(len(stream) - 256), n_windows))
    windows = [stream[s:s + 256] for s in starts]
    # whole stories from their start: <|endoftext|> then the story, truncated to the context
    pick2 = rng.sample(range(len(stories)), n_stories)
    whole = [([tok.eot_id] + tok.encode(stories[i]))[:256] for i in pick2]
    return windows, whole
