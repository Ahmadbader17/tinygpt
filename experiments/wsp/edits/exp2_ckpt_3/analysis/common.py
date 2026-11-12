"""Shared helpers: fixed dev token windows (weights never touched here)."""
from pathlib import Path
import random
from tinygpt.tokenizer import BPETokenizer


def dev_windows(n=64, length=256, seed=0, skip=0):
    """n random `length`-token windows from the first 400 dev stories joined by <|endoftext|>."""
    tok = BPETokenizer.load(Path("tokenizer.json"))
    stories = [s.strip() for s in Path("dev/stories.txt").read_text(encoding="utf-8").split("\n\n") if s.strip()]
    stream = []
    for s in stories[1:401]:  # the first chunk of the file starts mid-story
        stream.append(tok.eot_id)
        stream.extend(tok.encode(s))
    rng = random.Random(seed)
    starts = [rng.randrange(0, len(stream) - length) for _ in range(n + skip)][skip:]
    return [stream[a:a + length] for a in starts]
