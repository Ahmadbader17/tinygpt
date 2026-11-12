"""Fixed dev evaluation rows, built once from dev/stories.txt (no model involved).

windows: 64 random 256-token windows of the dev token stream (stories joined by <|endoftext|>).
stories: 64 whole stories, each as <|endoftext|> + story, kept only if it fits in 256 tokens.
"""
import json, random
from pathlib import Path
from tinygpt.tokenizer import BPETokenizer

def build():
    out = Path("analysis/evalsets.json")
    if out.exists():
        return json.loads(out.read_text())
    tok = BPETokenizer.load(Path("tokenizer.json"))
    text = Path("dev/stories.txt").read_text(encoding="utf-8")
    stories = [s.strip() for s in text.split("\n\n") if s.strip()][1:]  # first one is a fragment
    rng = random.Random(1234)
    pick = rng.sample(range(len(stories)), 600)
    enc = [[tok.eot_id] + tok.encode(stories[i]) for i in pick]
    whole = [e for e in enc if len(e) <= 256][:64]
    stream = [t for e in enc for t in e]
    starts = [rng.randrange(0, len(stream) - 256) for _ in range(64)]
    windows = [stream[s:s + 256] for s in starts]
    d = {"windows": windows, "stories": whole}
    out.write_text(json.dumps(d))
    return d

if __name__ == "__main__":
    d = build()
    print(len(d["windows"]), len(d["stories"]), sum(map(len, d["stories"])))
