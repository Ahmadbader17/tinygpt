"""Confirmation reads on data not used so far. usage: 08_read_confirm.py <ckpt dir> <windows|stories> <purpose>
windows: 64 fresh random 256-token windows (seed 1). stories: 64 whole dev stories from their start
(<|endoftext|> + story, stories 401.. that fit in 256 tokens)."""
import sys
from pathlib import Path
sys.path.insert(0, "analysis")
from common import dev_windows
from tinygpt.tokenizer import BPETokenizer
from tinygpt.wsp.sandbox import Sandbox
ckpt, mode, purpose = sys.argv[1], sys.argv[2], sys.argv[3]
if mode == "windows":
    seqs = dev_windows(64, 256, 1)
else:
    tok = BPETokenizer.load(Path("tokenizer.json"))
    stories = [s.strip() for s in Path("dev/stories.txt").read_text(encoding="utf-8").split("\n\n") if s.strip()]
    seqs = []
    for s in stories[401:]:
        ids = [tok.eot_id] + tok.encode(s)
        if len(ids) <= 256:
            seqs.append(ids)
        if len(seqs) == 64:
            break
sb = Sandbox.open(Path("work"))
out = sb.read(Path(ckpt), seqs, purpose=purpose)
print(f"call {out['call']} ckpt={ckpt} mode={mode} mean_nll={out['mean_nll']:.4f} tokens={sum(out['tokens'])} used={sb.used} remaining={sb.remaining}")
