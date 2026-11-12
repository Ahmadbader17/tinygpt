"""One sandbox read: mean loss of a checkpoint on fixed dev windows. usage: 07_read.py <ckpt dir> <seed> <purpose>"""
import sys
from pathlib import Path
sys.path.insert(0, "analysis")
from common import dev_windows
from tinygpt.wsp.sandbox import Sandbox
ckpt, seed, purpose = sys.argv[1], int(sys.argv[2]), sys.argv[3]
sb = Sandbox.open(Path("work"))
out = sb.read(Path(ckpt), dev_windows(64, 256, seed), purpose=purpose)
print(f"call {out['call']} ckpt={ckpt} seed={seed} mean_nll={out['mean_nll']:.4f} used={sb.used} remaining={sb.remaining}")
