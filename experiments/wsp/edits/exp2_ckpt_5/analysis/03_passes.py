"""The four forward passes: damaged vs repaired on the same fixed dev rows."""
import json, sys
from pathlib import Path
import numpy as np
from tinygpt.wsp.sandbox import Sandbox
sys.path.insert(0, "analysis")
from evalsets import build

ws = Path(".")
d = build()
sb = Sandbox.open(ws / "work")
res = {}
o = sb.read(ws / "checkpoint", d["windows"],
            purpose="H1: the damaged checkpoint's loss on 64 random 256-token dev windows is above the healthy 1.688, "
                    "and in block 5 the output norm of head 5 is several times that of the other heads "
                    "(weights show its proj columns are ~5.6x too large)",
            capture=("head_out", "resid_pre", "token_loss"), layers=(5,))
res["damaged_windows"] = o["mean_nll"]
z = np.load(o["capture_file"])
ho = z["head_out5"]
print("damaged windows mean_nll", o["mean_nll"])
print("block5 head_out mean L2 norm per head:", np.linalg.norm(ho, axis=-1).mean(axis=(0, 2)).round(2))
print("block5 resid_pre mean L2 norm:", np.linalg.norm(z["resid_pre5"], axis=-1).mean().round(2))
del z, ho
Path(o["capture_file"]).unlink()  # large file; the summary numbers above are what was needed
o = sb.read(ws / "checkpoint", d["stories"],
            purpose="H1b: the damaged checkpoint's loss on 64 whole dev stories from <|endoftext|> is above the healthy 1.605")
res["damaged_stories"] = o["mean_nll"]; print("damaged stories mean_nll", o["mean_nll"])
o = sb.read(ws / "repaired", d["windows"],
            purpose="H2: dividing head 5's columns of blocks.5.attn.proj.weight by the weight-derived factor 5.556 "
                    "brings loss on the same 64 windows back to about the healthy 1.688",
            capture=("head_out",), layers=(5,))
res["repaired_windows"] = o["mean_nll"]; print("repaired windows mean_nll", o["mean_nll"])
z = np.load(o["capture_file"])
print("repaired block5 head_out mean L2 norm per head:", np.linalg.norm(z["head_out5"], axis=-1).mean(axis=(0, 2)).round(2))
del z; Path(o["capture_file"]).unlink()
o = sb.read(ws / "repaired", d["stories"],
            purpose="H2b: the same repair brings loss on the same 64 whole stories back to about the healthy 1.605")
res["repaired_stories"] = o["mean_nll"]; print("repaired stories mean_nll", o["mean_nll"])
print("used", sb.used, "remaining", sb.remaining)
Path("analysis/03_passes.json").write_text(json.dumps(res, indent=1))
