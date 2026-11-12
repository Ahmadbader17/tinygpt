"""Passes 1 and 2: loss of the damaged checkpoint on dev windows (with L5 attention captured) and on whole stories."""
import sys, json
from pathlib import Path
sys.path.insert(0, "analysis")
from common import dev_sets
from tinygpt.wsp.sandbox import Sandbox

windows, whole = dev_sets()
sb = Sandbox.open(Path("work"))
o1 = sb.read(Path("checkpoint"), windows,
             purpose="Baseline: the weights show blocks.5.attn.proj columns 192-223 (head 6) are exactly zero. "
                     "Hypothesis: this is the only damage and it costs a modest loss increase over healthy 1.688 on 256-token windows. "
                     "Capture L5 attention to test whether head 6 attends like its weight-neighbours h1/h7.",
             capture=("attn", "token_loss"), layers=(5,))
print("windows mean_nll", o1["mean_nll"], o1["capture_file"])
o2 = sb.read(Path("checkpoint"), whole,
             purpose="Baseline on whole dev stories from <|endoftext|>: hypothesis that the zeroed L5 head 6 raises loss above healthy 1.605 by a similar margin as on windows.")
print("stories mean_nll", o2["mean_nll"])
print("used", sb.used, "remaining", sb.remaining)
