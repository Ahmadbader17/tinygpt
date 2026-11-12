"""Pass 4: the surrogate edit on the whole-story set of pass 2 (an independent sample), to confirm or reject the small gain of pass 3."""
import sys, json
from pathlib import Path
sys.path.insert(0, "analysis")
from common import dev_sets
from tinygpt.wsp.sandbox import Sandbox
windows, whole = dev_sets()
sb = Sandbox.open(Path("work"))
o = sb.read(Path("edited/surrogate_h1h7"), whole,
            purpose="Confirmation on independent data: pass 3 showed the h1/h7 surrogate for L5 head 6 lowers window loss by 0.006. "
                    "Hypothesis: the gain is real, so whole-story loss also falls below the damaged 1.5792; if it does not, the surrogate is rejected and the repair is the identity.")
print("surrogate stories mean_nll", o["mean_nll"], "used", sb.used, "remaining", sb.remaining)
Path("analysis/08_rows.json").write_text(json.dumps({"nll": o["nll"], "tokens": o["tokens"]}))
