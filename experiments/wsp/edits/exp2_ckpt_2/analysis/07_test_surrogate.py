"""Pass 3: same dev windows as pass 1, on the surrogate edit."""
import sys
from pathlib import Path
sys.path.insert(0, "analysis")
from common import dev_sets
from tinygpt.wsp.sandbox import Sandbox
windows, whole = dev_sets()
sb = Sandbox.open(Path("work"))
o = sb.read(Path("edited/surrogate_h1h7"), windows,
            purpose="Test: is L5 head 6 functionally redundant with its weight/attention neighbours h1 and h7? Edit sets its zeroed "
                    "write matrix to mean(OV1, OV7) routed through V6 (weights only). Weights predict NO gain (OV cosine of h1,h7 is 0.15); "
                    "adopt only if window loss falls clearly below the damaged 1.7194, otherwise the lost write matrix is declared unrecoverable.")
print("surrogate windows mean_nll", o["mean_nll"], "used", sb.used, "remaining", sb.remaining)
