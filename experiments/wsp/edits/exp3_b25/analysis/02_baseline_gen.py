"""Pass 1: baseline sampled generations, prompts 0-63, seed 0."""
import sys; sys.path.insert(0, "analysis")
from common import gen
gen("checkpoint", 0, 64,
    "Baseline for later paired comparisons (prompts 0-63, seed 0, T=0.8, top-k 40), and material to test the "
    "hypothesis that sampled 4-gram loops are copies of earlier context (names/phrases) rather than tight cycles.")
