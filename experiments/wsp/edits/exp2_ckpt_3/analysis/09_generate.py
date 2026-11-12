"""Sanity generation from the repaired checkpoint on 8 dev prompts x 2 seeds."""
import json
from pathlib import Path
from tinygpt.wsp.sandbox import Sandbox
from tinygpt.wsp.generation import DecodeConfig
prompts = [d["tokens"] for d in json.loads(Path("dev/prompts.json").read_text())[:8]]
sb = Sandbox.open(Path("work"))
out = sb.generate(Path("repaired"), prompts, seeds=(0, 1), decode=DecodeConfig(120, temperature=0.8, top_k=40),
                  purpose="Sanity check: the repaired checkpoint should produce coherent, non-looping story text (no side effect of shrinking layer-3 attention output).")
print({k: v for k, v in out.items() if k != "weights_sha256"}, sb.used, sb.remaining)
rows = json.loads(Path(out["generations_file"]).read_text())
for r in rows[:3]:
    print("---", r["text"][:400])
