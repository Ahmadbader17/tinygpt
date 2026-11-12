"""Exp 1, post hoc: how much does the result depend on where the thresholds sit?

NOT preregistered. Written after the verification, because two true positives of the
"positional" class measured within 0.01 of the 0.80 threshold. Uses the saved scores and the
committed predictions only; no model is run. Writes results/exp1_sensitivity.json.

    uv run python experiments/wsp/exp1_sensitivity.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from tinygpt.wsp import headscore
from tinygpt.wsp.headscore import CLASSES, evaluate_predictions, spearman

RESULTS = Path(__file__).resolve().parent / "results"
VARIANTS = {
    "preregistered": {},
    "positional 0.75": {"positional": 0.75},
    "positional 0.85": {"positional": 0.85},
    "inactive 0.0015": {"inactive": 0.0015},
    "inactive 0.003": {"inactive": 0.003},
    "previous_token 0.25": {"previous_token": 0.25},
    "previous_token 0.30": {"previous_token": 0.30},
    "strict: positional 0.85, inactive 0.0015": {"positional": 0.85, "inactive": 0.0015},
    "loose: positional 0.75, inactive 0.003": {"positional": 0.75, "inactive": 0.003},
}


def main() -> None:
    data = json.loads((RESULTS / "exp1_verification.json").read_text(encoding="utf-8"))
    scores = {c: np.asarray(data["scores"][c]) for c in CLASSES}
    confidence = {c: np.asarray(data["confidence"][c]) for c in CLASSES}
    layer_of = np.repeat(np.arange(6), 8)
    original = dict(headscore.THRESHOLDS)
    out: dict[str, object] = {"note": "post hoc, not preregistered", "variants": {}}
    variants: dict[str, object] = {}
    for name, change in VARIANTS.items():
        headscore.THRESHOLDS.update({**original, **change})
        rep = evaluate_predictions(confidence, scores, layer_of)
        micro = rep["micro_over_all_classes"]
        variants[name] = {
            "positives_per_class": {c: rep["per_class"][c]["positives"] for c in CLASSES},
            "micro_all_classes": {
                k: micro[k] for k in ("tp", "fp", "fn", "precision", "recall", "f1")
            },
            "within_layer_p": rep["permutation_within_layer"]["p_value"],
            "h1_met": rep["h1_met"],
        }
        print(name, variants[name]["micro_all_classes"], "p", variants[name]["within_layer_p"])
    headscore.THRESHOLDS.update(original)
    out["variants"] = variants
    # rank agreement inside the layers that were read exactly or nearly so, and outside them
    early = layer_of <= 1
    pos, conf = scores["positional"], confidence["positional"]
    out["positional_rank_agreement"] = {
        "layers_0_1": spearman(conf[early], pos[early]),
        "layers_2_5": spearman(conf[~early], pos[~early]),
    }
    out["margins"] = {
        "positional_true_positives": {
            data["heads"][i]: float(pos[i]) for i in np.flatnonzero((pos >= 0.8) & (conf >= 0.5))
        },
        "largest_previous_token_score": float(scores["previous_token"].max()),
        "largest_start_anchor_score": float(scores["start_anchor"].max()),
        "largest_induction_score": float(scores["induction"].max()),
        "largest_duplicate_token_score": float(scores["duplicate_token"].max()),
    }
    print(json.dumps({k: out[k] for k in ("positional_rank_agreement", "margins")}, indent=1))
    (RESULTS / "exp1_sensitivity.json").write_text(
        json.dumps(out, indent=1) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
