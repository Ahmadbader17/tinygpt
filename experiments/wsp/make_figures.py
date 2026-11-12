"""Regenerate every figure and table of the report from saved result files only.

Reads `experiments/wsp/results/*.json`. Runs no model and reads no checkpoint or story.

    uv run python experiments/wsp/make_figures.py

Writes `docs/img/wsp_*.png` and `experiments/wsp/results/tables.md`.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[2]
RESULTS = REPO / "experiments" / "wsp" / "results"
FIGURES = REPO / "docs" / "img"
LOSS_TOLERANCE = 0.02  # nats, PLAN.md section 7
GREY, BLUE, ORANGE, GREEN, RED = "#7f7f7f", "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"
ATTEMPT_LABEL = {"b10": "edit, 10 passes", "b25": "edit, 25 passes", "b50": "edit, 50 passes"}
ATTEMPT_COLOUR = {"b10": "#1f77b4", "b25": "#9467bd", "b50": "#2ca02c"}
FAMILY_LABEL = {
    "head_scale": "head scaled up",
    "rank_one": "rank-one addition",
    "head_zero": "head zeroed",
    "ln_gain": "LayerNorm gain",
    "noise": "noise",
}
Interval = dict[str, float]


def load(results: Path, name: str) -> Any:
    return json.loads((results / name).read_text(encoding="utf-8"))


def err(ci: Interval) -> list[list[float]]:
    """Asymmetric error bar (below, above) of an interval, for matplotlib."""
    return [[ci["estimate"] - ci["low"]], [ci["high"] - ci["estimate"]]]


def fmt(ci: Interval, digits: int = 3, signed: bool = False) -> str:
    sign = "+" if signed else ""
    return (
        f"{ci['estimate']:{sign}.{digits}f} "
        f"[{ci['low']:{sign}.{digits}f}, {ci['high']:{sign}.{digits}f}]"
    )


def yes(flag: bool) -> str:
    return "yes" if flag else "no"


# ---------------------------------------------------------------- Exp 1


def figure_exp1(ver: dict[str, Any], out: Path) -> None:
    """Confidence given from the weights against the measured score, two non-empty classes."""
    layers = [int(h.split(".")[0]) for h in ver["heads"]]
    thresholds = ver["report"]["thresholds"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    panels = (
        ("positional", "measured positional score (pattern cosine across texts)", False),
        ("inactive", "measured loss change when the head is zeroed (nats)", True),
    )
    for ax, (cls, xlabel, log) in zip(axes, panels, strict=True):
        scores = [abs(s) for s in ver["scores"][cls]]
        points = ax.scatter(
            scores, ver["confidence"][cls], c=layers, cmap="viridis", s=36, edgecolor="k", lw=0.4
        )
        if log:
            ax.set_xscale("log")
        ax.axvline(thresholds[cls], color=RED, ls="--", lw=1, label="class threshold")
        ax.axhline(0.5, color=GREY, ls=":", lw=1, label="predicted positive above")
        ax.set_xlabel(xlabel)
        ax.set_ylabel("confidence from weights alone")
        ax.set_ylim(-0.03, 1.03)
        side = "left of" if cls == "inactive" else "right of"
        ax.set_title(f"{cls}: in class {side} the red line")
        ax.legend(fontsize=8, loc="upper right" if cls == "inactive" else "upper left")
        fig.colorbar(points, ax=ax, label="layer")
    fig.suptitle("Exp 1: 48 heads, zero forward passes before the prediction", fontsize=11)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def table_exp1(ver: dict[str, Any]) -> str:
    rep = ver["report"]
    rows = [
        "| Class | Heads in class | Predicted | Hits | False alarms | Missed | Precision | "
        "Recall | Chance precision | Spearman (confidence, score) |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for cls, r in rep["per_class"].items():
        empty = r["positives"] == 0 and r["predicted"] == 0
        precision = "n/a" if r["predicted"] == 0 else f"{r['precision']:.2f}"
        recall = "n/a" if r["positives"] == 0 else f"{r['recall']:.2f}"
        rows.append(
            f"| {cls.replace('_', ' ')} | {r['positives']} | {r['predicted']} | {r['tp']} | "
            f"{r['fp']} | {r['fn']} | {precision} | {recall} | "
            f"{'n/a' if empty else format(r['chance_precision'], '.2f')} | "
            f"{r['spearman']['rho']:+.2f} |"
        )
    micro = rep["micro_over_classes_with_positives"]
    glob, within = rep["permutation_global"], rep["permutation_within_layer"]
    rows += [
        "",
        f"Micro-average over classes with positives: precision {micro['precision']:.3f}, "
        f"recall {micro['recall']:.3f}, F1 {micro['f1']:.3f}.",
        f"Permutation null, heads shuffled globally: mean F1 {glob['null_mean_f1']:.2f}, "
        f"p = {glob['p_value']:.4f}. Shuffled within layer: mean F1 "
        f"{within['null_mean_f1']:.2f}, 95th percentile {within['null_95th_f1']:.2f}, "
        f"p = {within['p_value']:.4f}. H1 met: {yes(rep['h1_met'])}.",
    ]
    return "\n".join(rows)


# ---------------------------------------------------------------- Exp 2


def figure_exp2(exp2: dict[str, Any], out: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.5, 4.2))
    labels = []
    for x, (name, case) in enumerate(exp2["cases"].items()):
        family = FAMILY_LABEL.get(case["fault"]["family"], case["fault"]["family"])
        labels.append(f"{name}\n{family}\n(+{case['damage']:.2f} nats)")
        if case["recovery"] is None:
            ax.text(x, 0.03, "no repair", ha="center", fontsize=8, color=GREY)
            continue
        controls = [c["recovery"] for c in case["controls"].values() if c["kind"] == "gauss"]
        ax.scatter(
            [x + 0.18] * len(controls),
            controls,
            color=GREY,
            s=22,
            marker="x",
            label="random edit, same support and norm" if x == 0 else None,
        )
        ax.errorbar(
            x,
            case["recovery"]["estimate"],
            yerr=err(case["recovery"]),
            fmt="o",
            color=BLUE if case["tensor_correct"] else ORANGE,
            capsize=4,
        )
    ax.errorbar([], [], fmt="o", color=BLUE, label="repair, right tensor named (95% interval)")
    ax.errorbar([], [], fmt="o", color=ORANGE, label="repair, wrong tensor named")
    ax.axhline(0, color="k", lw=0.8)
    ax.axhline(1, color=GREEN, ls="--", lw=1, label="full recovery (clean model)")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=8)
    low = min(-0.5, ax.get_ylim()[0])
    ax.set_ylim(max(low, -3.0), 1.25)
    ax.set_ylabel("share of lost test loss recovered")
    ax.set_title("Exp 2: blind repair of planted faults, sealed test stories")
    ax.legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def table_exp2(exp2: dict[str, Any]) -> str:
    rows = [
        "| Checkpoint | Planted fault | Tensor planted | Tensor named | Tensor right | "
        "Sub-location right | Damage (nats) | Recovery [95%] | Best random control | "
        "Beats all controls | Cosine with true inverse | Passes | Repair counted |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for name, c in exp2["cases"].items():
        fault = c["fault"]
        sub = c.get("sublocation_correct")
        controls = [k["recovery"] for k in c.get("controls", {}).values() if k["kind"] == "gauss"]
        cosine = c.get("cosine_with_true_inverse")
        rows.append(
            f"| {name} | {FAMILY_LABEL.get(fault['family'], fault['family'])} | "
            f"`{fault['tensor']}` | `{c['diagnosis'].get('tensor', 'none')}` | "
            f"{yes(c['tensor_correct'])} | {'n/a' if sub is None else yes(sub)} | "
            f"{c['damage']:+.3f} | {fmt(c['recovery'], 2) if c['recovery'] else 'no repair'} | "
            f"{format(max(controls), '+.2f') if controls else 'n/a'} | "
            f"{yes(c.get('beats_all_controls', False))} | "
            f"{'n/a' if cosine is None else format(cosine, '.3f')} | {c['passes_used']} | "
            f"{yes(c['repair_succeeded'])} |"
        )
    mean_all = exp2["mean_recovery_counting_no_repair_as_zero"]
    rows += [
        "",
        f"Clean test loss {exp2['clean_loss']:.4f}. Tensors located: "
        f"{exp2['tensors_located']} of 5 (H2a met: {yes(exp2['h2a_met'])}). Repairs that "
        f"count: {exp2['repairs_succeeded']} of 5 (H2b met: {yes(exp2['h2b_met'])}). Mean "
        f"recovery {mean_all:.2f} (a missing repair counted as zero).",
    ]
    return "\n".join(rows)


# ---------------------------------------------------------------- Exp 3


def exp3_models(exp3: dict[str, Any]) -> list[tuple[str, str, str]]:
    """(key, label, colour) of every evaluated model, in display order."""
    models = [("unedited", "unedited", GREY), ("repetition_penalty", "penalty 1.2", ORANGE)]
    models += [(k, ATTEMPT_LABEL[k], BLUE) for k in ATTEMPT_LABEL if k in exp3["models"]]
    return models


def figure_exp3_headline(exp3: dict[str, Any], out: Path) -> None:
    """Loop rate and rep-4 per model with the real-text level, and the paired loss change."""
    models = exp3_models(exp3)
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))
    for ax, metric, title in (
        (axes[0], "loop_rate", "loop rate (a 4-gram three or more times)"),
        (axes[1], "rep4", "rep-4 (repeated 4-gram fraction)"),
    ):
        for x, (key, _, colour) in enumerate(models):
            for decoding, marker, shift in (("sample", "o", -0.12), ("greedy", "s", 0.12)):
                ci = exp3["models"][key][decoding][metric]
                ax.errorbar(
                    x + shift,
                    ci["estimate"],
                    yerr=err(ci),
                    fmt=marker,
                    color=colour,
                    capsize=3,
                    mfc=colour if decoding == "sample" else "white",
                )
        real = exp3["real_text"][metric]
        ax.axhspan(real["low"], real["high"], color=GREEN, alpha=0.15)
        ax.axhline(real["estimate"], color=GREEN, ls="--", lw=1, label="real stories")
        ax.errorbar([], [], fmt="o", color="k", label="sampling (temp 0.8, top-k 40)")
        ax.errorbar([], [], fmt="s", color="k", mfc="white", label="greedy")
        ax.set_xticks(range(len(models)))
        ax.set_xticklabels([label for _, label, _ in models], rotation=20, fontsize=8)
        ax.set_ylabel(title)
        ax.set_ylim(bottom=0)
        ax.legend(fontsize=7)
    ax = axes[2]
    shown = [m for m in models if m[0] != "unedited"]
    for x, (key, _, colour) in enumerate(shown):
        ci = exp3["versus_unedited"][key]["sample"]["loss_diff"]
        ax.errorbar(x, ci["estimate"], yerr=err(ci), fmt="o", color=colour, capsize=3)
    ax.axhline(0, color="k", lw=0.8)
    ax.axhline(LOSS_TOLERANCE, color=RED, ls="--", lw=1, label="tolerance +0.02 nats")
    ax.set_xticks(range(len(shown)))
    ax.set_xticklabels([label for _, label, _ in shown], rotation=20, fontsize=8)
    ax.set_ylabel("test loss minus unedited (nats per token)")
    ax.set_yscale("symlog", linthresh=0.03)
    ax.legend(fontsize=7)
    fig.suptitle(
        "Exp 3: repetition and loss on sealed prompts and stories (95% intervals)", fontsize=11
    )
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def figure_exp3_tradeoff(exp3: dict[str, Any], out: Path) -> None:
    """Loop-rate change against loss change: edits, matched random edits, and the penalty."""
    fig, ax = plt.subplots(figsize=(7, 4.6))
    seen_control = False
    for key, attempt in exp3["attempts"].items():
        if not attempt["edited"]:
            continue
        for control in attempt["controls"].values():
            diff = control["versus_unedited"]["sample"]
            ax.scatter(
                diff["loss_diff"]["estimate"],
                diff["loop_rate"]["estimate"],
                color=GREY,
                marker="x" if control["kind"] == "gauss" else "+",
                s=28,
                label=None if seen_control else "matched random edits",
            )
            seen_control = True
        diff = exp3["versus_unedited"][key]["sample"]
        ax.errorbar(
            diff["loss_diff"]["estimate"],
            diff["loop_rate"]["estimate"],
            xerr=err(diff["loss_diff"]),
            yerr=err(diff["loop_rate"]),
            fmt="o",
            color=ATTEMPT_COLOUR[key],
            capsize=3,
            label=ATTEMPT_LABEL[key],
        )
    diff = exp3["versus_unedited"]["repetition_penalty"]["sample"]
    ax.errorbar(
        diff["loss_diff"]["estimate"],
        diff["loop_rate"]["estimate"],
        xerr=err(diff["loss_diff"]),
        yerr=err(diff["loop_rate"]),
        fmt="D",
        color=ORANGE,
        capsize=3,
        label="repetition penalty 1.2 (decoding)",
    )
    ax.axhline(0, color="k", lw=0.8)
    ax.axvline(0, color="k", lw=0.8)
    ax.axvline(LOSS_TOLERANCE, color=RED, ls="--", lw=1, label="loss tolerance")
    ax.set_xscale("symlog", linthresh=0.03)
    ax.set_xlabel("test loss minus unedited (nats per token)")
    ax.set_ylabel("loop rate minus unedited (sampling)")
    ax.set_title("Exp 3: success needs a point below zero and left of the red line")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def table_exp3(exp3: dict[str, Any]) -> str:
    rows = [
        "| Model | Decoder | Loop rate | Long-loop rate | rep-2 | rep-3 | rep-4 | distinct-2 | "
        "Mean length | Ended by itself | Judge loss | Test loss | Perplexity |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    real = exp3["real_text"]
    rows.append(
        f"| real stories | n/a | {fmt(real['loop_rate'])} | {fmt(real['long_loop_rate'])} | "
        f"{real['rep2']['estimate']:.3f} | {real['rep3']['estimate']:.3f} | "
        f"{real['rep4']['estimate']:.4f} | {real['distinct2']:.3f} | "
        f"{real['length']['estimate']:.1f} | n/a | n/a | n/a | n/a |"
    )
    for key, label, _ in exp3_models(exp3):
        model = exp3["models"][key]
        for decoding in ("sample", "greedy"):
            g = model[decoding]
            judge = fmt(g["judge_loss"]) if "judge_loss" in g else "n/a"
            rows.append(
                f"| {label} | {decoding} | {fmt(g['loop_rate'])} | {fmt(g['long_loop_rate'])} | "
                f"{g['rep2']['estimate']:.3f} | {g['rep3']['estimate']:.3f} | "
                f"{g['rep4']['estimate']:.4f} | {g['distinct2']:.3f} | "
                f"{g['length']['estimate']:.1f} | {g['eot_rate']:.2f} | {judge} | "
                f"{fmt(model['loss']['loss'], 4)} | {model['loss']['perplexity']:.3f} |"
            )
    rows += [
        "",
        "Paired differences from the unedited model (same prompts, seeds and stories; 95%):",
        "",
        "| Model | Decoder | Loop rate | rep-4 | Long-loop rate | Length ratio | Test loss |",
        "|---|---|---|---|---|---|---|",
    ]
    for key, label, _ in exp3_models(exp3):
        if key == "unedited":
            continue
        for decoding in ("sample", "greedy"):
            d = exp3["versus_unedited"][key][decoding]
            rows.append(
                f"| {label} | {decoding} | {fmt(d['loop_rate'], 3, True)} | "
                f"{fmt(d['rep4'], 4, True)} | {fmt(d['long_loop_rate'], 3, True)} | "
                f"{d['length_ratio']:.3f} | {fmt(d['loss_diff'], 4, True)} |"
            )
    rows += [
        "",
        "Preregistered rule (sampling decoder):",
        "",
        "| Attempt | Passes used / budget | Edited | Loop rate falls | rep-4 falls | "
        "Loss within +0.02 | Length kept | All (95%) | All (98.3%, headline) | "
        "Random-control loop-rate drops | Beats all controls |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for key, a in exp3["attempts"].items():
        c95, c983 = a["criteria"]["95"], a["criteria"]["98.3"]
        if not a["edited"]:
            rows.append(
                f"| {ATTEMPT_LABEL[key]} | {a['passes_used']} / {a['budget']} | no (editor "
                "declared the unedited model final) | n/a | n/a | n/a | n/a | no | no | n/a | n/a |"
            )
            continue
        drops = "; ".join(
            f"{kind}: " + ", ".join(f"{v:+.3f}" for v in values)
            for kind, values in a["control_loop_rate_drops"].items()
        )
        beats = ", ".join(f"{kind}: {yes(v)}" for kind, v in a["specific"].items())
        rows.append(
            f"| {ATTEMPT_LABEL[key]} | {a['passes_used']} / {a['budget']} | yes | "
            f"{yes(c95['loop_rate_falls'])} | {yes(c95['rep4_falls'])} | "
            f"{yes(c95['loss_within_tolerance'])} | {yes(c95['length_kept'])} | "
            f"{yes(c95['all'])} | {yes(c983['all'])} | {drops} (edit: "
            f"{a['loop_rate_drop']:+.3f}) | {beats} |"
        )
    pen = exp3["penalty_under_the_same_rule"]
    rows += [
        "",
        f"Headline H3 met: {yes(exp3['headline_h3_met'])}. The repetition penalty under the "
        f"same rule: loop rate falls {yes(pen['loop_rate_falls'])}, rep-4 falls "
        f"{yes(pen['rep4_falls'])}, loss within tolerance {yes(pen['loss_within_tolerance'])}, "
        f"length kept {yes(pen['length_kept'])}, all {yes(pen['all'])}.",
    ]
    return "\n".join(rows)


def table_samples(generations: dict[str, Any], count: int = 5) -> str:
    """The first generation (seed 0) of the first prompts, for every model: no picking."""
    names = [n for n in ("unedited", "repetition_penalty", *ATTEMPT_LABEL) if n in generations]
    rows = []
    for p in range(min(count, len(generations["unedited"]["sample"]))):
        prompt = generations["unedited"]["sample"][p]["prompt"]
        rows.append(f"**Sealed prompt {p + 1}:** {prompt}")
        rows.append("")
        for name in names:
            text = generations[name]["sample"][p]["text"].replace("\n", " ").strip()
            rows.append(f"- *{name}:* {text}")
        rows.append("")
    return "\n".join(rows)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    parser.add_argument("--results", type=Path, default=RESULTS)
    parser.add_argument("--figures", type=Path, default=FIGURES)
    args = parser.parse_args(argv)
    args.figures.mkdir(parents=True, exist_ok=True)
    sections = []

    ver = load(args.results, "exp1_verification.json")
    figure_exp1(ver, args.figures / "wsp_exp1_heads.png")
    sections.append("## Exp 1: reading heads from weights\n\n" + table_exp1(ver))

    if (args.results / "final_eval.json").exists():
        final = load(args.results, "final_eval.json")
        figure_exp2(final["exp2"], args.figures / "wsp_exp2_recovery.png")
        figure_exp3_headline(final["exp3"], args.figures / "wsp_exp3_headline.png")
        figure_exp3_tradeoff(final["exp3"], args.figures / "wsp_exp3_tradeoff.png")
        sealed = final["sealed"]
        sections.append(
            "## Sealed test data\n\n"
            f"{sealed['stories_scored']} stories ({sealed['tokens_scored']} tokens) scored, "
            f"{sealed['prompts']} prompts; dropped: {json.dumps(sealed['dropped'])}. "
            f"Audit triggers: {final['audit_triggers'] or 'none'}."
        )
        sections.append("## Exp 2: blind repair\n\n" + table_exp2(final["exp2"]))
        sections.append("## Exp 3: reducing repetition\n\n" + table_exp3(final["exp3"]))
        generations = load(args.results, "final_generations.json")
        sections.append(
            "## Exp 3: first generation of the first five sealed prompts (seed 0)\n\n"
            + table_samples(generations)
        )
    text = "# Tables (generated by make_figures.py from result files; do not edit)\n\n"
    (args.results / "tables.md").write_text(text + "\n\n".join(sections) + "\n", encoding="utf-8")
    print(f"wrote {args.results / 'tables.md'} and figures under {args.figures}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
