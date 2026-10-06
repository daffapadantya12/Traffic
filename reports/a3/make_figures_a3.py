"""Assignment 3 result figures from the learning results (and, for the deck, at slide resolution).

  python3 reports/a3/make_figures_a3.py --learning reports/a3/learning
writes figures/models_taskA.png: macro-F1 and accuracy on ambulance ticks for every model and baseline on the
held-out seed, as two small multiples on one scale (0 to 1), baselines in grey.
"""
from __future__ import annotations

import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
BLUE, GREY, INK, INK2, GRID = "#2a78d6", "#b9b8b3", "#0b0b0b", "#52514e", "#dddcd8"
plt.rcParams.update({"font.size": 9, "font.family": "DejaVu Sans", "axes.edgecolor": GRID, "axes.labelcolor": INK2,
                     "xtick.color": INK2, "ytick.color": INK2, "axes.titlecolor": INK, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.titlesize": 9.5, "axes.titleweight": "bold"})
MODELS = {"logistic_regression": "Logistic regression", "decision_tree": "Decision tree", "random_forest": "Random forest",
          "gradient_boosting": "Gradient boosting", "neural_network": "Neural network"}
BASES = {"stillness": "Keep the current stage", "majority": "Most common stage", "random_uniform": "Random stage"}


def models_taskA(learning: str, out: str) -> str:
    A = json.load(open(os.path.join(learning, "results.json")))["task_a"]
    rows = [(MODELS.get(k, k), v, False) for k, v in sorted(A["models"].items(), key=lambda kv: kv[1]["macro_f1"])]
    rows = [(BASES[k], A["baselines"][k], True) for k in ("random_uniform", "majority", "stillness")] + rows
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.9), sharey=True)
    for ax, (key, title) in zip(axes, (("macro_f1", "Macro-F1 (every stage equal)"),
                                       ("ambulance_accuracy", "Accuracy, ambulance ticks"))):
        y = range(len(rows))
        vals = [r[1][key] for r in rows]
        ax.barh(list(y), vals, color=[GREY if r[2] else BLUE for r in rows], height=0.62, edgecolor="white", linewidth=1.5)
        for i, v in zip(y, vals):
            ax.text(v + 0.012, i, f"{v:.2f}", va="center", fontsize=8, color=INK)
        ax.set_xlim(0, 1.08)
        ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
        ax.xaxis.grid(True, color=GRID, linewidth=0.6)
        ax.set_axisbelow(True)
        ax.set_title(title, loc="left")
        ax.tick_params(axis="y", length=0)
    axes[0].set_yticks(list(range(len(rows))))
    axes[0].set_yticklabels([r[0] for r in rows])
    for lab, r in zip(axes[0].get_yticklabels(), rows):
        lab.set_color(INK2 if r[2] else INK)
    split = A["split"].split()[-1]
    fig.text(0.0, -0.04, f"Held-out seed {split}: {A['n_test']:,} ticks. Grey: baselines. Blue: trained models.", fontsize=8, color=INK2)
    fig.tight_layout()
    path = os.path.join(out, "models_taskA.png")
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return path


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--learning", default=os.path.join(HERE, "learning"))
    ap.add_argument("--out", default=os.path.join(HERE, "figures"))
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    print(models_taskA(a.learning, a.out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
