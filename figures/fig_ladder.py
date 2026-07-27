#!/usr/bin/env python3
"""fig_ladder.py -- ablation-ladder summary bars (validation IoU per rung).

Not just the bars: the four events the ladder exists to establish are called out
on the chart itself (terrain channels produce the first debris signal, Dice wakes
debris up, the isolation rung recovers d2's regression, the distance head wins),
so the figure carries Section 6's argument without the reader holding Table 2 in
their head. Same palette as the rest of the report: glacier blue, debris orange.

Output: report/latex/figures/fig_ladder.png
Usage:  python fig_ladder.py
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = Path(__file__).resolve().parent.parent / "latex" / "figures" / "fig_ladder.png"

RUNGS = ["a", "b1", "b2", "c", "d1", "d2", "d2b", "e1", "e2", "f", "final"]
GLACIER = [0.322, 0.438, 0.568, 0.627, 0.602, 0.596, 0.612, 0.584, 0.575, 0.665, 0.655]
DEBRIS = [0.000, 0.000, 0.066, 0.013, 0.171, 0.106, 0.196, 0.210, 0.190, 0.259, 0.234]

# one-line reminder of what each rung changed, printed under the rung letter
DELTA = ["visible\nbands", "+NIR/SWIR\n/thermal", "+elevation\n+slope", "+augment",
         "+Dice", "+focal, pos\nweight, ×3", "isolate:\n×3 only", "+attention",
         "+dropout", "+distance\nhead", "60-epoch\nrerun"]

# (rung index, text, y of the label, colour key)
CALLOUTS = [
    (2, "terrain channels give\nthe first debris signal", "debris"),
    (4, "Dice wakes\ndebris up", "debris"),
    (6, "isolation rung recovers\nd2's regression", "debris"),
    (9, "distance head:\nbest on both tasks", "glacier"),
]

BLUE, ORANGE = "#2a78d6", "#eb6834"
INK, INK2, MUTED, GRID, AXIS = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"

plt.rcParams.update({
    "font.family": "Segoe UI", "font.size": 11,
    "axes.edgecolor": AXIS, "axes.linewidth": 0.9,
    "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": MUTED,
    "figure.facecolor": "white", "axes.facecolor": "white",
})


def main():
    x = np.arange(len(RUNGS))
    w = 0.38
    fig, ax = plt.subplots(figsize=(12.6, 6.1))
    fig.subplots_adjust(left=0.055, right=0.995, top=0.87, bottom=0.20)

    ax.axvspan(x[9] - 0.5, x[9] + 0.5, color="#f2f6fc", zorder=0)      # best rung f
    bg = ax.bar(x - w / 2, GLACIER, w, color=BLUE, zorder=3)
    bd = ax.bar(x + w / 2, DEBRIS, w, color=ORANGE, zorder=3)
    for rect, v in list(zip(bg, GLACIER)) + list(zip(bd, DEBRIS)):
        ax.annotate(f"{v:.2f}".lstrip("0") if v else "0",
                    (rect.get_x() + rect.get_width() / 2, v), xytext=(0, 4),
                    textcoords="offset points", ha="center", fontsize=9.5,
                    color=INK2, zorder=4)

    # ---- callouts: leader line from just above the bar up to the note
    for i, text, key in CALLOUTS:
        col = BLUE if key == "glacier" else ORANGE
        top = max(GLACIER[i], DEBRIS[i])
        y = 0.90 if key == "glacier" else 0.80
        ax.annotate(text, xy=(x[i] + (w / 2 if key == "debris" else -w / 2), top + 0.035),
                    xytext=(x[i], y), ha="center", va="bottom", fontsize=10,
                    color=col, linespacing=1.35, zorder=6,
                    arrowprops=dict(arrowstyle="-", color=col, linewidth=0.9,
                                    alpha=0.55, shrinkA=2, shrinkB=3))

    ax.spines[["top", "right"]].set_visible(False)
    ax.spines["left"].set_bounds(0, 0.7)     # no bare spine in the callout band
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.set_xticks(x, RUNGS, fontsize=12, fontweight="semibold")
    ax.tick_params(axis="x", length=0, pad=6)
    for i, d in enumerate(DELTA):
        ax.annotate(d, (x[i], 0), xytext=(0, -34), textcoords="offset points",
                    ha="center", va="top", fontsize=8.8, color=MUTED,
                    linespacing=1.3, annotation_clip=False)
    ax.set_ylabel("validation IoU")
    ax.set_ylim(0, 1.02)
    ax.set_yticks(np.arange(0, 0.71, 0.1))
    ax.margins(x=0.012)

    ax.legend([bg, bd], ["Glacier IoU", "Debris IoU"], loc="upper left",
              bbox_to_anchor=(0.0, 1.06), frameon=False, fontsize=11.5, ncol=2)
    fig.savefig(OUT, dpi=220)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
