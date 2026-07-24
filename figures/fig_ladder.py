#!/usr/bin/env python3
"""fig_ladder.py -- ablation-ladder summary bars (validation IoU per rung).

Same palette as the rest of the report: glacier = blue, debris = orange.
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
    fig, ax = plt.subplots(figsize=(12.6, 4.9))
    fig.subplots_adjust(left=0.055, right=0.99, top=0.9, bottom=0.11)

    ax.axvspan(x[9] - 0.5, x[9] + 0.5, color="#f2f6fc", zorder=0)  # best rung f
    bg = ax.bar(x - w / 2, GLACIER, w, color=BLUE, zorder=3)
    bd = ax.bar(x + w / 2, DEBRIS, w, color=ORANGE, zorder=3)
    for rect, v in list(zip(bg, GLACIER)) + list(zip(bd, DEBRIS)):
        ax.annotate(f"{v:.2f}".lstrip("0") if v else "0",
                    (rect.get_x() + rect.get_width() / 2, v), xytext=(0, 4),
                    textcoords="offset points", ha="center", fontsize=9.5,
                    color=INK2)

    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.set_xticks(x, RUNGS)
    ax.set_ylabel("validation IoU")
    ax.set_ylim(0, 0.78)
    ax.margins(x=0.015)
    ax.annotate("best rung", (x[9], 0.755), ha="center", fontsize=10, color=MUTED)

    ax.legend([bg, bd], ["Glacier IoU", "Debris IoU"], loc="upper left",
              frameon=False, fontsize=11.5)
    fig.savefig(OUT, dpi=220)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
