#!/usr/bin/env python3
"""fig_training_curves.py -- training-dynamics figure from the Kaggle metrics logs.

Four panels from analysis/runs_kaggle/*/runs/*/*/metrics.jsonl:
  (a) per-head training loss vs epoch, final 60-epoch run (colour = head, dash = Dice)
  (b) validation glacier/debris IoU vs epoch, final run, best checkpoint marked
  (c) validation glacier IoU per epoch across ladder rungs a, b1, b2, c, f
  (d) validation debris IoU per epoch across rungs d1, d2b, e1, f

Output: report/latex/figures/fig_training_curves.png
Usage:  python fig_training_curves.py
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent.parent   # Final Assignment/
RUNS = ROOT / "analysis" / "runs_kaggle"
OUT = ROOT / "report" / "latex" / "figures" / "fig_training_curves.png"

# rung -> run directory (a_v3 / b1_v5 are the corrected re-runs; see 05_experiment_log)
RUN_DIR = {"a": "a_v3", "b1": "b1_v5", "b2": "b2", "c": "c", "d1": "d1",
           "d2b": "d2b", "e1": "e1", "f": "f", "final": "final"}

INK, INK2, MUTED, GRID, AXIS = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"          # head identities
BLUE_ORD = ["#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#0d366b"]   # ordinal, light->dark
ORANGE_ORD = ["#f6b795", "#ef8354", "#d95926", "#8f3410"]

plt.rcParams.update({
    "font.family": "Segoe UI", "font.size": 11,
    "axes.edgecolor": AXIS, "axes.linewidth": 0.9,
    "axes.labelcolor": INK2, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.titlecolor": INK, "figure.facecolor": "white", "axes.facecolor": "white",
})


def load(rung):
    p = RUNS / RUN_DIR[rung] / "runs" / rung.split("_")[0] / rung / "metrics.jsonl"
    if not p.exists():                       # layout is runs/<cfg>/<cfg>/
        p = next((RUNS / RUN_DIR[rung]).glob("runs/*/*/metrics.jsonl"))
    rows = [json.loads(l) for l in p.read_text().splitlines() if '"epoch"' in l]
    return rows


def style(ax, title, xlab="epoch", ylab=None):
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.set_title(title, loc="left", fontsize=12.5, fontweight="semibold", pad=10)
    ax.set_xlabel(xlab)
    if ylab:
        ax.set_ylabel(ylab)
    ax.margins(x=0.02)


def endlabels(ax, entries, min_gap_px=17):
    """Direct line labels at (x, y) with vertical dodging so none overlap.
    entries: list of (x, y, label, color)."""
    fig = ax.figure
    fig.canvas.draw()  # realise the transform
    disp = [(ax.transData.transform((x, y))[1], x, y, s, c) for x, y, s, c in entries]
    disp.sort(reverse=True)  # top of panel first
    placed = []
    for py, x, y, s, c in disp:
        if placed and placed[-1] - py < min_gap_px:
            py = placed[-1] - min_gap_px
        placed.append(py)
        y_adj = ax.transData.inverted().transform((0, py))[1]
        ax.annotate(s, (x, y_adj), xytext=(6, 0), textcoords="offset points",
                    fontsize=10.5, color=c, fontweight="semibold", va="center",
                    annotation_clip=False)


def main():
    fin = load("final")
    ep = [r["epoch"] for r in fin]

    fig, axes = plt.subplots(2, 2, figsize=(12.8, 8.6))
    (axA, axB), (axC, axD) = axes
    fig.subplots_adjust(hspace=0.52, wspace=0.24, left=0.06, right=0.9,
                        top=0.93, bottom=0.08)

    # ---------------- (a) training loss per head, final run
    spec = [("glacier_bce", BLUE, "-", "glacier BCE"),
            ("glacier_dice", BLUE, (0, (5, 3)), "glacier Dice"),
            ("type_bce", ORANGE, "-", "type BCE"),
            ("type_dice", ORANGE, (0, (5, 3)), "type Dice"),
            ("dist_l1", AQUA, "-", "distance L1")]
    labelsA = []
    for key, col, ls, lab in spec:
        ys = [r["train"][key] for r in fin]
        axA.plot(ep, ys, color=col, linestyle=ls, linewidth=2)
        labelsA.append((ep[-1], ys[-1], lab, col))
    style(axA, "(a)  Training loss per task head, final run", ylab="loss")
    axA.set_ylim(0, None)
    endlabels(axA, labelsA)

    # ---------------- (b) validation IoU, final run
    g = [r["val"]["glacier_iou"] for r in fin]
    d = [r["val"]["debris_iou"] for r in fin]
    best = 32
    axB.axvline(best, color=AXIS, linewidth=1.2, linestyle=(0, (4, 4)))
    axB.plot(ep, g, color=BLUE, linewidth=2.2)
    axB.plot(ep, d, color=ORANGE, linewidth=2.2)
    endlabels(axB, [(ep[-1], g[-1], "glacier IoU", BLUE),
                    (ep[-1], d[-1], "debris IoU", ORANGE)])
    axB.scatter([best, best], [g[best - 1], d[best - 1]], s=34, zorder=5,
                color=[BLUE, ORANGE], edgecolor="white", linewidth=1.2)
    axB.annotate("best checkpoint (epoch 32)", (best, axB.get_ylim()[1]),
                 xytext=(-6, -2), textcoords="offset points", fontsize=10,
                 color=MUTED, ha="right", va="top")
    style(axB, "(b)  Validation IoU, final run", ylab="IoU")
    axB.set_ylim(0, 0.78)

    # ---------------- (c) glacier IoU across ladder rungs
    labelsC = []
    for rung, col in zip(["a", "b1", "b2", "c", "f"], BLUE_ORD):
        rows = load(rung)
        lw = 2.6 if rung == "f" else 1.8
        xs = [r["epoch"] for r in rows]
        ys = [r["val"]["glacier_iou"] for r in rows]
        axC.plot(xs, ys, color=col, linewidth=lw)
        labelsC.append((xs[-1], ys[-1], rung, col))
    style(axC, "(c)  Validation glacier IoU across ladder rungs", ylab="glacier IoU")
    axC.set_ylim(0, 0.72)
    endlabels(axC, labelsC)

    # ---------------- (d) debris IoU across ladder rungs
    labelsD = []
    for rung, col in zip(["d1", "d2b", "e1", "f"], ORANGE_ORD):
        rows = load(rung)
        lw = 2.6 if rung == "f" else 1.8
        xs = [r["epoch"] for r in rows]
        ys = [r["val"]["debris_iou"] for r in rows]
        axD.plot(xs, ys, color=col, linewidth=lw)
        labelsD.append((xs[-1], ys[-1], rung, col))
    style(axD, "(d)  Validation debris IoU across ladder rungs", ylab="debris IoU")
    axD.set_ylim(0, None)
    endlabels(axD, labelsD)

    fig.savefig(OUT, dpi=220)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
