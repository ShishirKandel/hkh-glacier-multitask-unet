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
import numpy as np

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
    style(axB, "(b)  Validation IoU, final run", ylab="IoU")
    axB.set_ylim(0, 0.78)                                  # before the annotation,
    axB.annotate("selected checkpoint\n(epoch 32)", (best, 0.775),   # which anchors to it
                 xytext=(-7, -2), textcoords="offset points", fontsize=10,
                 color=MUTED, ha="right", va="top", linespacing=1.4)

    # ---------------- (c, d) per-rung trajectories
    # Raw per-epoch validation IoU on this data is genuinely noisy (the validation
    # zone holds only 118 debris-bearing patches). Plotting five raw traces on one
    # axis is unreadable, so each rung is drawn twice: the raw trace faint, and the
    # 3-evaluation running mean bold. That mean is not cosmetic smoothing, it is
    # exactly the quantity Section 4.5's checkpoint rule selects on.
    def smooth3(ys):
        return [float(np.mean(ys[max(0, i - 2):i + 1])) for i in range(len(ys))]

    def trajectories(ax, rungs, cols, metric, title, ylab, ylim):
        labels = []
        for rung, col in zip(rungs, cols):
            rows = load(rung)
            xs = [r["epoch"] for r in rows]
            ys = [r["val"][metric] for r in rows]
            sm = smooth3(ys)
            ax.plot(xs, ys, color=col, linewidth=0.9, alpha=0.30, zorder=2)
            ax.plot(xs, sm, color=col, linewidth=2.8 if rung == "f" else 2.0, zorder=3)
            labels.append((xs[-1], sm[-1], rung, col))
        style(ax, title, ylab=ylab)
        ax.set_ylim(*ylim)
        endlabels(ax, labels)

    trajectories(axC, ["a", "b1", "b2", "c", "f"], BLUE_ORD, "glacier_iou",
                 "(c)  Validation glacier IoU across ladder rungs",
                 "glacier IoU", (0, 0.72))
    trajectories(axD, ["d1", "d2b", "e1", "f"], ORANGE_ORD, "debris_iou",
                 "(d)  Validation debris IoU across ladder rungs",
                 "debris IoU", (0, 0.345))
    axD.spines["left"].set_bounds(0, 0.30)   # headroom for the note, no bare spine
    axD.annotate("bold = 3-evaluation running mean, the quantity\n"
                 "checkpoint selection scores; faint = raw",
                 xy=(0.03, 0.97), xycoords="axes fraction", fontsize=9.5,
                 color=MUTED, va="top", linespacing=1.4)

    fig.savefig(OUT, dpi=220)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
