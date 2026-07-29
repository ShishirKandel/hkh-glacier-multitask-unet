#!/usr/bin/env python3
"""fig_collapse.py -- the all-background collapse and the initialisation fix.

Section 4.3 describes the project's sharpest diagnostic result in prose: the first
complete run drove its training loss down while validation glacier IoU sat at
exactly zero, and prior-logit head initialisation turned that identical
configuration into a run that learns.

The figure is built to make the diagnosis itself visible, not just the outcome.
Both runs are overlaid on BOTH panels, so the reader sees the two quantities
disagree: on the training loss the collapsed run looks like it is learning
normally, and only the validation metric exposes it. That is precisely why the
bug survived to a complete run, and it is the transferable lesson of Section 4.3.

An earlier draft put one run per panel with the loss on a twin axis. It was
discarded: a twin axis invites the reader to compare a loss against an IoU, which
is the exact confusion the figure exists to dispel.

Read straight from the shipped run logs, never retyped:
  analysis/runs_kaggle/a_v2  -- unweighted BCE, lr 1e-3, zero-bias heads (collapse)
  analysis/runs_kaggle/a_v3  -- prior-logit init at p=0.08/0.10, lr 3e-4 (fixed)

Output: report/latex/figures/fig_collapse.png
Usage:  python fig_collapse.py
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

TOOLS = Path(__file__).resolve().parent
ROOT = TOOLS.parents[1]
RUNS = ROOT / "analysis" / "runs_kaggle"
OUT = TOOLS.parent / "latex" / "figures" / "fig_collapse.png"

BLUE, ORANGE = "#2a78d6", "#eb6834"
INK, INK2, MUTED, GRID, AXIS = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"

# The collapsed run is grey, not orange: orange means debris everywhere else in
# the report, and reusing it for "failed run" would break the palette's meaning.
COLLAPSE, FIXED = "#8d8b85", BLUE
XMAX = 28.5


plt.rcParams.update({
    "font.family": "Segoe UI", "font.size": 11,
    "axes.edgecolor": AXIS, "axes.linewidth": 0.9,
    "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": MUTED,
    "figure.facecolor": "white", "axes.facecolor": "white",
})


def metrics_path(run):
    """The run log, under either layout.

    The working tree keeps the Kaggle artefact tree intact
    (<rung>/runs/a/a/metrics.jsonl); the published repository flattens it to
    <rung>/metrics.jsonl. Accepting both is what lets this script run from a
    clone, which is the only reason to publish it."""
    for candidate in (RUNS / run / "runs" / "a" / "a" / "metrics.jsonl",
                      RUNS / run / "metrics.jsonl"):
        if candidate.is_file():
            return candidate
    raise SystemExit("fig_collapse: no metrics.jsonl for run %r under %s" % (run, RUNS))


def series(run):
    """(epochs, validation glacier IoU, training glacier BCE) from one run log.

    The first record of a metrics.jsonl is a run header with no 'epoch' key, so
    the epoch rows are selected rather than the file simply being sliced."""
    rows = [json.loads(line) for line in
            metrics_path(run).read_text(encoding="utf-8").splitlines() if line.strip()]
    rows = [r for r in rows if "epoch" in r]
    return ([r["epoch"] for r in rows],
            [r["val"]["glacier_iou"] for r in rows],
            [r["train"]["glacier_bce"] for r in rows])


def style(ax, ylabel, ytop, yticks):
    ax.set_xlim(0, XMAX)
    ax.set_ylim(0, ytop)
    ax.set_yticks(yticks)
    ax.set_xlabel("epoch")
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines["left"].set_bounds(0, yticks[-1])


def main():
    ep2, iou2, bce2 = series("a_v2")
    ep3, iou3, bce3 = series("a_v3")

    fig, (axl, axr) = plt.subplots(1, 2, figsize=(12.6, 4.3))
    fig.subplots_adjust(left=0.055, right=0.988, top=0.80, bottom=0.135, wspace=0.20)

    # ---- (a) what the model is actually judged on
    axl.plot(ep3, iou3, color=FIXED, linewidth=2.3, zorder=4)
    axl.plot(ep2, iou2, color=COLLAPSE, linewidth=2.3, zorder=3)
    axl.plot(ep2, iou2, "o", color=COLLAPSE, markersize=3.2, zorder=3)
    style(axl, "validation glacier IoU", 0.60, [0, 0.1, 0.2, 0.3, 0.4, 0.5])
    axl.set_title("(a) the metric that matters", loc="left", fontsize=12.5,
                  fontweight="semibold", color=INK, pad=16)
    axl.annotate("prior-logit init, lr 3e-4:\nescapes at epoch 6, peaks 0.463",
                 (ep3[5], iou3[5]), xytext=(9.3, 0.585), ha="left", va="top",
                 fontsize=10.5, color=FIXED, linespacing=1.35,
                 arrowprops=dict(arrowstyle="-", color=FIXED, linewidth=0.9,
                                 alpha=0.55, shrinkA=2, shrinkB=4))
    # Parked out along the zero line rather than above it: every position over the
    # collapsed trace is crossed by the fixed run's climb.
    axl.annotate("zero-bias heads, lr 1e-3:\nexactly 0.000 from epoch 4",
                 (7.0, 0.0), xytext=(12.0, 0.012), ha="left", va="bottom",
                 fontsize=10.5, color=ORANGE, linespacing=1.35,
                 arrowprops=dict(arrowstyle="-", color=ORANGE, linewidth=0.9,
                                 alpha=0.55, shrinkA=2, shrinkB=4))

    # ---- (b) what the training loop reports, which says nothing is wrong
    axr.plot(ep3, bce3, color=FIXED, linewidth=2.3, zorder=4)
    axr.plot(ep2, bce2, color=COLLAPSE, linewidth=2.3, zorder=3)
    axr.plot(ep2, bce2, "o", color=COLLAPSE, markersize=3.2, zorder=3)
    style(axr, "training glacier BCE", 0.60, [0, 0.1, 0.2, 0.3, 0.4, 0.5])
    axr.set_title("(b) the signal that hid it", loc="left", fontsize=12.5,
                  fontweight="semibold", color=INK, pad=16)
    axr.annotate("the collapsed run's loss falls\n0.454 to 0.247, a textbook curve",
                 (ep2[3], bce2[3]), xytext=(7.4, 0.505), ha="left", va="top",
                 fontsize=10.5, color=ORANGE, linespacing=1.35,
                 arrowprops=dict(arrowstyle="-", color=ORANGE, linewidth=0.9,
                                 alpha=0.55, shrinkA=2, shrinkB=4))
    # Point at the collapsed run's LAST loss value, not the fixed run's: the claim
    # is that the two are indistinguishable while both were still running.
    axr.annotate("and lands in the same band\nas the run that works",
                 (ep2[-1], bce2[-1]), xytext=(14.0, 0.115), ha="left", va="bottom",
                 fontsize=10.5, color=MUTED, linespacing=1.35,
                 arrowprops=dict(arrowstyle="-", color=MUTED, linewidth=0.9,
                                 alpha=0.5, shrinkA=2, shrinkB=4))

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=220)
    plt.close(fig)
    print("fig_collapse: wrote %s" % OUT)


if __name__ == "__main__":
    main()
