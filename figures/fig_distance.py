#!/usr/bin/env python3
"""fig_distance.py -- what the boundary-distance head is asked to learn, and what it learns.

Task (iii) is a third of this project's stated contribution and the modification
that wins rung (f), yet it is the one task the report never shows: Figure 1 plots
the input channels and the labels, Figure 7 plots the glacier and debris
predictions, and neither contains a distance field. This appendix figure closes
that, for one validation patch:

  (a) the expert glacier label, with the boundary the distance is measured from;
  (b) the target field, the truncated signed distance shipped in the shards;
  (c) the final checkpoint's predicted field, titled with this patch's MAE.

Conventions taken from the shard contract, not assumed: `dist` is int8, signed
POSITIVE INSIDE the glacier and negative outside, clipped to [-32, 31] px; the
training target is that value over 32, and eval.py multiplies the mean absolute
error back by 32 to report pixels. Both panels here are therefore drawn in
pixels on one symmetric scale, so target and prediction are directly comparable.

The colour ramp is deliberately blue-inside: blue already means glacier ice
everywhere else in the report. Debris orange is NOT used, because it would read
as the debris class rather than as "outside the glacier".

Output: report/latex/figures/fig_distance.png
Usage:  python fig_distance.py   (~1 min on CPU)
"""
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "code"))
from model import build_model  # noqa: E402

SHARDS = ROOT / "analysis" / "shards"
CKPT = ROOT / "analysis" / "runs_kaggle" / "final" / "runs" / "final" / "final" / "best.pt"
MANIFEST = ROOT / "analysis" / "subset_manifest.csv"
OUT = ROOT / "report" / "latex" / "figures" / "fig_distance.png"

NODATA = 65535
DIST_CLIP = 32.0                       # int8 truncation bound, in pixels
ZONE = "val"                           # a selection patch, as in Figure 7's first row

INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
CLEAN_RGB = np.array([42, 120, 214]) / 255.0
DEBRIS_RGB = np.array([235, 104, 52]) / 255.0
INVALID_RGB = np.array([225, 224, 217]) / 255.0

# outside -> boundary -> inside. Warm grey for rock, white at the zero contour,
# the report's dark glacier blue deep inside the ice.
DIST_CMAP = LinearSegmentedColormap.from_list(
    "signed_distance", ["#8f8a7c", "#c9c4b6", "#ffffff", "#6da7ec", "#184f95"])

plt.rcParams.update({"font.family": "Segoe UI", "figure.facecolor": "white"})


def zone_index(zone):
    idx = {}
    for p in sorted((SHARDS / zone).glob("shard_s*.npz")):
        with np.load(p) as z:
            for i, n in enumerate(z["names"]):
                idx[str(n)] = (p, i)
    return idx


def main():
    ckpt = torch.load(CKPT, map_location="cpu", weights_only=False)
    cfg = ckpt["cfg"]
    if not cfg.get("heads", {}).get("dist"):
        raise SystemExit("fig_distance: checkpoint has no distance head")
    channels = np.asarray(cfg["channels"])
    model = build_model(cfg)
    model.load_state_dict(ckpt["model"])
    model.eval()

    spec = json.loads((SHARDS / "quant_spec.json").read_text())
    qmin, qscale = np.asarray(spec["qmin"], np.float32), np.asarray(spec["scale"], np.float32)
    stats = json.loads((SHARDS / "train_stats.json").read_text())
    mean, std = np.asarray(stats["mean"], np.float32), np.asarray(stats["std"], np.float32)

    # Candidates: glacier-rich and debris-bearing, so the field has a real interior
    # rather than a thin sliver. Among those, the patch shown is the one whose MAE
    # is CLOSEST TO THE ZONE MEAN reported in Table 3, not the best or the biggest.
    # Ranking purely by glacier area picks a patch that is 69% ice and lands at
    # 7.2 px, well above the 4.3 px the report quotes; showing that as the
    # illustration of the task would misrepresent the head's typical error.
    zone_mae = json.loads((CKPT.parent / ("eval_%s.json" % ZONE)).read_text())["dist_mae_px"]
    rows = [r for r in csv.DictReader(open(MANIFEST, encoding="utf-8"))
            if r["zone"] == ZONE and int(r["debris_px"]) > 0]
    rows.sort(key=lambda r: -int(r["glacier_px"]))
    idx = zone_index(ZONE)

    def evaluate(name):
        path, i = idx[name]
        with np.load(path) as z:
            q, m, d = z["imgs"][i], z["masks"][i], z["dist"][i]
        nodata = q == NODATA
        valid = ~nodata.any(axis=-1)
        img = qmin + q.astype(np.float32) * qscale
        x = (img[..., channels] - mean[channels]) / std[channels]
        x[nodata[..., channels]] = 0.0
        target_px = d.astype(np.float32)                   # shards already store pixels
        with torch.no_grad():
            out = model(torch.from_numpy(np.ascontiguousarray(x.transpose(2, 0, 1)))[None])
        pred_px = out["dist"][0, 0].numpy() * DIST_CLIP    # undo the /32 target scaling
        return {"m": m, "valid": valid, "target": target_px, "pred": pred_px,
                "mae": float(np.abs(pred_px - target_px)[valid].mean())}

    cands = [r["img"] for r in rows if r["img"] in idx][:12]
    if not cands:
        raise SystemExit("fig_distance: no candidate patch found in the %s shards" % ZONE)
    scored = []
    for nm in cands:
        res = evaluate(nm)
        scored.append((abs(res["mae"] - zone_mae), nm, res))
        print("fig_distance:   candidate %-24s MAE %5.2f px" % (nm, res["mae"]))
    scored.sort(key=lambda t: t[0])
    _, name, res = scored[0]
    m, valid = res["m"], res["valid"]
    target_px, pred_px, mae = res["target"], res["pred"], res["mae"]

    print("fig_distance: zone MAE %.2f px, chose %s" % (zone_mae, name))
    glacier = (m.sum(-1) > 0)
    print("fig_distance: %s  valid %.0f%%  glacier %.1f%%  patch MAE %.2f px"
          % (name, 100 * valid.mean(), 100 * glacier[valid].mean(), mae))

    label = np.zeros(glacier.shape + (3,), np.float32)
    label[:] = 0.97
    label[glacier] = CLEAN_RGB
    label[(m[..., 1] > 0)] = DEBRIS_RGB
    label[~valid] = INVALID_RGB

    fig, axes = plt.subplots(1, 3, figsize=(12.6, 4.75))
    fig.subplots_adjust(left=0.005, right=0.885, top=0.86, bottom=0.02, wspace=0.045)

    axes[0].imshow(label)
    axes[0].set_title("(a) expert label", loc="left", fontsize=13,
                      fontweight="semibold", color=INK, pad=9)

    shown = []
    for ax, field, title in ((axes[1], target_px, "(b) target distance field"),
                             (axes[2], pred_px, "(c) predicted field")):
        f = np.ma.masked_where(~valid, field)
        im = ax.imshow(f, cmap=DIST_CMAP, vmin=-DIST_CLIP, vmax=DIST_CLIP)
        im.cmap.set_bad(INVALID_RGB)
        ax.set_title(title, loc="left", fontsize=13, fontweight="semibold",
                     color=INK, pad=9)
        shown.append(im)

    # The zero contour on both fields: it IS the glacier boundary, which is the
    # whole point of the task and is otherwise only implicit in the colour ramp.
    for ax, field in ((axes[1], target_px), (axes[2], pred_px)):
        ax.contour(np.where(valid, field, np.nan), levels=[0.0],
                   colors="#0b0b0b", linewidths=0.9, alpha=0.75)

    axes[2].annotate("MAE %.1f px on this patch" % mae, (1.0, 1.0),
                     xycoords="axes fraction", xytext=(0, 11),
                     textcoords="offset points", ha="right", fontsize=11.5, color=INK2)

    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)

    cax = fig.add_axes([0.898, 0.10, 0.014, 0.70])
    cb = fig.colorbar(shown[-1], cax=cax)
    # Kept short on purpose. A rotated label longer than the bar is taller than the
    # bar and runs off the canvas; the full sign convention lives in the caption.
    cb.set_label("distance to boundary (px), + inside", fontsize=11, color=INK2)
    cb.ax.tick_params(labelsize=10, colors=MUTED)
    cb.outline.set_visible(False)

    fig.savefig(OUT, dpi=200)
    plt.close(fig)
    print("fig_distance: wrote %s" % OUT)


if __name__ == "__main__":
    main()
