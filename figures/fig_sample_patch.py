#!/usr/bin/env python3
"""fig_sample_patch.py -- Figure 1: what one HKH patch actually contains.

Eight panels over a single 512x512 training-zone patch, in the report's house
style and using the report's class colours (clean = blue, debris = orange), so
Figure 1 and Figure 6 read as the same legend:

  row 1  true colour | false colour (SWIR1,NIR,Red) | thermal B6 | NDSI (sign-corrected)
  row 2  SRTM elevation | slope | per-pixel validity | expert labels on true colour

The validity panel is the point of the figure: it shows the Landsat-7 scan-line
corrector gaps as the diagonal bands that Section 3.2 quantifies, and it is the
mask that gates every loss term and every metric.

The patch is chosen (not hand-picked from the eval zones) to satisfy: training
zone only, source scene acquired AFTER the May 2003 SLC failure so the striping
really is SLC-off, debris present, and validity near the corpus mean.

Output: report/latex/figures/fig_sample_patch.png
Usage:  python fig_sample_patch.py [patch_name.npy]
"""
import csv
import json
import sys
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

ROOT = Path(__file__).resolve().parent.parent.parent
SHARDS = ROOT / "analysis" / "shards"
MANIFEST = ROOT / "analysis" / "subset_manifest.csv"
OUT = ROOT / "report" / "latex" / "figures" / "fig_sample_patch.png"

# Training zone, scene LE07_149035_20070915 (post May-2003 SLC failure), 45% valid,
# 30,019 glacier px of which 7,436 debris. See the module docstring for the criteria.
DEFAULT_PATCH = "slice_16_img_148.npy"

NODATA = 65535
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
CLEAN, DEBRIS = "#2a78d6", "#eb6834"          # identical to Figures 3, 4, 6
CLEAN_RGB = np.array([42, 120, 214]) / 255.0
DEBRIS_RGB = np.array([235, 104, 52]) / 255.0
INVALID_RGB = np.array([225, 224, 217]) / 255.0

plt.rcParams.update({
    "font.family": "Segoe UI", "font.size": 10.5,
    "figure.facecolor": "white", "axes.facecolor": "white",
    "text.color": INK2,
})


def find_patch(name):
    for zone in ("train", "val", "test", "east"):
        for p in sorted((SHARDS / zone).glob("shard_s*.npz")):
            with np.load(p) as z:
                names = [str(n) for n in z["names"]]
                if name in names:
                    i = names.index(name)
                    return zone, z["imgs"][i], z["masks"][i]
    raise SystemExit("fig_sample_patch: %s not found in %s" % (name, SHARDS))


def dequantise(q):
    spec = json.loads((SHARDS / "quant_spec.json").read_text())
    qmin = np.asarray(spec["qmin"], np.float32)
    scale = np.asarray(spec["scale"], np.float32)
    nodata = q == NODATA
    img = qmin + q.astype(np.float32) * scale
    img[nodata] = np.nan
    return img, nodata


def stretch(bands, valid, pct=(2, 98)):
    """Percentile stretch to [0,1] over valid pixels; invalid pixels go pale grey."""
    out = np.zeros(bands.shape, np.float32)
    for c in range(bands.shape[-1]):
        v = bands[..., c][valid]
        lo, hi = np.percentile(v, pct)
        out[..., c] = np.clip((bands[..., c] - lo) / max(hi - lo, 1e-6), 0, 1)
    out[~valid] = INVALID_RGB
    return out


def frame(ax, title):
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_color(GRID)
        s.set_linewidth(0.9)
    ax.set_title(title, fontsize=11, color=INK, pad=7, fontweight="semibold")


def bar(fig, ax, im, label):
    """Slim horizontal colour bar tucked under a panel."""
    box = ax.get_position()
    cax = fig.add_axes([box.x0 + box.width * 0.10, box.y0 - 0.021,
                        box.width * 0.80, 0.011])
    cb = fig.colorbar(im, cax=cax, orientation="horizontal")
    cb.outline.set_visible(False)
    cb.ax.tick_params(length=0, labelsize=8.5, colors=MUTED, pad=2)
    cb.set_label(label, fontsize=8.5, color=MUTED, labelpad=3)


def main():
    name = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_PATCH
    zone, q, mask = find_patch(name)
    img, nodata = dequantise(q)
    valid = ~nodata.any(axis=-1)

    meta = next((r for r in csv.DictReader(open(MANIFEST, encoding="utf-8"))
                 if r["img"] == name), {})

    # Shard masks are canonical 2-channel: ch0 = clean ice, ch1 = debris-covered.
    # 15 corpus masks double-label a few pixels as both; debris wins, as in training.
    clean = mask[..., 0] > 0
    debris = mask[..., 1] > 0
    labels = np.zeros(clean.shape, np.uint8)
    labels[clean & ~debris] = 1
    labels[debris] = 2

    truecol = stretch(img[..., [2, 1, 0]], valid)          # B3, B2, B1
    falsecol = stretch(img[..., [4, 3, 2]], valid)         # SWIR1, NIR, Red

    fig, axes = plt.subplots(2, 4, figsize=(13.0, 7.7))
    fig.subplots_adjust(left=0.012, right=0.988, top=0.855, bottom=0.135,
                        wspace=0.045, hspace=0.30)

    fig.text(0.012, 0.958, "What one HKH patch contains, and how much of it is missing",
             fontsize=15, color=INK, fontweight="semibold")
    fig.text(0.012, 0.918,
             "%s  ·  training zone  ·  scene %s  ·  %.2f°E %.2f°N  ·  "
             "%s glacier px of which %s debris  ·  %.0f%% of pixels valid in all 15 channels"
             % (name.replace(".npy", ""), meta.get("scene", "n/a"),
                float(meta.get("lon", 0)), float(meta.get("lat", 0)),
                format(int(meta.get("glacier_px", 0)), ","),
                format(int(meta.get("debris_px", 0)), ","), 100 * valid.mean()),
             fontsize=10.5, color=MUTED)

    axes[0, 0].imshow(truecol)
    frame(axes[0, 0], "True colour  (B3, B2, B1)")

    axes[0, 1].imshow(falsecol)
    frame(axes[0, 1], "False colour  (SWIR1, NIR, Red)")

    thermal = np.where(valid, img[..., 5], np.nan)
    im = axes[0, 2].imshow(thermal, cmap="inferno")
    frame(axes[0, 2], "Thermal  (B6 VCID 1)")

    ndsi = np.where(valid, -img[..., 11], np.nan)          # ch 11 ships sign-flipped
    im2 = axes[0, 3].imshow(ndsi, cmap="RdBu", vmin=-1, vmax=1)
    frame(axes[0, 3], "NDSI  (sign-corrected)")

    # cividis rather than a hypsometric ramp: perceptually uniform, colourblind-safe,
    # and it does not put a blue band on the valley floors that reads as water.
    elev = np.where(valid, img[..., 13], np.nan)
    im3 = axes[1, 0].imshow(elev, cmap="cividis")
    frame(axes[1, 0], "SRTM elevation")

    slope = np.where(valid, img[..., 14], np.nan)
    im4 = axes[1, 1].imshow(slope, cmap="magma")
    frame(axes[1, 1], "Slope")

    axes[1, 2].imshow(valid, cmap=ListedColormap(["#d8d6cd", "#3d4b5c"]),
                      vmin=0, vmax=1, interpolation="nearest")
    frame(axes[1, 2], "Per-pixel validity mask")

    base = truecol * 0.52 + 0.30
    base[~valid] = INVALID_RGB
    over = base.copy()
    over[(labels == 1) & valid] = base[(labels == 1) & valid] * 0.25 + CLEAN_RGB * 0.75
    over[(labels == 2) & valid] = base[(labels == 2) & valid] * 0.25 + DEBRIS_RGB * 0.75
    axes[1, 3].imshow(over)
    frame(axes[1, 3], "Expert labels  (ICIMOD)")

    fig.canvas.draw()                                       # positions before colourbars
    bar(fig, axes[0, 2], im, "DN")
    bar(fig, axes[0, 3], im2, "snow/ice positive")
    bar(fig, axes[1, 0], im3, "m")
    bar(fig, axes[1, 1], im4, "degrees")

    handles = [Patch(facecolor=CLEAN, label="clean ice"),
               Patch(facecolor=DEBRIS, label="debris-covered ice"),
               Patch(facecolor="#3d4b5c", label="valid pixel"),
               Patch(facecolor="#d8d6cd", label="invalid: SLC-off gap or scene edge")]
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False,
               fontsize=10.5, handlelength=1.1, handleheight=1.1,
               handletextpad=0.5, columnspacing=2.4,
               bbox_to_anchor=(0.5, 0.012))

    fig.savefig(OUT, dpi=200)
    print("wrote %s  (zone=%s, valid=%.3f)" % (OUT, zone, valid.mean()))


if __name__ == "__main__":
    main()
