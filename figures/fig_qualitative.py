#!/usr/bin/env python3
"""fig_qualitative.py -- qualitative predictions montage from the final checkpoint.

Runs the final model (CPU) on debris-rich candidate patches from val/test/east
shards, scores each patch, and composes a 4-row montage: a validation success,
a test success, a test failure and an east OOD example. Classes are rendered as
colour overlays on the true-colour composite instead of black label masks.

Output: report/latex/figures/fig_qualitative.png
Usage:  python fig_qualitative.py   (~2 min on CPU)
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

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "code"))
from model import build_model  # noqa: E402

SHARDS = ROOT / "analysis" / "shards"
CKPT = ROOT / "analysis" / "runs_kaggle" / "final" / "runs" / "final" / "final" / "best.pt"
MANIFEST = ROOT / "analysis" / "subset_manifest.csv"
OUT = ROOT / "report" / "latex" / "figures" / "fig_qualitative.png"

NODATA = 65535
BLUE, ORANGE = "#2a78d6", "#eb6834"
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
CLEAN_RGB = np.array([42, 120, 214]) / 255.0
DEBRIS_RGB = np.array([235, 104, 52]) / 255.0
INVALID_RGB = np.array([225, 224, 217]) / 255.0
# error-map palette: agreement quiet, misses loud
HIT, FP, FN = "#9aa79b", "#7b52c9", "#d92b4b"
HIT_RGB = np.array([154, 167, 155]) / 255.0
FP_RGB = np.array([123, 82, 201]) / 255.0
FN_RGB = np.array([217, 43, 75]) / 255.0


def zone_index(zone):
    """name -> (npz path, item index) for one shard zone."""
    idx = {}
    for p in sorted((SHARDS / zone).glob("shard_s*.npz")):
        with np.load(p) as z:
            for i, n in enumerate(z["names"]):
                idx[str(n)] = (p, i)
    return idx


def load_patch(path, i):
    with np.load(path) as z:
        return z["imgs"][i], z["masks"][i]


def prep(q, spec, stats, channels):
    nodata = q == NODATA
    valid = ~nodata.any(axis=-1)
    img = spec["qmin"] + q.astype(np.float32) * spec["scale"]
    x = (img[..., channels] - stats["mean"][channels]) / stats["std"][channels]
    x[nodata[..., channels]] = 0.0
    rgb_dn = img[..., [2, 1, 0]]                     # B3,B2,B1 true colour
    return x, valid, rgb_dn


@torch.no_grad()
def predict(model, x):
    t = torch.from_numpy(np.ascontiguousarray(x.transpose(2, 0, 1)))[None]
    out = model(t)
    pg = torch.sigmoid(out["glacier"][0, 0]).numpy() > 0.5
    pt = torch.sigmoid(out["type"][0, 0]).numpy() > 0.5
    return pg.astype(np.int64) + (pg & pt).astype(np.int64)  # 0 bg / 1 clean / 2 debris


def iou(a, b, valid):
    a, b = a & valid, b & valid
    inter, union = (a & b).sum(), (a | b).sum()
    return inter / union if union else float("nan")


def stretch(rgb_dn, valid):
    v = rgb_dn[valid]
    lo, hi = np.percentile(v, [2, 98], axis=0)
    out = np.clip((rgb_dn - lo) / np.maximum(hi - lo, 1e-6), 0, 1)
    out[~valid] = INVALID_RGB
    return out


def overlay(rgb, labels, valid):
    base = rgb * 0.52 + 0.30                        # washed base so classes pop
    base[~valid] = INVALID_RGB
    out = base.copy()
    for cls, col in ((1, CLEAN_RGB), (2, DEBRIS_RGB)):
        m = (labels == cls) & valid
        out[m] = base[m] * 0.25 + col * 0.75
    return out


def errormap(rgb, truth, pred, valid):
    """Where the glacier head agrees and where it fails. Agreement is kept quiet so
    the two error classes carry the panel; a false negative is the costly one for
    hazard screening, so it gets the strong colour."""
    base = rgb * 0.35 + 0.52
    base[~valid] = INVALID_RGB
    out = base.copy()
    t, p = truth > 0, pred > 0
    for m, col in (((t & p) & valid, HIT_RGB),
                   ((~t & p) & valid, FP_RGB),
                   ((t & ~p) & valid, FN_RGB)):
        out[m] = base[m] * 0.18 + col * 0.82
    return out


def main():
    ckpt = torch.load(CKPT, map_location="cpu", weights_only=False)
    cfg = ckpt["cfg"]
    channels = np.asarray(cfg["channels"])
    model = build_model(cfg)
    model.load_state_dict(ckpt["model"])
    model.eval()

    spec = json.loads((SHARDS / "quant_spec.json").read_text())
    spec = {"qmin": np.asarray(spec["qmin"], np.float32),
            "scale": np.asarray(spec["scale"], np.float32)}
    stats = json.loads((SHARDS / "train_stats.json").read_text())
    stats = {"mean": np.asarray(stats["mean"], np.float32),
             "std": np.asarray(stats["std"], np.float32)}

    rows = list(csv.DictReader(open(MANIFEST, encoding="utf-8")))
    pools = {}
    for zone, k in (("val", 8), ("test", 10), ("east", 8)):
        zz = [r for r in rows if r["zone"] == zone and int(r["debris_px"]) > 0]
        zz.sort(key=lambda r: -int(r["debris_px"]))
        pools[zone] = [r["img"] for r in zz[:k]]

    results = {}
    for zone, names in pools.items():
        zidx = zone_index(zone)
        for name in names:
            q, m = load_patch(*zidx[name])
            x, valid, rgb_dn = prep(q, spec, stats, channels)
            pred = predict(model, x)
            truth = (m.sum(-1) > 0).astype(np.int64) + (m[..., 1] > 0).astype(np.int64)
            results[(zone, name)] = {
                "rgb": stretch(rgb_dn, valid), "valid": valid,
                "truth": truth, "pred": pred,
                "g_iou": iou(pred > 0, truth > 0, valid),
                "d_iou": iou(pred == 2, truth == 2, valid),
            }
            print(f"{zone:5s} {name:24s} gIoU {results[(zone, name)]['g_iou']:.3f} "
                  f"dIoU {results[(zone, name)]['d_iou']:.3f}")

    def pick(zone, key, worst=False):
        cands = [(k, v) for k, v in results.items() if k[0] == zone]
        cands.sort(key=lambda kv: kv[1][key], reverse=not worst)
        return cands[0]

    sel = [("Validation", "success", *pick("val", "d_iou")),
           ("Test (Karakoram)", "success", *pick("test", "d_iou")),
           ("Test (Karakoram)", "failure", *pick("test", "g_iou", worst=True)),
           ("East OOD", "transfer", *pick("east", "d_iou"))]

    plt.rcParams.update({"font.family": "Segoe UI", "figure.facecolor": "white"})
    n = len(sel)
    fig, axes = plt.subplots(n, 4, figsize=(13.4, 3.55 * n))
    fig.subplots_adjust(left=0.115, right=0.996, top=0.958, bottom=0.042,
                        wspace=0.028, hspace=0.075)

    for c, t in enumerate(("Landsat-7 true colour", "Expert label", "Model prediction",
                           "Glacier-extent error")):
        axes[0, c].set_title(t, fontsize=13, color=INK, pad=10, fontweight="semibold")

    for r, (zone_lab, kind, (zone, name), d) in enumerate(sel):
        axes[r, 0].imshow(d["rgb"])
        axes[r, 1].imshow(overlay(d["rgb"], d["truth"], d["valid"]))
        axes[r, 2].imshow(overlay(d["rgb"], d["pred"], d["valid"]))
        axes[r, 3].imshow(errormap(d["rgb"], d["truth"], d["pred"], d["valid"]))
        for ax in axes[r]:
            ax.axis("off")
        di = f"{d['d_iou']:.2f}" if np.isfinite(d["d_iou"]) else "n/a"
        axes[r, 0].annotate(
            f"{zone_lab}\n{name.replace('.npy', '')}\n"
            f"glacier IoU {d['g_iou']:.2f}\ndebris IoU {di}",
            xy=(0, 0.5), xycoords="axes fraction", xytext=(-12, 0),
            textcoords="offset points", ha="right", va="center",
            fontsize=11, color=INK2, linespacing=1.7)

    swatch = lambda c, l: plt.Line2D([], [], marker="s", linestyle="", markersize=11,
                                     markerfacecolor=c, markeredgewidth=0, label=l)
    labels = [(tuple(CLEAN_RGB), "clean ice"),
              (tuple(DEBRIS_RGB), "debris-covered ice"),
              (tuple(INVALID_RGB), "invalid (SLC-off / nodata)"),
              (tuple(HIT_RGB), "glacier hit"),
              (tuple(FN_RGB), "missed glacier"),
              (tuple(FP_RGB), "false glacier")]
    fig.legend(handles=[swatch(c, l) for c, l in labels], loc="lower center", ncol=6,
               frameon=False, fontsize=11, handletextpad=0.4, columnspacing=1.6)

    fig.savefig(OUT, dpi=190)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
