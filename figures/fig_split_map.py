#!/usr/bin/env python3
"""fig_split_map.py -- geography-style study-area map of the geographic split.

Shaded-relief basemap (Esri World Shaded Relief tiles, cached locally), patch
footprint centroids coloured by zone, direct zone callouts, country borders
(Natural Earth 50m), north arrow and scale bar. No graticule/axes by design.

Output: report/latex/figures/fig_split_map.png
Usage:  python fig_split_map.py
"""
import csv
import json
import math
import tempfile
import time
import urllib.request
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent.parent
OUT = ROOT / "report" / "latex" / "figures" / "fig_split_map.png"
MANIFEST = ROOT / "analysis" / "subset_manifest.csv"
CACHE = Path(tempfile.gettempdir()) / "hkh_map_tiles"

TILE_URL = ("https://server.arcgisonline.com/ArcGIS/rest/services/"
            "World_Shaded_Relief/MapServer/tile/{z}/{y}/{x}")
BORDERS_URL = ("https://raw.githubusercontent.com/nvkelso/natural-earth-vector/"
               "master/geojson/ne_50m_admin_0_boundary_lines_land.geojson")

LON0, LON1, LAT0, LAT1 = 66.5, 99.5, 25.8, 39.8
Z = 7
R = 6378137.0
WORLD = 2 * math.pi * R

ZCOL = {"train": "#2a78d6", "val": "#1baf7a", "test": "#eb6834", "east": "#4a3aa7"}
EXCL = "#9a988f"
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"


def merc(lon, lat):
    return (R * math.radians(lon),
            R * math.asinh(math.tan(math.radians(lat))))


def fetch(url, dest):
    if dest.exists():
        return dest
    req = urllib.request.Request(url, headers={"User-Agent": "report-figure/1.0"})
    for attempt in range(3):
        try:
            dest.write_bytes(urllib.request.urlopen(req, timeout=30).read())
            return dest
        except Exception:
            if attempt == 2:
                raise
            time.sleep(1.5)


def basemap():
    n = 2 ** Z
    x0 = int((LON0 + 180) / 360 * n)
    x1 = int((LON1 + 180) / 360 * n)
    y0 = int((1 - math.asinh(math.tan(math.radians(LAT1))) / math.pi) / 2 * n)
    y1 = int((1 - math.asinh(math.tan(math.radians(LAT0))) / math.pi) / 2 * n)
    CACHE.mkdir(parents=True, exist_ok=True)
    cols = []
    for ty in range(y0, y1 + 1):
        row = []
        for tx in range(x0, x1 + 1):
            p = fetch(TILE_URL.format(z=Z, y=ty, x=tx), CACHE / f"sr_{Z}_{ty}_{tx}.png")
            row.append(np.asarray(Image.open(p).convert("RGB")))
        cols.append(np.hstack(row))
    img = np.vstack(cols)
    # mercator extent of the stitched tile block
    mx0 = x0 / n * WORLD - WORLD / 2
    mx1 = (x1 + 1) / n * WORLD - WORLD / 2
    my1 = WORLD / 2 - y0 / n * WORLD
    my0 = WORLD / 2 - (y1 + 1) / n * WORLD
    return img, (mx0, mx1, my0, my1)


def borders():
    p = CACHE / "ne_borders.geojson"
    fetch(BORDERS_URL, p)
    gj = json.loads(p.read_text(encoding="utf-8"))
    segs = []
    for f in gj["features"]:
        g = f["geometry"]
        lines = g["coordinates"] if g["type"] == "MultiLineString" else [g["coordinates"]]
        for line in lines:
            pts = [merc(lo, la) for lo, la in line
                   if LON0 - 2 <= lo <= LON1 + 2 and LAT0 - 2 <= la <= LAT1 + 2]
            if len(pts) > 1:
                segs.append(pts)
    return segs


def main():
    rows = list(csv.DictReader(open(MANIFEST, encoding="utf-8")))
    img, ext = basemap()

    fig, ax = plt.subplots(figsize=(13.6, 6.9))
    fig.subplots_adjust(0, 0, 1, 1)
    ax.imshow(img, extent=ext, origin="upper", interpolation="bilinear", alpha=0.92)

    for seg in borders():
        xs, ys = zip(*seg)
        ax.plot(xs, ys, color="#7d7b74", linewidth=0.7, alpha=0.75, zorder=2)

    # patches: excluded first (underneath), then zones
    def sc(zones, color, size, alpha, z):
        xy = [merc(float(r["lon"]), float(r["lat"])) for r in rows if r["zone"] in zones]
        if xy:
            xs, ys = zip(*xy)
            ax.scatter(xs, ys, s=size, c=color, alpha=alpha, linewidths=0, zorder=z)

    sc({"buffer_removed", "dropped_straddler"}, EXCL, 7, 0.55, 3)
    for zone, col in ZCOL.items():
        sc({zone}, col, 11, 0.85, 4)

    halo = [pe.withStroke(linewidth=3.5, foreground="white")]

    def callout(lon, lat, s, col, fs=17):
        ax.annotate(s, merc(lon, lat), ha="center", fontsize=fs, color=col,
                    fontweight="bold", path_effects=halo, zorder=6,
                    family="Segoe UI")

    callout(70.1, 31.9, "TRAIN", ZCOL["train"])
    callout(80.9, 31.6, "TRAIN", ZCOL["train"])
    callout(76.1, 29.3, "VALIDATION", ZCOL["val"])
    callout(73.7, 38.9, "TEST", ZCOL["test"])
    callout(90.6, 26.45, "EAST (OOD)", ZCOL["east"])

    # country labels, small-caps style
    for lon, lat, name in [(69.3, 33.9, "PAKISTAN"), (78.9, 27.2, "INDIA"),
                           (84.2, 28.5, "NEPAL"), (88.5, 36.6, "CHINA (TIBET)")]:
        ax.annotate(name, merc(lon, lat), ha="center", fontsize=10.5, color="#6b6963",
                    path_effects=[pe.withStroke(linewidth=2.5, foreground="white")],
                    zorder=5, family="Segoe UI")

    # legend (counts from the frozen split)
    counts = {"train": 1056, "val": 283, "test": 315, "east": 293}
    names = {"train": "Train", "val": "Validation", "test": "Test (one-shot)",
             "east": "East OOD (one-shot)"}
    lx, ly0, dy = 0.822, 0.955, 0.052
    ax.annotate("Geographic split (2,269 patches)", (lx - 0.012, ly0),
                xycoords="axes fraction", fontsize=12.5, color=INK,
                fontweight="bold", family="Segoe UI", path_effects=halo, va="top")
    items = [(names[z], ZCOL[z], counts[z]) for z in ("train", "val", "test", "east")]
    items.append(("Excluded (buffer / straddler)", EXCL, 322))
    for i, (name, col, cnt) in enumerate(items):
        y = ly0 - 0.055 - i * dy
        ax.scatter([lx - 0.012], [y - 0.012], s=52, c=col, transform=ax.transAxes,
                   zorder=6, linewidths=0)
        ax.annotate(f"{name}  ({cnt:,})", (lx + 0.002, y), xycoords="axes fraction",
                    fontsize=11, color=INK2, family="Segoe UI", va="top",
                    path_effects=halo)

    # north arrow (top-left)
    ax.annotate("N", (0.028, 0.90), xycoords="axes fraction", fontsize=15,
                color=INK2, fontweight="bold", ha="center", family="Segoe UI",
                path_effects=halo)
    ax.annotate("", xy=(0.028, 0.965), xytext=(0.028, 0.885),
                xycoords="axes fraction",
                arrowprops=dict(arrowstyle="-|>", color=INK2, linewidth=1.8))

    # scale bar: 250 km of ground distance at the map's mid-latitude
    midlat = (LAT0 + LAT1) / 2
    bar_m = 250_000 / math.cos(math.radians(midlat))
    x0, _ = merc(67.3, 27.2)
    _, ybar = merc(67.3, 27.2)
    ax.plot([x0, x0 + bar_m], [ybar, ybar], color=INK2, linewidth=2.6,
            solid_capstyle="butt", zorder=6,
            path_effects=[pe.withStroke(linewidth=5, foreground="white")])
    for xx, lab in [(x0, "0"), (x0 + bar_m, "250 km")]:
        ax.annotate(lab, (xx, ybar), xytext=(0, -14), textcoords="offset points",
                    ha="center", fontsize=10, color=INK2, family="Segoe UI",
                    path_effects=halo)

    ax.annotate("Basemap: Esri World Shaded Relief \u00b7 borders: Natural Earth",
                (0.998, 0.006), xycoords="axes fraction", ha="right", fontsize=8.5,
                color=MUTED, family="Segoe UI", path_effects=halo)

    e0 = merc(LON0, LAT0)
    e1 = merc(LON1, LAT1)
    ax.set_xlim(e0[0], e1[0])
    ax.set_ylim(e0[1], e1[1])
    ax.set_aspect("equal")
    ax.axis("off")

    fig.savefig(OUT, dpi=210, bbox_inches="tight", pad_inches=0.02)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
