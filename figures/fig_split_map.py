#!/usr/bin/env python3
"""fig_split_map.py -- geography-style study-area map of the geographic split.

Shaded-relief basemap (Esri World Shaded Relief tiles, cached locally), patch
footprint centroids coloured by zone, direct zone callouts, country borders
(Natural Earth 50m), north arrow and scale bar.

Drawn as a two-panel plate, each panel at its own scale and carrying its own
scale bar: the western sector holds the train, validation and test zones, the
eastern one the out-of-distribution zone. The two clusters sit 30 degrees of
longitude apart, so a single frame has to be a letterbox in which the terrain
is tiny; splitting them lets both sectors be read at roughly twice the size on
the same page width.

No graticule and no coordinate ruler: this is a map, not a scatter plot, so
zones are named on the terrain and the scale is given by the bars. The frozen
band limits are stated numerically in Section 3.5 instead.

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
NE_BASE = ("https://raw.githubusercontent.com/nvkelso/natural-earth-vector/"
           "master/geojson/")
BORDERS_FILE = "ne_50m_admin_0_boundary_lines_land.geojson"
DISPUTED_FILE = "ne_50m_admin_0_boundary_lines_disputed_areas.geojson"

# The two panels, cropped to their own data with a small margin. Panel A holds
# every training, validation and test patch; panel B the east OOD zone.
PANEL_A = (66.95, 84.25, 29.05, 40.05)      # lon0, lon1, lat0, lat1
PANEL_B = (85.55, 97.85, 26.40, 30.00)
LON0, LON1, LAT0, LAT1 = 66.5, 99.5, 25.8, 40.4   # border-clipping bounds only
Z = 7
R = 6378137.0
WORLD = 2 * math.pi * R

ZCOL = {"train": "#2a78d6", "val": "#1baf7a", "test": "#eb6834", "east": "#4a3aa7"}
EXCL = "#9a988f"
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
BORDER = "#3f3d38"          # dark enough to read over the sepia relief at print size

# the frozen assignment rule, from analysis/split.json -> bands
BANDS = [("test", 72.95, 74.65), ("val", 76.85, 78.15), ("east", 86.0, LON1)]


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


def basemap(lon0, lon1, lat0, lat1):
    n = 2 ** Z
    x0 = int((lon0 + 180) / 360 * n)
    x1 = int((lon1 + 180) / 360 * n)
    y0 = int((1 - math.asinh(math.tan(math.radians(lat1))) / math.pi) / 2 * n)
    y1 = int((1 - math.asinh(math.tan(math.radians(lat0))) / math.pi) / 2 * n)
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


def _segments(name):
    """Natural Earth boundary lines, clipped to the study region, in mercator.

    Clipping keeps CONTIGUOUS runs rather than filtering point by point: a line
    that leaves the box and re-enters it would otherwise be rejoined by a
    straight chord across the gap, drawing a border where there is none."""
    p = CACHE / name
    fetch(NE_BASE + name, p)
    gj = json.loads(p.read_text(encoding="utf-8"))
    segs, run = [], []
    for f in gj["features"]:
        g = f.get("geometry")
        if not g:
            continue
        lines = g["coordinates"] if g["type"] == "MultiLineString" else [g["coordinates"]]
        for line in lines:
            for c in line:
                lo, la = c[0], c[1]
                if LON0 - 2 <= lo <= LON1 + 2 and LAT0 - 2 <= la <= LAT1 + 2:
                    run.append(merc(lo, la))
                else:
                    if len(run) > 1:
                        segs.append(run)
                    run = []
            if len(run) > 1:
                segs.append(run)
            run = []
    return segs


def borders():
    """Settled land boundaries, and separately the disputed ones.

    The land file stops dead at every disputed boundary, and in this region that
    is most of the northern frontier: the Line of Control, Aksai Chin and the
    Siachen lines all fall inside the training and test bands. Drawing the land
    file alone left the map's most relevant borders blank, which is what made it
    read as terrain with no countries on it."""
    return _segments(BORDERS_FILE), _segments(DISPUTED_FILE)


HALO = [pe.withStroke(linewidth=3.5, foreground="white")]


def draw_panel(ax, rows, segs, disputed, extent, title, callouts, places, bar_lonlat):
    """Render one sector: relief, borders, evaluation bands, patches, labels."""
    lon0, lon1, lat0, lat1 = extent
    img, ext = basemap(*extent)
    ax.imshow(img, extent=ext, origin="upper", interpolation="bilinear", alpha=0.92)

    # Country borders, cased. A grey hairline disappeared into the shaded relief
    # once the plate was scaled to the measure, so the map read as elevation with
    # no countries on it. A white casing under a dark line is the standard
    # cartographic answer: it holds the border clear of the terrain at any size.
    for seg in segs:
        xs, ys = zip(*seg)
        ax.plot(xs, ys, color="white", linewidth=3.2, alpha=0.85,
                solid_capstyle="round", zorder=2)
        ax.plot(xs, ys, color=BORDER, linewidth=1.35, alpha=0.95,
                solid_capstyle="round", zorder=2.2)
    # Disputed boundaries dashed, the usual convention: several of them run
    # through the test and validation bands, and the report takes no position
    # on any of them.
    for seg in disputed:
        xs, ys = zip(*seg)
        ax.plot(xs, ys, color="white", linewidth=3.2, alpha=0.85,
                solid_capstyle="round", zorder=2)
        ax.plot(xs, ys, color=BORDER, linewidth=1.2, alpha=0.9,
                linestyle=(0, (4.5, 3)), zorder=2.2)

    # evaluation longitude bands: the split rule itself, drawn on the terrain
    ylo, yhi = merc(lon0, lat0)[1], merc(lon0, lat1)[1]
    for zone, a, b in BANDS:
        if b < lon0 or a > lon1:
            continue
        xa, xb = merc(a, 0)[0], merc(b, 0)[0]
        if b < LON1:                       # east is open-ended; tinting it would
            ax.fill_betweenx([ylo, yhi], xa, xb, color=ZCOL[zone], alpha=0.13,
                             linewidth=0, zorder=1)        # swamp the whole panel
        for xe, lon in ((xa, a), (xb, b)):
            if lon0 <= lon <= lon1:
                ax.plot([xe, xe], [ylo, yhi], color=ZCOL[zone], linewidth=1.0,
                        alpha=0.6, linestyle=(0, (5, 4)), zorder=2)

    def sc(zones, color, size, alpha, z):
        xy = [merc(float(r["lon"]), float(r["lat"])) for r in rows if r["zone"] in zones]
        if xy:
            xs, ys = zip(*xy)
            ax.scatter(xs, ys, s=size, c=color, alpha=alpha, linewidths=0, zorder=z)

    sc({"buffer_removed", "dropped_straddler"}, EXCL, 13, 0.75, 3)
    for zone, col in ZCOL.items():
        sc({zone}, col, 16, 0.85, 4)

    for lon, lat, s, col in callouts:
        ax.annotate(s, merc(lon, lat), ha="center", fontsize=18, color=col,
                    fontweight="bold", path_effects=HALO, zorder=6, family="Segoe UI")
    # Country names carry the same weight as the borders they sit inside: at the
    # 167.6 mm measure the plate is scaled to about half size, so 11 pt here
    # printed at roughly 5 pt and the names were legible only under a lens.
    for lon, lat, name in places:
        ax.annotate(name, merc(lon, lat), ha="center", fontsize=12.5, color="#4a4844",
                    path_effects=[pe.withStroke(linewidth=3, foreground="white")],
                    zorder=5, family="Segoe UI")

    # scale bar: 250 km of ground distance at this panel's mid-latitude
    bar_m = 250_000 / math.cos(math.radians((lat0 + lat1) / 2))
    bx, by = merc(*bar_lonlat)
    ax.plot([bx, bx + bar_m], [by, by], color=INK2, linewidth=2.8,
            solid_capstyle="butt", zorder=6,
            path_effects=[pe.withStroke(linewidth=5, foreground="white")])
    for xx, lab in [(bx, "0"), (bx + bar_m, "250 km")]:
        ax.annotate(lab, (xx, by), xytext=(0, -15), textcoords="offset points",
                    ha="center", fontsize=10.5, color=INK2, family="Segoe UI",
                    path_effects=HALO)

    ax.annotate(title, (0.011, 0.982), xycoords="axes fraction", va="top",
                ha="left", fontsize=13, color=INK, fontweight="bold",
                family="Segoe UI", path_effects=HALO, zorder=7)

    p0, p1 = merc(lon0, lat0), merc(lon1, lat1)
    ax.set_xlim(p0[0], p1[0])
    ax.set_ylim(p0[1], p1[1])
    ax.set_aspect("equal")
    ax.axis("off")
    for sp in ax.spines.values():
        sp.set_visible(False)


def main():
    rows = list(csv.DictReader(open(MANIFEST, encoding="utf-8")))
    segs, disputed = borders()

    # one figure width; each panel keeps its own true aspect, so the plate is as
    # tall as the geography demands rather than squeezed into a letterbox
    fw = 13.6
    def span(p):
        a, b = merc(p[0], p[2]), merc(p[1], p[3])
        return b[0] - a[0], b[1] - a[1]
    ax_w, ay_h = span(PANEL_A)
    bx_w, by_h = span(PANEL_B)
    ha = fw * ay_h / ax_w
    hb = fw * by_h / bx_w
    gap = 0.30
    fh = ha + gap + hb

    fig = plt.figure(figsize=(fw, fh))
    axA = fig.add_axes([0, (hb + gap) / fh, 1.0, ha / fh])
    axB = fig.add_axes([0, 0, 1.0, hb / fh])

    draw_panel(
        axA, rows, segs, disputed, PANEL_A,
        "(a)  Western sector: training zones with the test and validation bands",
        [(70.1, 31.9, "TRAIN", ZCOL["train"]), (81.2, 32.6, "TRAIN", ZCOL["train"]),
         (76.0, 29.7, "VALIDATION", ZCOL["val"]), (73.75, 39.2, "TEST", ZCOL["test"])],
        # CHINA sits out in undisputed Tibet, east of the dashed Aksai Chin lines
        # and below the legend column, which is the only band of the panel that is
        # clear of both the patch dots and the legend rows.
        [(69.3, 33.9, "PAKISTAN"), (75.6, 29.5, "INDIA"), (68.7, 35.4, "AFGHANISTAN"),
         (82.5, 35.5, "CHINA (TIBET)")],
        (67.45, 29.75))
    draw_panel(
        axB, rows, segs, disputed, PANEL_B,
        "(b)  Eastern sector: the out-of-distribution zone, held out entirely",
        [(90.9, 29.35, "EAST (OOD)", ZCOL["east"])],
        [(86.6, 28.35, "NEPAL"), (90.4, 27.45, "BHUTAN"), (94.2, 26.75, "INDIA"),
         (92.6, 29.6, "CHINA (TIBET)")],
        (85.95, 26.75))

    # north arrow and legend live on panel (a); the plate shares one orientation
    axA.annotate("N", (0.026, 0.885), xycoords="axes fraction", fontsize=15,
                 color=INK2, fontweight="bold", ha="center", family="Segoe UI",
                 path_effects=HALO)
    axA.annotate("", xy=(0.026, 0.955), xytext=(0.026, 0.868),
                 xycoords="axes fraction",
                 arrowprops=dict(arrowstyle="-|>", color=INK2, linewidth=1.8))

    counts = {"train": 1056, "val": 283, "test": 315, "east": 293}
    names = {"train": "Train", "val": "Validation", "test": "Test (one-shot)",
             "east": "East OOD (one-shot)"}
    lx, ly0, dy = 0.775, 0.962, 0.038
    axA.annotate("Geographic split (2,269 patches)", (lx - 0.012, ly0),
                 xycoords="axes fraction", fontsize=12.5, color=INK,
                 fontweight="bold", family="Segoe UI", path_effects=HALO, va="top")
    items = [(names[z], ZCOL[z], counts[z]) for z in ("train", "val", "test", "east")]
    items.append(("Excluded (buffer / straddler)", EXCL, 322))
    for i, (name, col, cnt) in enumerate(items):
        y = ly0 - 0.042 - i * dy
        axA.scatter([lx - 0.012], [y - 0.009], s=58, c=col, transform=axA.transAxes,
                    zorder=6, linewidths=0)
        axA.annotate(f"{name}  ({cnt:,})", (lx + 0.004, y), xycoords="axes fraction",
                     fontsize=11, color=INK2, family="Segoe UI", va="top",
                     path_effects=HALO)

    fig.savefig(OUT, dpi=190, pad_inches=0)
    print("wrote", OUT, "%.2f x %.2f in" % (fw, fh))


if __name__ == "__main__":
    main()
