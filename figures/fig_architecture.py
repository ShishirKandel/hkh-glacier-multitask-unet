#!/usr/bin/env python3
"""fig_architecture.py -- generate the multi-task U-Net architecture figure.

Emits a hand-designed flat SVG (no plotting library), wraps it in HTML, and
renders it to report/latex/figures/fig_architecture.png with headless Edge.
Palette: report-wide categorical/sequential scheme (blue ramp for feature maps;
blue/orange/aqua for the three task heads).

Usage: python fig_architecture.py
"""
import math
import subprocess
import tempfile
from pathlib import Path

# ---------------------------------------------------------------- palette
BLUE_RAMP = ["#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95"]  # depth 0..4
HEAD = {"glacier": "#2a78d6", "type": "#eb6834", "dist": "#1baf7a"}
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
ARROW, SKIP = "#52514e", "#898781"
FONT = "'Segoe UI', 'Inter', system-ui, sans-serif"

# 960, not 1000: dropping the path labels and the spatial-size note emptied the
# band above the legend, and leaving it would print as a white gutter. The width
# is what sets on-page type size (the plate is placed to the measure), so this
# only trims dead height.
W, H = 1900, 960

# stages: (channels, spatial at 256^2 train crop)
ENC = [(32, 256), (64, 128), (128, 64), (256, 32)]
BOT = (512, 16)
DEC = [(256, 32), (128, 64), (64, 128), (32, 256)]  # bottom -> top

ROW_C = [200, 372, 522, 650, 756]        # y centre per depth row (0..3 stages, 4 bottleneck)
ENC_X = [340, 480, 615, 745]             # left edge per encoder stage
BOT_X = 880
DEC_X = [1060, 1180, 1310, 1450]         # left edge per decoder stage, bottom -> top


def bar_h(s):
    return 46 + 104 * (s / 256) ** 0.8


def bar_w(ch):
    return 18 + 10 * math.log2(ch / 32) if ch >= 32 else 18


def rrect(x, y, w, h, fill, rx=5, stroke="none", sw=0, opacity=1.0):
    s = f'stroke="{stroke}" stroke-width="{sw}"' if sw else 'stroke="none"'
    return (f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" '
            f'rx="{rx}" fill="{fill}" {s} opacity="{opacity}"/>')


def text(x, y, s, size=15, fill=INK2, anchor="middle", weight=400, spacing=None):
    ls = f' letter-spacing="{spacing}"' if spacing else ""
    return (f'<text x="{x:.1f}" y="{y:.1f}" font-family="{FONT}" font-size="{size}" '
            f'fill="{fill}" text-anchor="{anchor}" font-weight="{weight}"{ls}>{s}</text>')


def arrow(x1, y1, x2, y2, color=ARROW, width=2.2, dash=None, marker="arrow"):
    d = f' stroke-dasharray="{dash}"' if dash else ""
    return (f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" '
            f'stroke="{color}" stroke-width="{width}"{d} marker-end="url(#{marker})"/>')


def elbow(pts, color=ARROW, width=2.2, radius=11, marker="arrow"):
    """Orthogonal connector through pts, corners rounded by `radius`.

    Every segment is axis-aligned: the route leaves a block, runs straight, turns
    once through 90 degrees and runs straight into the next. Diagonal connectors
    read as auto-generated; right-angled ones read as drafted.
    """
    if len(pts) < 2:
        return ""
    d = ["M%.1f,%.1f" % pts[0]]
    for i in range(1, len(pts) - 1):
        (x0, y0), (x1, y1), (x2, y2) = pts[i - 1], pts[i], pts[i + 1]
        l1 = math.hypot(x1 - x0, y1 - y0)
        l2 = math.hypot(x2 - x1, y2 - y1)
        if l1 < 1 or l2 < 1:
            continue
        r = min(radius, l1 / 2, l2 / 2)
        d.append("L%.1f,%.1f" % (x1 - (x1 - x0) / l1 * r, y1 - (y1 - y0) / l1 * r))
        d.append("Q%.1f,%.1f %.1f,%.1f" % (x1, y1,
                                           x1 + (x2 - x1) / l2 * r,
                                           y1 + (y2 - y1) / l2 * r))
    d.append("L%.1f,%.1f" % pts[-1])
    return (f'<path d="{" ".join(d)}" fill="none" stroke="{color}" '
            f'stroke-width="{width}" stroke-linecap="round" '
            f'marker-end="url(#{marker})"/>')


def double_bar(x, yc, ch, depth, up_bar=False):
    """One stage: (optional transpose-conv outline bar) + two conv bars. Returns
    (svg, left_x, right_x, top_y, bottom_y, centre_x)."""
    w, h = bar_w(ch), bar_h(ENC[depth][1] if depth < 4 else BOT[1])
    h = bar_h([e for e in ENC + [BOT]][depth][1])
    y = yc - h / 2
    parts, cx = [], x
    if up_bar:  # ConvTranspose result: outline bar half-width
        parts.append(rrect(cx, y, w * 0.55, h, "#ffffff", stroke=BLUE_RAMP[min(depth, 4)], sw=1.6))
        cx += w * 0.55 + 4
    for _ in range(2):  # the double 3x3 conv block
        parts.append(rrect(cx, y, w, h, BLUE_RAMP[min(depth, 4)]))
        cx += w + 4
    return "".join(parts), x, cx - 4, y, y + h


def build():
    defs = f'''<defs>
      <marker id="arrow" viewBox="0 0 10 10" refX="8.5" refY="5" markerWidth="7.5"
              markerHeight="7.5" orient="auto-start-reverse">
        <path d="M 0 1 L 9 5 L 0 9 z" fill="{ARROW}"/></marker>
      <marker id="skiparrow" viewBox="0 0 10 10" refX="8.5" refY="5" markerWidth="8"
              markerHeight="8" orient="auto-start-reverse">
        <path d="M 0 1 L 9 5 L 0 9 z" fill="{SKIP}"/></marker>
      ''' + "".join(
        f'<marker id="head_{k}" viewBox="0 0 10 10" refX="8.5" refY="5" '
        f'markerWidth="7" markerHeight="7" orient="auto-start-reverse">'
        f'<path d="M 0 1 L 9 5 L 0 9 z" fill="{c}"/></marker>'
        for k, c in HEAD.items()) + '''
    </defs>'''
    s = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" '
         f'viewBox="0 0 {W} {H}">', defs,
         rrect(0, 0, W, H, "#ffffff", rx=0)]

    # ------------------------------------------------------------ input chip
    ix, iy, iw, ih = 52, ROW_C[0] - 78, 218, 158
    s.append(rrect(ix, iy, iw, ih, "#f4f7fc", rx=10, stroke="#c9d8ee", sw=1.4))
    s.append(text(ix + iw / 2, iy + 30, "Input patch", 17, INK, weight=600))
    s.append(text(ix + iw / 2, iy + 52, "10 channels (best config)", 14.5, INK2, weight=600))
    for k, line in enumerate(["Landsat-7 optical B1\u2013B5, B7",
                              "thermal B6 (both gains)",
                              "SRTM elevation \u00b7 slope",
                              "channel groups enter at rungs a\u2013b2"]):
        s.append(text(ix + iw / 2, iy + 78 + 22 * k, line, 13.5, MUTED))

    # ------------------------------------------------------------ encoder
    geo = {}
    # Stage labels hang below-LEFT, clear of the vertical drop that leaves each
    # block's centre; centred labels would sit underneath the connector.
    for i, (ch, sp) in enumerate(ENC):
        bars, xl, xr, yt, yb = double_bar(ENC_X[i], ROW_C[i], ch, i)
        s.append(bars)
        cx = (xl + xr) / 2
        geo[("enc", i)] = (xl, xr, yt, yb, cx)
        s.append(text(cx - 14, yb + 30, f"{ch} ch", 19, INK2, anchor="end", weight=600))
        s.append(text(cx - 14, yb + 57, f"{sp}\u00d7{sp}", 17.5, MUTED, anchor="end"))
    # bottleneck
    bars, xl, xr, yt, yb = double_bar(BOT_X, ROW_C[4], BOT[0], 4)
    s.append(bars)
    geo[("bot", 0)] = (xl, xr, yt, yb, (xl + xr) / 2)
    s.append(text((xl + xr) / 2, yb + 24, "512 ch \u00b7 16\u00d716", 16, INK2, weight=600))
    s.append(text((xl + xr) / 2, yb + 47, "bottleneck", 14.5, MUTED))

    # ------------------------------------------------------------ decoder
    for j, (ch, sp) in enumerate(DEC):          # j=0 bottom row (depth 3) .. j=3 top
        depth = 3 - j
        bars, xl, xr, yt, yb = double_bar(DEC_X[j], ROW_C[depth], ch, depth, up_bar=True)
        s.append(bars)
        cx = (xl + xr) / 2
        geo[("dec", depth)] = (xl, xr, yt, yb, cx)
        # below-RIGHT: the riser from the stage below arrives on this block's centre
        s.append(text(cx + 14, yb + 30, f"{ch} ch", 19, INK2, anchor="start", weight=600))
        s.append(text(cx + 14, yb + 57, f"{sp}\u00d7{sp}", 17.5, MUTED, anchor="start"))

    # ------------------------------------------------------------ arrows
    # input -> enc0
    e0 = geo[("enc", 0)]
    s.append(arrow(ix + iw + 8, ROW_C[0], e0[0] - 8, ROW_C[0]))
    # Encoder downsampling: straight DOWN out of the block, one 90-degree turn,
    # straight RIGHT into the left edge of the next stage.
    down = [geo[("enc", i)] for i in range(4)] + [geo[("bot", 0)]]
    for i in range(4):
        a, b = down[i], down[i + 1]
        y = ROW_C[i + 1]
        s.append(elbow([(a[4], a[3] + 7), (a[4], y), (b[0] - 9, y)]))
    # Expanding path: straight RIGHT out of the block, one turn, straight UP into
    # the underside of the next stage. Entering from below keeps the riser off the
    # left edge, which the skip connection already occupies.
    up = [geo[("bot", 0)]] + [geo[("dec", d)] for d in (3, 2, 1, 0)]
    for j in range(4):
        a, b = up[j], up[j + 1]
        y = ROW_C[4 - j]
        s.append(elbow([(a[1] + 7, y), (b[4], y), (b[4], b[3] + 9)]))
    # skip connections
    for i in range(4):
        a, b = geo[("enc", i)], geo[("dec", i)]
        yc = ROW_C[i]
        s.append(arrow(a[1] + 8, yc, b[0] - 8, yc, color=SKIP, width=2,
                       dash="7 6", marker="skiparrow"))
    s.append(text((geo[("enc", 0)][1] + geo[("dec", 0)][0]) / 2, ROW_C[0] - 16,
                  "skip connections (concatenate; optional attention gates)", 16.5, MUTED))
    # arrow labels: third annotation line under the top stages, clear of everything.
    # Pushed to +88 because the spatial-size line above it moved to +57.
    e0, d0g = geo[("enc", 0)], geo[("dec", 0)]
    s.append(text(e0[4] - 14, e0[3] + 88, "\u2193 max-pool 2\u00d72", 17.5, MUTED, anchor="end"))
    s.append(text(d0g[4] + 14, d0g[3] + 88, "\u2191 transpose conv 2\u00d72",
                  17.5, MUTED, anchor="start"))

    # ------------------------------------------------------------ heads
    d0 = geo[("dec", 0)]
    hx, hw, hh = 1580, 268, 74
    hys = [ROW_C[0] - 118, ROW_C[0] - 32, ROW_C[0] + 54]
    labels = [("glacier", "Glacier extent", "1\u00d71 conv \u00b7 BCE + Dice"),
              ("type", "Clean vs debris ice", "1\u00d71 conv \u00b7 BCE + Dice, glacier-masked"),
              ("dist", "Boundary distance", "1\u00d71 conv \u00b7 L1, validity-masked")]
    # The three heads fan out of one turning column, so the connectors stay
    # axis-aligned like the rest of the diagram instead of splaying diagonally.
    xj = 1545
    ystart = [ROW_C[0] - 26, hys[1] + hh / 2, ROW_C[0] + 26]
    for (key, title, sub), hy, y0 in zip(labels, hys, ystart):
        col = HEAD[key]
        yc = hy + hh / 2
        s.append(elbow([(d0[1] + 8, y0), (xj, y0), (xj, yc), (hx - 9, yc)],
                       color=col, width=2.6, radius=9, marker="head_" + key))
        s.append(rrect(hx, hy, hw, hh, "#ffffff", rx=12, stroke=col, sw=2))
        s.append(f'<circle cx="{hx + 24}" cy="{hy + hh / 2:.1f}" r="7" fill="{col}"/>')
        # The title grows; the sub-label cannot. "1x1 conv . BCE + Dice,
        # glacier-masked" already fills the 226 units between hx+42 and the box
        # edge, so raising it past 12.5 overruns the rounded rectangle. Widening
        # the head boxes would move them, which is out of scope here.
        s.append(text(hx + 42, hy + 30, title, 17, INK, anchor="start", weight=600))
        s.append(text(hx + 42, hy + 53, sub, 12.5, INK2, anchor="start"))

    # ------------------------------------------------------------ titles & legend
    # No CONTRACTING/EXPANDING PATH labels and no spatial-size note on the plate:
    # the module leader asked for both to move into the figure's caption, which is
    # where prose about the diagram belongs. The caption in body.md carries them.
    s.append(text(52, 64, "Multi-task U-Net", 24, INK, anchor="start", weight=700))
    s.append(text(52, 92, "encoder depth 4 \u00b7 base 32 \u00b7 \u22487.8M parameters \u00b7 "
                  "shared decoder, three task heads", 17, INK2, anchor="start"))
    ly = H - 66
    s.append(rrect(52, ly - 24, 1210, 58, "#fafaf9", rx=9, stroke="#e1e0d9", sw=1))
    lx = 76
    s.append(rrect(lx, ly - 8, 16, 26, BLUE_RAMP[2]))
    # 14.5, not higher: this is the widest legend label and its slot is only the
    # 330 units before the next swatch.
    s.append(text(lx + 26, ly + 10, "ConvBlock: 2× (3×3 conv + BatchNorm + ReLU)",
                  14.5, INK2, anchor="start"))
    # 355/190/175 rather than 330/190/190: the first label is the widest and needs
    # the extra 25 units at 14.5. The 25 is taken back off the last gap, so the row
    # still ends well inside the panel.
    lx += 355
    s.append(rrect(lx, ly - 8, 10, 26, "#ffffff", stroke=BLUE_RAMP[2], sw=1.6))
    s.append(text(lx + 20, ly + 10, "transpose-conv output", 14.5, INK2, anchor="start"))
    lx += 190
    s.append(f'<line x1="{lx}" y1="{ly + 5}" x2="{lx + 44}" y2="{ly + 5}" stroke="{SKIP}" '
             f'stroke-width="2" stroke-dasharray="7 6" marker-end="url(#skiparrow)"/>')
    s.append(text(lx + 54, ly + 10, "skip connection", 14.5, INK2, anchor="start"))
    lx += 175
    s.append(f'<circle cx="{lx + 8}" cy="{ly + 5}" r="7" fill="{HEAD["glacier"]}"/>')
    s.append(f'<circle cx="{lx + 28}" cy="{ly + 5}" r="7" fill="{HEAD["type"]}"/>')
    s.append(f'<circle cx="{lx + 48}" cy="{ly + 5}" r="7" fill="{HEAD["dist"]}"/>')
    s.append(text(lx + 66, ly + 10, "task heads on the shared 32-ch decoder output",
                  14.5, INK2, anchor="start"))

    s.append("</svg>")
    return "".join(s)


def main():
    svg = build()
    html = ("<!doctype html><html><head><meta charset='utf-8'><style>"
            "html,body{margin:0;padding:0;background:#ffffff;}</style></head>"
            f"<body>{svg}</body></html>")
    tmp = Path(tempfile.gettempdir()) / "fig_architecture.html"
    tmp.write_text(html, encoding="utf-8")
    out = Path(__file__).resolve().parent.parent / "latex" / "figures" / "fig_architecture.png"
    edge = r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"
    subprocess.run([edge, "--headless", "--disable-gpu",
                    f"--screenshot={out}", f"--window-size={W},{H}",
                    "--force-device-scale-factor=2", "--hide-scrollbars",
                    tmp.as_uri()], check=True, timeout=120)
    print("wrote", out)


if __name__ == "__main__":
    main()
