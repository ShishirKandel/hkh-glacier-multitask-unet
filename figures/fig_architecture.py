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

W, H = 1900, 1000

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
    for i, (ch, sp) in enumerate(ENC):
        bars, xl, xr, yt, yb = double_bar(ENC_X[i], ROW_C[i], ch, i)
        s.append(bars)
        geo[("enc", i)] = (xl, xr, yt, yb, (xl + xr) / 2)
        s.append(text((xl + xr) / 2, yb + 22, f"{ch} ch", 13.5, INK2, weight=600))
        s.append(text((xl + xr) / 2, yb + 40, f"{sp}\u00d7{sp}", 12.5, MUTED))
    # bottleneck
    bars, xl, xr, yt, yb = double_bar(BOT_X, ROW_C[4], BOT[0], 4)
    s.append(bars)
    geo[("bot", 0)] = (xl, xr, yt, yb, (xl + xr) / 2)
    s.append(text((xl + xr) / 2, yb + 22, "512 ch \u00b7 16\u00d716", 13.5, INK2, weight=600))
    s.append(text((xl + xr) / 2, yb + 40, "bottleneck", 12.5, MUTED))

    # ------------------------------------------------------------ decoder
    for j, (ch, sp) in enumerate(DEC):          # j=0 bottom row (depth 3) .. j=3 top
        depth = 3 - j
        bars, xl, xr, yt, yb = double_bar(DEC_X[j], ROW_C[depth], ch, depth, up_bar=True)
        s.append(bars)
        geo[("dec", depth)] = (xl, xr, yt, yb, (xl + xr) / 2)
        s.append(text((xl + xr) / 2, yb + 22, f"{ch} ch", 13.5, INK2, weight=600))
        s.append(text((xl + xr) / 2, yb + 40, f"{sp}\u00d7{sp}", 12.5, MUTED))

    # ------------------------------------------------------------ arrows
    # input -> enc0
    e0 = geo[("enc", 0)]
    s.append(arrow(ix + iw + 8, ROW_C[0], e0[0] - 8, ROW_C[0]))
    # encoder downsampling
    for i in range(3):
        a, b = geo[("enc", i)], geo[("enc", i + 1)]
        s.append(arrow(a[4] + 14, a[3] + 6, b[4] - 10, b[2] - 6))
    a, b = geo[("enc", 3)], geo[("bot", 0)]
    s.append(arrow(a[4] + 14, a[3] + 6, b[4] - 10, b[2] - 6))
    # bottleneck -> decoder upsampling
    a, b = geo[("bot", 0)], geo[("dec", 3)]
    s.append(arrow(a[4] + 14, a[2] - 6, b[4] - 14, b[3] + 6))
    for depth in (3, 2, 1):
        a, b = geo[("dec", depth)], geo[("dec", depth - 1)]
        s.append(arrow(a[4] + 12, a[2] - 6, b[4] - 14, b[3] + 6))
    # skip connections
    for i in range(4):
        a, b = geo[("enc", i)], geo[("dec", i)]
        yc = ROW_C[i]
        s.append(arrow(a[1] + 8, yc, b[0] - 8, yc, color=SKIP, width=2,
                       dash="7 6", marker="skiparrow"))
    s.append(text((geo[("enc", 0)][1] + geo[("dec", 0)][0]) / 2, ROW_C[0] - 14,
                  "skip connections (concatenate; optional attention gates)", 13.5, MUTED))
    # arrow labels: third annotation line under the top stages, clear of everything
    e0, d0g = geo[("enc", 0)], geo[("dec", 0)]
    s.append(text((e0[0] + e0[1]) / 2, e0[3] + 58, "\u2198 max-pool 2\u00d72", 12.5, MUTED))
    s.append(text((d0g[0] + d0g[1]) / 2, d0g[3] + 58, "\u2197 transpose conv 2\u00d72",
                  12.5, MUTED))

    # ------------------------------------------------------------ heads
    d0 = geo[("dec", 0)]
    hx, hw, hh = 1580, 268, 74
    hys = [ROW_C[0] - 118, ROW_C[0] - 32, ROW_C[0] + 54]
    labels = [("glacier", "Glacier extent", "1\u00d71 conv \u00b7 BCE + Dice"),
              ("type", "Clean vs debris ice", "1\u00d71 conv \u00b7 BCE + Dice, glacier-masked"),
              ("dist", "Boundary distance", "1\u00d71 conv \u00b7 L1, validity-masked")]
    for (key, title, sub), hy in zip(labels, hys):
        col = HEAD[key]
        s.append(arrow(d0[1] + 8, ROW_C[0] + (hy + hh / 2 - ROW_C[0]) * 0.35,
                       hx - 8, hy + hh / 2, color=col, width=2.6))
        s.append(rrect(hx, hy, hw, hh, "#ffffff", rx=12, stroke=col, sw=2))
        s.append(f'<circle cx="{hx + 24}" cy="{hy + hh / 2:.1f}" r="7" fill="{col}"/>')
        s.append(text(hx + 42, hy + 31, title, 15.5, INK, anchor="start", weight=600))
        s.append(text(hx + 42, hy + 53, sub, 12.5, INK2, anchor="start"))

    # ------------------------------------------------------------ titles & legend
    s.append(text(52, 64, "Multi-task U-Net", 24, INK, anchor="start", weight=700))
    s.append(text(52, 90, "encoder depth 4 \u00b7 base 32 \u00b7 \u22487.8M parameters \u00b7 "
                  "shared decoder, three task heads", 14.5, INK2, anchor="start"))
    ly = H - 66
    s.append(text(W - 52, ly - 2, "spatial sizes shown for a 256\u00d7256 training crop;",
                  13, MUTED, anchor="end"))
    s.append(text(W - 52, ly + 16, "fully convolutional: evaluated on full 512\u00d7512 patches",
                  13, MUTED, anchor="end"))
    s.append(rrect(52, ly - 24, 1210, 58, "#fafaf9", rx=9, stroke="#e1e0d9", sw=1))
    lx = 76
    s.append(rrect(lx, ly - 8, 16, 26, BLUE_RAMP[2]))
    s.append(text(lx + 26, ly + 10, "ConvBlock: 2\u00d7 (3\u00d73 conv + BatchNorm + ReLU)",
                  13.5, INK2, anchor="start"))
    lx += 330
    s.append(rrect(lx, ly - 8, 10, 26, "#ffffff", stroke=BLUE_RAMP[2], sw=1.6))
    s.append(text(lx + 20, ly + 10, "transpose-conv output", 13.5, INK2, anchor="start"))
    lx += 190
    s.append(f'<line x1="{lx}" y1="{ly + 5}" x2="{lx + 44}" y2="{ly + 5}" stroke="{SKIP}" '
             f'stroke-width="2" stroke-dasharray="7 6" marker-end="url(#skiparrow)"/>')
    s.append(text(lx + 54, ly + 10, "skip connection", 13.5, INK2, anchor="start"))
    lx += 190
    s.append(f'<circle cx="{lx + 8}" cy="{ly + 5}" r="7" fill="{HEAD["glacier"]}"/>')
    s.append(f'<circle cx="{lx + 28}" cy="{ly + 5}" r="7" fill="{HEAD["type"]}"/>')
    s.append(f'<circle cx="{lx + 48}" cy="{ly + 5}" r="7" fill="{HEAD["dist"]}"/>')
    s.append(text(lx + 66, ly + 10, "task heads on the shared 32-ch decoder output",
                  13.5, INK2, anchor="start"))

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
