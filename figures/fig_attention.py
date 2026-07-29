#!/usr/bin/env python3
"""fig_attention.py -- the additive attention gate tested at rung (e1).

The module leader's standing direction is that every model and task used should
carry an architecture figure. Figure 3 draws the network; this draws the ONE block
that changes when `attention: true` is set, which Figure 3 can only mention in
seven words on a skip connection. The two figures are not variants of each other:
Figure 3 is the whole U-Net at stage resolution, this is a single gate at tensor
resolution.

Transcribed from code/model.py::AttentionGate, not from the paper, so what is drawn
is what ran:

    a = sigmoid(psi(relu(wg(g) + wx(x))))
    return x * a

with wg and wx 1x1 convolutions C -> C/2 carrying no bias, psi a 1x1 convolution
C/2 -> 1 with bias, and the resulting single-channel coefficient map broadcast back
over all C channels of the skip. `g` is the decoder signal AFTER its transpose
convolution, so both inputs are already at the same resolution and no resampling
appears in the diagram.

Canvas note: W is what sets on-page type size, because the plate is placed to the
167.6 mm measure. At W = 1300 one SVG unit prints at 0.366 pt, so the 15-unit
labels here land near 5.5 pt, in line with the rest of the figure suite. Figure 3
prints its smallest labels at roughly 3.4 pt because its W is 1900.

Output: report/latex/figures/fig_attention.png
Usage:  python fig_attention.py
"""
from pathlib import Path

from fig_architecture import (ARROW, BLUE_RAMP, HEAD, INK, INK2, MUTED,
                              arrow, elbow, render, rrect, text)

W, H = 1300, 620

SKIP_Y, GATE_Y = 152, 296        # the two input rows
MID_Y = 224                      # where they merge
MUL_Y = 468                      # the multiply row
BYPASS_X = 258                   # see build(): clear of both the gating chip and Wg

ACCENT = HEAD["glacier"]         # the gate is a glacier-head-side mechanism
COEFF = HEAD["dist"]             # the coefficient map, distinct from the features
PANEL = "#f4f7fc"
EDGE = "#c9d8ee"


def chip(x, y, w, h, title, sub, stroke=EDGE, fill=PANEL):
    return "".join([
        rrect(x, y, w, h, fill, rx=10, stroke=stroke, sw=1.4),
        text(x + w / 2, y + h / 2 - 4, title, 17, INK, weight=600),
        text(x + w / 2, y + h / 2 + 20, sub, 14, MUTED),
    ])


def op(cx, cy, glyph, r=23, colour=ARROW):
    """A circled operator: the add and the elementwise multiply."""
    return "".join([
        f'<circle cx="{cx}" cy="{cy}" r="{r}" fill="#ffffff" '
        f'stroke="{colour}" stroke-width="2"/>',
        text(cx, cy + 8, glyph, 24, colour, weight=600),
    ])


def build():
    s = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" '
         f'viewBox="0 0 {W} {H}">',
         rrect(0, 0, W, H, "#ffffff", rx=0)]

    s.append(text(40, 50, "Additive attention gate (rung e1)", 22, INK,
                  anchor="start", weight=700))
    s.append(text(40, 78, "one skip connection when attention is enabled · "
                  "code/model.py::AttentionGate", 15, INK2, anchor="start"))

    # ---- inputs
    s.append(chip(40, SKIP_Y - 38, 190, 76, "skip x", "encoder, C ch"))
    s.append(chip(40, GATE_Y - 38, 190, 76, "gating g", "decoder, C ch"))

    # ---- the two 1x1 projections
    # Named Wx / Wg after the attributes in model.py rather than set as sub- or
    # superscripts, which rendered inconsistently between the two rows.
    for y, label in ((SKIP_Y, "Wx · 1×1 conv"), (GATE_Y, "Wg · 1×1 conv")):
        s.append(rrect(292, y - 32, 186, 64, "#ffffff", rx=9, stroke=ACCENT, sw=1.8))
        s.append(text(385, y - 4, label, 16, INK, weight=600))
        s.append(text(385, y + 19, "C → C/2, no bias", 13.5, MUTED))
        s.append(arrow(232, y, 284, y))

    # ---- merge, ReLU, psi, sigmoid
    s.append(elbow([(480, SKIP_Y), (520, SKIP_Y), (520, MID_Y - 26)]))
    s.append(elbow([(480, GATE_Y), (520, GATE_Y), (520, MID_Y + 26)]))
    s.append(op(520, MID_Y, "+"))

    s.append(arrow(545, MID_Y, 592, MID_Y))
    s.append(rrect(596, MID_Y - 30, 118, 60, "#ffffff", rx=9, stroke=ACCENT, sw=1.8))
    s.append(text(655, MID_Y + 6, "ReLU", 17, INK, weight=600))

    s.append(arrow(716, MID_Y, 762, MID_Y))
    s.append(rrect(766, MID_Y - 32, 196, 64, "#ffffff", rx=9, stroke=ACCENT, sw=1.8))
    s.append(text(864, MID_Y - 4, "ψ · 1×1 conv", 16, INK, weight=600))
    s.append(text(864, MID_Y + 19, "C/2 → 1 channel", 13.5, MUTED))

    s.append(arrow(964, MID_Y, 1010, MID_Y))
    s.append(rrect(1014, MID_Y - 30, 112, 60, "#ffffff", rx=9, stroke=COEFF, sw=1.8))
    s.append(text(1070, MID_Y + 6, "σ", 22, INK, weight=600))

    # ---- the coefficient map drops to the multiply row
    s.append(elbow([(1070, MID_Y + 34), (1070, MUL_Y - 27)], color=COEFF, width=2.4))
    s.append(text(1088, (MID_Y + MUL_Y) / 2 + 6,
                  "α ∈ (0,1), 1 channel", 15, COEFF, anchor="start"))

    # ---- the skip itself bypasses the whole computation and is re-weighted.
    # It branches off the x -> Wx arrow at BYPASS_X, not out of the chip's underside:
    # dropping from under the chip would run the line straight through the gating
    # chip below it. BYPASS_X sits right of that chip (ends x=230) and left of Wg
    # (starts x=292), so the descent crosses nothing.
    s.append(elbow([(BYPASS_X, SKIP_Y), (BYPASS_X, MUL_Y), (1043, MUL_Y)],
                   color=ARROW, width=2.4))
    s.append(f'<circle cx="{BYPASS_X}" cy="{SKIP_Y}" r="5" fill="{ARROW}"/>')
    s.append(op(1070, MUL_Y, "×"))
    s.append(text(320, MUL_Y - 18, "the skip tensor itself, unchanged", 15, MUTED,
                  anchor="start"))

    s.append(arrow(1095, MUL_Y, 1150, MUL_Y))
    s.append(rrect(1154, MUL_Y - 38, 116, 76, PANEL, rx=10, stroke=EDGE, sw=1.4))
    s.append(text(1212, MUL_Y - 6, "x · α", 18, INK, weight=600))
    s.append(text(1212, MUL_Y + 18, "C ch", 13.5, MUTED))

    # ---- the one sentence that says why the block exists
    s.append(rrect(40, H - 78, 1220, 54, "#fafaf9", rx=9, stroke="#e1e0d9", sw=1))
    s.append(text(62, H - 45,
                  "α is one coefficient per pixel, broadcast over all C channels: the "
                  "gate suppresses background regions of the skip before it is",
                  15, INK2, anchor="start"))
    s.append(text(62, H - 25,
                  "concatenated into the decoder. Rung (e1) traded glacier IoU for debris "
                  "IoU and was not adopted; see Table 2 and Section 6.",
                  15, INK2, anchor="start"))

    s.append("</svg>")
    return "".join(s)


def main():
    out = Path(__file__).resolve().parent.parent / "latex" / "figures" / "fig_attention.png"
    render(build(), out, W, H, stem="fig_attention")


if __name__ == "__main__":
    main()
