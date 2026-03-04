"""
generate_reference_chart.py
============================
Generates a print-ready PNG of the SalivADetector colour reference chart.

Calibration curve:  Hue (°) = -3.0353 × ln(conc) + 298.78
Assay hue range:    280° – 302°  (blue-violet → magenta-purple)

6 patches
  • 3 neutral  : White / Mid-Gray / Black   (correct exposure + white balance)
  • 3 chromatic: Hue 280° / 291° / 302°    (anchors in the assay colour range)

Print at 100 % on a colour laser printer (no scaling).
After printing, photograph the card under your reference lighting, measure
the actual RGB values with the app, and update the known_rgb entries in
color_correction.py accordingly.
"""

import colorsys
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# ──────────────────────────── helper ───────────────────────────────────────────

def hue_to_rgb(hue_deg: float, s: float = 0.65, v: float = 0.82):
    """Convert HSV (hue in 0-360°, s and v in 0-1) to integer RGB tuple."""
    r, g, b = colorsys.hsv_to_rgb(hue_deg / 360.0, s, v)
    return (round(r * 255), round(g * 255), round(b * 255))


# ──────────────────────────── patch definitions ────────────────────────────────
#
# Each entry: (label_top, label_bot, rgb)
#   label_top  – name printed inside the patch
#   label_bot  – RGB value printed below the patch
#   rgb        – target digital RGB (what gets printed ideally)
#
# These RGB values are written into color_correction.py as DEFAULT_PATCHES so
# the app knows what each patch *should* look like.

H_LOW  = 302.0   # hue at low  concentration (ln(conc) ≈ -1)
H_MID  = 291.0   # hue at mid  concentration (ln(conc) ≈  2.3)
H_HIGH = 280.0   # hue at high concentration (ln(conc) ≈  6.3)

PATCHES = [
    # ── neutral ──────────────────────────────────────────
    ("White",     (235, 235, 235)),
    ("Mid Gray",  (128, 128, 128)),
    ("Black",     ( 25,  25,  25)),
    # ── chromatic (assay range) ───────────────────────────
    (f"Hue {H_LOW:.0f}°\n(low conc)",  hue_to_rgb(H_LOW)),
    (f"Hue {H_MID:.0f}°\n(mid conc)",  hue_to_rgb(H_MID)),
    (f"Hue {H_HIGH:.0f}°\n(high conc)", hue_to_rgb(H_HIGH)),
]

print("Patch RGB values:")
for label, rgb in PATCHES:
    print(f"  {label.split(chr(10))[0]:20s}  RGB{rgb}")

# ──────────────────────────── layout (300 DPI) ─────────────────────────────────

DPI = 300

def mm(x):
    return int(x * DPI / 25.4)


N          = len(PATCHES)
PATCH_W    = mm(14)     # each patch width
PATCH_H    = mm(20)     # each patch height (colour area)
LABEL_H    = mm(14)     # text area below patch
GAP        = mm(4)      # gap between patches
BORDER     = mm(5)      # outer margin
TITLE_H    = mm(12)     # space for title + subtitle

TOTAL_W = BORDER * 2 + PATCH_W * N + GAP * (N - 1)
TOTAL_H = BORDER * 2 + TITLE_H + PATCH_H + LABEL_H

BG        = (250, 250, 250)
FG        = ( 30,  30,  30)
SUBTLE    = (120, 120, 120)
DIVIDER   = (180, 180, 180)

img  = Image.new("RGB", (TOTAL_W, TOTAL_H), BG)
draw = ImageDraw.Draw(img)

# ── fonts ─────────────────────────────────────────────────────────────────────
def load_font(names, size):
    for name in names:
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            pass
    return ImageFont.load_default()

font_title = load_font(["arialbd.ttf", "Arial Bold.ttf", "DejaVuSans-Bold.ttf"], mm(4))
font_sub   = load_font(["arial.ttf",   "Arial.ttf",      "DejaVuSans.ttf"],      mm(2.8))
font_patch = load_font(["arialbd.ttf", "Arial Bold.ttf", "DejaVuSans-Bold.ttf"], mm(2.6))
font_rgb   = load_font(["arial.ttf",   "Arial.ttf",      "DejaVuSans.ttf"],      mm(2.2))

# ── title ─────────────────────────────────────────────────────────────────────
draw.text(
    (TOTAL_W // 2, BORDER + mm(2)),
    "SalivADetector — Colour Reference Chart",
    fill=FG, font=font_title, anchor="mt",
)
draw.text(
    (TOTAL_W // 2, BORDER + mm(7)),
    f"Print at 100 %  ·  {N} patches  ·  {TOTAL_W/DPI*25.4:.0f} mm × {TOTAL_H/DPI*25.4:.0f} mm",
    fill=SUBTLE, font=font_sub, anchor="mt",
)

# divider line between neutral / chromatic groups
divider_x = BORDER + PATCH_W * 3 + GAP * 3 - GAP // 2
draw.line(
    [(divider_x, BORDER + TITLE_H - mm(1)),
     (divider_x, BORDER + TITLE_H + PATCH_H + mm(1))],
    fill=DIVIDER, width=2,
)

# ── patches ───────────────────────────────────────────────────────────────────
for i, (label, rgb) in enumerate(PATCHES):
    x0 = BORDER + i * (PATCH_W + GAP)
    y0 = BORDER + TITLE_H

    # colour rectangle
    draw.rectangle([x0, y0, x0 + PATCH_W - 1, y0 + PATCH_H - 1],
                   fill=rgb, outline=FG, width=1)

    # patch name inside the swatch (white text on dark, dark text on light)
    brightness = 0.299 * rgb[0] + 0.587 * rgb[1] + 0.114 * rgb[2]
    text_col = (255, 255, 255) if brightness < 140 else (30, 30, 30)
    lines = label.split("\n")
    for j, line in enumerate(lines):
        draw.text(
            (x0 + PATCH_W // 2, y0 + PATCH_H // 2 + (j - len(lines) / 2 + 0.5) * mm(3.5)),
            line, fill=text_col, font=font_patch, anchor="mm",
        )

    # RGB label below patch
    draw.text(
        (x0 + PATCH_W // 2, y0 + PATCH_H + mm(3)),
        f"RGB({rgb[0]}, {rgb[1]}, {rgb[2]})",
        fill=FG, font=font_rgb, anchor="mt",
    )

    # hex label
    draw.text(
        (x0 + PATCH_W // 2, y0 + PATCH_H + mm(7)),
        f"#{rgb[0]:02X}{rgb[1]:02X}{rgb[2]:02X}",
        fill=SUBTLE, font=font_rgb, anchor="mt",
    )

# group labels
draw.text((BORDER + PATCH_W * 1 + GAP, BORDER + TITLE_H + PATCH_H + mm(11)),
          "<  neutral patches  >",
          fill=SUBTLE, font=font_sub, anchor="mt")
draw.text((divider_x + GAP + PATCH_W * 1, BORDER + TITLE_H + PATCH_H + mm(11)),
          "<  assay range patches  >",
          fill=SUBTLE, font=font_sub, anchor="mt")

# ── cut guide border ──────────────────────────────────────────────────────────
draw.rectangle([1, 1, TOTAL_W - 2, TOTAL_H - 2], outline=(100, 100, 100), width=3)

# ── save ──────────────────────────────────────────────────────────────────────
out = "reference_chart.png"
img.save(out, dpi=(DPI, DPI))
print(f"\nSaved: {out}")
print(f"Print size: {TOTAL_W/DPI*25.4:.1f} mm × {TOTAL_H/DPI*25.4:.1f} mm  at {DPI} DPI")
print("\nAfter printing, photograph the card under your standard lighting,")
print("measure the actual patch colours with the app, and update")
print("DEFAULT_PATCHES in salivaidapp/color_correction.py.")
