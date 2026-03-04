"""
generate_sticker.py
====================
Generates a print-ready 11 mm × 11 mm colour reference sticker
for the SalivADetector POCT device (blue area below the sample window).

Layout — 2 rows × 3 columns:
  Row 1 (top):    White  |  Mid Gray  |  Black          ← neutral patches
  Row 2 (bottom): Hue302 |  Hue291    |  Hue280         ← assay-range patches

Print at exactly 100 % scale on a colour laser printer.
No scaling, no fit-to-page.
"""

import colorsys
from PIL import Image, ImageDraw

# ──────────────────────────── patches (same order as DEFAULT_PATCHES) ──────────

def hue_to_rgb(h, s=0.65, v=0.82):
    r, g, b = colorsys.hsv_to_rgb(h / 360.0, s, v)
    return (round(r * 255), round(g * 255), round(b * 255))

# Row 1 – neutral,  Row 2 – chromatic
GRID = [
    [(235, 235, 235), (128, 128, 128), (25,  25,  25 )],   # White / Mid Gray / Black
    [hue_to_rgb(302), hue_to_rgb(291), hue_to_rgb(280)],   # low / mid / high conc
]

ROWS = len(GRID)      # 2
COLS = len(GRID[0])   # 3

# ──────────────────────────── dimensions ───────────────────────────────────────

DPI     = 1200
def mm(x): return int(round(x * DPI / 25.4))

STICKER_W = mm(11)
STICKER_H = mm(11)

BORDER  = mm(0.4)   # outer white margin (also cut guide)
DIVIDER = mm(0.3)   # gap between patches

# compute patch sizes from available space
avail_w = STICKER_W - 2 * BORDER - (COLS - 1) * DIVIDER
avail_h = STICKER_H - 2 * BORDER - (ROWS - 1) * DIVIDER
PW = avail_w // COLS
PH = avail_h // ROWS

# ──────────────────────────── draw ────────────────────────────────────────────

img  = Image.new("RGB", (STICKER_W, STICKER_H), (255, 255, 255))
draw = ImageDraw.Draw(img)

for r, row in enumerate(GRID):
    for c, rgb in enumerate(row):
        x0 = BORDER + c * (PW + DIVIDER)
        y0 = BORDER + r * (PH + DIVIDER)
        x1 = x0 + PW - 1
        y1 = y0 + PH - 1
        draw.rectangle([x0, y0, x1, y1], fill=rgb)

# blue detection border (thick, used by app for auto-detection)
BLUE = (0, 0, 200)
BORDER_PX = mm(0.5)
draw.rectangle([0, 0, STICKER_W - 1, STICKER_H - 1], outline=BLUE, width=BORDER_PX)

# thin divider line between neutral row and chromatic row
div_y = BORDER + PH + DIVIDER // 2
draw.line([(BORDER, div_y), (STICKER_W - BORDER, div_y)], fill=(180, 180, 180), width=1)

# ──────────────────────────── save ────────────────────────────────────────────

out = "reference_sticker.png"
img.save(out, dpi=(DPI, DPI))

print(f"Saved: {out}")
print(f"Size:  {STICKER_W}x{STICKER_H} px  =  {STICKER_W/DPI*25.4:.1f} mm x {STICKER_H/DPI*25.4:.1f} mm  at {DPI} DPI")
print()
print("Patch grid (top-left to bottom-right):")
print("  Row 1:  White          Mid Gray       Black")
print("  Row 2:  Hue302 low     Hue291 mid     Hue280 high")
print()
print("RGB values:")
for r, row in enumerate(GRID):
    label = "Neutral   " if r == 0 else "Chromatic "
    for c, rgb in enumerate(row):
        print(f"  [{r},{c}] {label}  RGB{rgb}  #{rgb[0]:02X}{rgb[1]:02X}{rgb[2]:02X}")
