"""
make_test_image.py
Generates a synthetic device photo for testing the Streamlit app.
Simulates: white device body + coloured sample window + reference sticker.
Run: python make_test_image.py
"""
import colorsys
import numpy as np
import cv2

# ── image resolution (simulates a phone photo at ~50 px/mm) ───────────────────
PPM   = 50          # pixels per mm
W_MM  = 30          # device width  (mm)
H_MM  = 40          # device height (mm)
IW    = W_MM * PPM
IH    = H_MM * PPM

def mm(x): return int(x * PPM)

def hsv_rgb(h, s=0.65, v=0.82):
    r, g, b = colorsys.hsv_to_rgb(h / 360, s, v)
    return (int(b*255), int(g*255), int(r*255))   # BGR

# ── device body ────────────────────────────────────────────────────────────────
img = np.full((IH, IW, 3), (220, 220, 220), dtype=np.uint8)   # light grey body
cv2.rectangle(img, (0, 0), (IW-1, IH-1), (80, 80, 80), mm(0.5))  # border

# ── sample window (11 mm × 5.3 mm, 2 mm from top, centred) ────────────────────
MARGIN  = mm(2)
WIN_X   = mm(9.5)      # centre the 11 mm window in 30 mm body
WIN_Y   = mm(2)
WIN_W   = mm(11)
WIN_H   = mm(5.3)

# fill with a test hue (mid-concentration ≈ 291°)
TEST_HUE = 291.0
win_color = hsv_rgb(TEST_HUE)
cv2.rectangle(img, (WIN_X, WIN_Y), (WIN_X+WIN_W, WIN_Y+WIN_H), win_color, -1)
cv2.rectangle(img, (WIN_X, WIN_Y), (WIN_X+WIN_W, WIN_Y+WIN_H), (60,60,60), 1)

# ── sticker (11 mm × 11 mm, 2 mm below window) ────────────────────────────────
STK_X  = WIN_X
STK_Y  = WIN_Y + WIN_H + mm(2)
STK_W  = WIN_W
STK_H  = mm(11)

BORDER_PX  = mm(0.4)
DIVIDER_PX = mm(0.3)
COLS, ROWS = 3, 2

avail_w = STK_W - 2*BORDER_PX - (COLS-1)*DIVIDER_PX
avail_h = STK_H - 2*BORDER_PX - (ROWS-1)*DIVIDER_PX
PW = avail_w // COLS
PH = avail_h // ROWS

# same colours as the sticker PNG
PATCHES = [
    [(235,235,235), (128,128,128), (25,25,25)],     # neutral row (RGB)
    [hsv_rgb(302),  hsv_rgb(291),  hsv_rgb(280)],   # chromatic row (BGR already)
]
# neutral patches are RGB tuples — convert to BGR
PATCHES[0] = [(b, g, r) for (r, g, b) in PATCHES[0]]

cv2.rectangle(img, (STK_X, STK_Y), (STK_X+STK_W, STK_Y+STK_H), (255,255,255), -1)
cv2.rectangle(img, (STK_X, STK_Y), (STK_X+STK_W, STK_Y+STK_H), (60,60,60), 1)

for r in range(ROWS):
    for c in range(COLS):
        px = STK_X + BORDER_PX + c*(PW+DIVIDER_PX)
        py = STK_Y + BORDER_PX + r*(PH+DIVIDER_PX)
        color = PATCHES[r][c]
        cv2.rectangle(img, (px, py), (px+PW, py+PH), color, -1)

# add slight noise to simulate camera
noise = np.random.normal(0, 6, img.shape).astype(np.int16)
img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

out = "test_device.jpg"
cv2.imwrite(out, img, [cv2.IMWRITE_JPEG_QUALITY, 92])
print(f"Saved: {out}  ({IW}x{IH} px,  {W_MM}x{H_MM} mm at {PPM} px/mm)")
print(f"Test hue injected: {TEST_HUE}° → expected concentration ≈ exp((hue-298.78)/-3.0353)")
import math
ln_c = (TEST_HUE - 298.78) / -3.0353
print(f"  ln(conc) = {ln_c:.4f},  conc = {math.exp(ln_c):.4f} ug/mL")
