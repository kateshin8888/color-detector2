from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np

# ──────────────────────────── calibration constants ────────────────────────────
# Model: Hue_deg = CAL_M * ln(concentration) + CAL_B
# Inverse: ln(concentration) = (Hue_deg - CAL_B) / CAL_M
CAL_M: float = -3.0353
CAL_B: float = 298.78

# rect = (x, y, w, h)  in pixel coordinates of the original image
Rect = Tuple[int, int, int, int]


@dataclass
class AnalyzeConfig:
    roi_shrink: float = 0.85          # shrink each side of detected rect by this fraction
    s_min: int = 40                   # minimum saturation (0-255)
    v_min: int = 40                   # minimum value/brightness (0-255)
    v_max: int = 240                  # maximum value (exclude over-exposed pixels)
    highlight_s_max: int = 60         # specular highlight: s <= this ...
    highlight_v_min: int = 220        # ... and v >= this → excluded
    summary: str = "median"           # "median" | "trimmed_mean" | "circular_mean"
    threshold_concentration: Optional[float] = None


# ──────────────────────────── rectangle auto-detection ─────────────────────────

# Known physical aspect ratio of the sample window (11 mm × 5.3 mm)
_WINDOW_ASPECT = 11.0 / 5.3   # ≈ 2.075


def detect_well_rect(
    img_bgr: np.ndarray,
    min_area_ratio: float = 0.005,
    max_area_ratio: float = 0.50,
) -> Optional[Rect]:
    """Auto-detect the rectangular sample window using two strategies.

    Strategy 1 (preferred): red-border detection.
        If the device window is outlined in red, a HSV red-colour mask
        is used to find the border reliably regardless of lighting.

    Strategy 2 (fallback): edge-based detection.
        Canny edges (auto-threshold) → morphological close → contour
        fitting with approxPolyDP.  Candidates are scored by how
        closely their aspect ratio matches the known window geometry
        (11 mm × 5.3 mm ≈ 2.08 : 1).

    Returns (x, y, w, h) or None.
    """
    img_h, img_w = img_bgr.shape[:2]
    img_area = img_h * img_w

    def _score(_x, _y, w, h) -> float:
        """Higher is better; returns -1 if candidate is out of bounds."""
        area = w * h
        if area < min_area_ratio * img_area or area > max_area_ratio * img_area:
            return -1.0
        aspect = w / h if h > 0 else 0.0
        aspect_score = 1.0 / (1.0 + abs(aspect - _WINDOW_ASPECT))
        area_score   = min(area / (0.15 * img_area), 1.0)
        return aspect_score * area_score

    candidates: list[tuple[float, Rect]] = []

    # ── Strategy 1: red border ────────────────────────────────────────────
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    h_ch, s_ch, v_ch = cv2.split(hsv)
    red_mask = (
        ((h_ch <= 10) | (h_ch >= 170)) &
        (s_ch > 80) &
        (v_ch > 60)
    ).astype(np.uint8) * 255

    k5 = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    red_closed = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, k5)
    red_cnts, _ = cv2.findContours(red_closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in red_cnts:
        x, y, cw, ch = cv2.boundingRect(cnt)
        s = _score(x, y, cw, ch)
        if s > 0:
            candidates.append((s + 1.0, (x, y, cw, ch)))   # +1 bonus for colour match

    # ── Strategy 2: edge-based ────────────────────────────────────────────
    gray    = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # auto Canny thresholds via image median
    med = float(np.median(blurred))
    lo  = max(0,   int(0.33 * med))
    hi  = min(255, int(1.33 * med))
    edges = cv2.Canny(blurred, lo, hi)

    k7     = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, k7)
    cnts, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in cnts:
        peri  = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.04 * peri, True)
        # prefer quadrilateral fits, fall back to bounding box
        x, y, cw, ch = cv2.boundingRect(approx if len(approx) == 4 else cnt)
        s = _score(x, y, cw, ch)
        if s > 0:
            candidates.append((s, (x, y, cw, ch)))

    if not candidates:
        return None

    candidates.sort(key=lambda c: c[0], reverse=True)
    return candidates[0][1]


def detect_sticker_rect(
    img_bgr: np.ndarray,
    min_area_ratio: float = 0.005,
    max_area_ratio: float = 0.50,
) -> Optional[Rect]:
    """Detect the blue-outlined colour reference sticker.

    Looks for the blue border (HSV hue 100-130) and returns the
    bounding rect of the largest qualifying blue contour.

    Returns (x, y, w, h) or None.
    """
    img_h, img_w = img_bgr.shape[:2]
    img_area = img_h * img_w

    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    h_ch, s_ch, v_ch = cv2.split(hsv)

    # Blue: OpenCV hue 90-140 (≈ 180-280°), tolerant of printer/lighting shift
    blue_mask = (
        (h_ch >= 90) & (h_ch <= 140) &
        (s_ch > 50) &
        (v_ch > 40)
    ).astype(np.uint8) * 255

    k5 = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    blue_closed = cv2.morphologyEx(blue_mask, cv2.MORPH_CLOSE, k5)
    cnts, _ = cv2.findContours(blue_closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    best: Optional[Rect] = None
    best_score = -1.0
    _STICKER_ASPECT = 11.0 / 11.0   # square sticker

    for cnt in cnts:
        x, y, cw, ch = cv2.boundingRect(cnt)
        area = cw * ch
        if area < min_area_ratio * img_area or area > max_area_ratio * img_area:
            continue
        aspect = cw / ch if ch > 0 else 0.0
        score = 1.0 / (1.0 + abs(aspect - _STICKER_ASPECT))
        if score > best_score:
            best_score = score
            best = (x, y, cw, ch)

    return best


# ──────────────────────────── hue summary helpers ──────────────────────────────

def _circular_mean_hue(hue_vals: np.ndarray) -> float:
    """Circular mean for hue values in OpenCV scale [0, 179]."""
    angles = hue_vals * (2.0 * np.pi / 180.0)
    sin_mean = np.mean(np.sin(angles))
    cos_mean = np.mean(np.cos(angles))
    mean_angle = np.arctan2(sin_mean, cos_mean)
    if mean_angle < 0:
        mean_angle += 2.0 * np.pi
    return float(mean_angle * 180.0 / (2.0 * np.pi))


def _summarize_hue(hue_vals: np.ndarray, method: str) -> float:
    if method == "median":
        return float(np.median(hue_vals))
    elif method == "trimmed_mean":
        p10 = np.percentile(hue_vals, 10)
        p90 = np.percentile(hue_vals, 90)
        trimmed = hue_vals[(hue_vals >= p10) & (hue_vals <= p90)]
        return float(np.mean(trimmed)) if len(trimmed) > 0 else float(np.mean(hue_vals))
    elif method == "circular_mean":
        return _circular_mean_hue(hue_vals)
    else:
        raise ValueError(f"Unknown summary method: {method!r}")


def _circular_std_deg(hue_vals_cv: np.ndarray) -> float:
    """Circular standard deviation of hue in degrees (0-360 scale)."""
    angles = hue_vals_cv * (2.0 * np.pi / 180.0)
    r_len = float(np.abs(np.mean(np.exp(1j * angles))))
    r_len = max(r_len, 1e-12)
    return float(np.sqrt(-2.0 * np.log(r_len))) * (180.0 / np.pi)


# ──────────────────────────── main analysis function ───────────────────────────

def analyze_well_image(
    file_bytes: bytes,
    cfg: AnalyzeConfig,
    rect: Optional[Rect] = None,           # (x, y, w, h)
    correction_matrix: Optional[np.ndarray] = None,  # (3, 4) affine RGB matrix
) -> dict:
    """Analyze a rectangular sample window from raw image bytes.

    Parameters
    ----------
    correction_matrix : optional (3, 4) float32 array from color_correction.py
        If provided, the image is colour-corrected before hue extraction.

    Returns a dict with:
        rect_used, hue_deg, hue_deg_std, n_pixels,
        ln_concentration_est, concentration_est,
        color_corrected, img_bgr, img_overlay_bgr,
        (optional) threshold_concentration, above_threshold
    """
    # 1. decode
    arr = np.frombuffer(file_bytes, np.uint8)
    img_bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise ValueError("Cannot decode image.")
    img_h, img_w = img_bgr.shape[:2]

    # 1b. apply colour correction if provided
    color_corrected = correction_matrix is not None
    if color_corrected:
        from color_correction import apply_correction
        img_bgr = apply_correction(img_bgr, correction_matrix)

    # 2. determine rect
    if rect is None:
        rect = detect_well_rect(img_bgr)
    if rect is None:
        # fallback: centre 50% of the image
        rect = (img_w // 4, img_h // 4, img_w // 2, img_h // 2)

    rx, ry, rw, rh = rect

    # 3. apply roi_shrink (shrink each edge inward)
    sx = int(rw * (1.0 - cfg.roi_shrink) / 2)
    sy = int(rh * (1.0 - cfg.roi_shrink) / 2)
    x1 = max(0, rx + sx)
    y1 = max(0, ry + sy)
    x2 = min(img_w, rx + rw - sx)
    y2 = min(img_h, ry + rh - sy)

    if x2 <= x1 or y2 <= y1:
        raise ValueError("ROI collapsed after shrinking — lower roi_shrink value.")

    # 4. rectangular mask
    mask_roi = np.zeros((img_h, img_w), dtype=bool)
    mask_roi[y1:y2, x1:x2] = True

    # 5. HSV
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    hue_ch, sat_ch, val_ch = cv2.split(hsv)

    # 6. pixel filters
    def _apply_filters(s_min, v_min, v_max):
        mask_sv = (sat_ch >= s_min) & (val_ch >= v_min) & (val_ch <= v_max)
        mask_hi = (sat_ch <= cfg.highlight_s_max) & (val_ch >= cfg.highlight_v_min)
        return mask_roi & mask_sv & ~mask_hi

    mask_valid = _apply_filters(cfg.s_min, cfg.v_min, cfg.v_max)
    thresholds_relaxed = False

    if not np.any(mask_valid):
        # auto-relax: try with minimal thresholds before giving up
        mask_valid = _apply_filters(s_min=10, v_min=10, v_max=250)
        thresholds_relaxed = True
        if not np.any(mask_valid):
            raise ValueError(
                "No valid pixels found in the selected area. "
                "Make sure the green crop box covers the sample window, not the device body."
            )

    valid_h = hue_ch[mask_valid].astype(np.float32)

    # 7. hue summary and stats
    hue_cv = _summarize_hue(valid_h, cfg.summary)
    hue_deg = hue_cv * 2.0
    hue_deg_std = _circular_std_deg(valid_h)
    n_pixels = int(np.sum(mask_valid))

    # 8. calibration: ln(conc) = (hue_deg - CAL_B) / CAL_M
    if abs(CAL_M) < 1e-12:
        raise ValueError("CAL_M is near zero; check calibration constants.")
    ln_concentration_est = float((hue_deg - CAL_B) / CAL_M)
    concentration_est = float(np.exp(ln_concentration_est))

    # 9. overlay — outer rect (full detected), inner rect (analysis zone)
    img_overlay = img_bgr.copy()
    cv2.rectangle(img_overlay, (rx, ry), (rx + rw, ry + rh), (255, 120, 0), 1)   # blue: full rect
    cv2.rectangle(img_overlay, (x1, y1), (x2, y2), (0, 220, 0), 2)               # green: analysis zone

    result: dict = {
        "rect_used": rect,
        "analysis_rect": (x1, y1, x2 - x1, y2 - y1),
        "color_corrected": color_corrected,
        "thresholds_relaxed": thresholds_relaxed,
        "hue_deg": hue_deg,
        "hue_deg_std": hue_deg_std,
        "n_pixels": n_pixels,
        "ln_concentration_est": ln_concentration_est,
        "concentration_est": concentration_est,
        "img_bgr": img_bgr,
        "img_overlay_bgr": img_overlay,
    }

    # 10. optional threshold
    if cfg.threshold_concentration is not None:
        result["threshold_concentration"] = float(cfg.threshold_concentration)
        result["above_threshold"] = concentration_est >= cfg.threshold_concentration

    return result
