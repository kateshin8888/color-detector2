import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.append(str(ROOT))

import streamlit as st
import cv2
import numpy as np
from PIL import Image
from streamlit_cropper import st_cropper

from detector2 import AnalyzeConfig, analyze_well_image, detect_well_rect, detect_sticker_rect, CAL_M, CAL_B
from color_correction import (
    DEFAULT_PATCHES,
    locate_sticker_patches,
    measure_patch,
    compute_correction_matrix,
    apply_correction,
    make_swatch,
    delta_e_simple,
)

DISPLAY_W      = 640
THRESHOLD_CONC = 7.43   # µg/mL


def bgr_to_rgb(img: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(page_title="SalivADetector", layout="centered")
st.title("SalivADetector")
st.caption("Lactoferrin concentration analysis · Alzheimer's screening POCT")

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Advanced settings")
    with st.expander("Analysis parameters", expanded=False):
        summary    = st.selectbox("Hue summary", ["median", "trimmed_mean", "circular_mean"])
        roi_shrink = st.slider("ROI shrink", 0.60, 1.00, 0.90, 0.01)
        s_min      = st.slider("S min", 0, 255, 20)
        v_min      = st.slider("V min", 0, 255, 40)
        v_max      = st.slider("V max", 0, 255, 240)
    with st.expander("Highlight removal", expanded=False):
        highlight_s_max = st.slider("Highlight S max", 0, 255, 60)
        highlight_v_min = st.slider("Highlight V min", 0, 255, 220)
    with st.expander("Device calibration", expanded=False):
        st.caption(
            "The calibration curve was measured in a well plate. "
            "If your POCT device gives systematically different hue readings, "
            "set an offset here to correct for it."
        )
        if "hue_offset" not in st.session_state:
            st.session_state.hue_offset = 0.0
        st.slider("Hue offset (°)", -30.0, 30.0, step=0.5, key="hue_offset")
        st.caption(
            f"Applied: **{st.session_state.hue_offset:+.1f}°**  "
            "(positive = device reads lower than well plate)"
        )
        st.divider()
        st.markdown("**One-point calibration**")
        st.caption(
            "Photograph a sample of *known* concentration in the POCT device, "
            "press Analyze, then enter the true concentration below and click **Set offset**."
        )
        known_cal_conc = st.number_input(
            "Known concentration (µg/mL)", min_value=0.001, value=7.43,
            format="%.3f", key="known_cal_conc",
        )
        col_cal, col_rst = st.columns(2)
        if col_cal.button("Set offset", use_container_width=True):
            if st.session_state.get("last_raw_hue") is not None:
                expected = CAL_M * float(np.log(known_cal_conc)) + CAL_B
                st.session_state.hue_offset = float(expected - st.session_state.last_raw_hue)
                st.rerun()
            else:
                st.warning("Analyze a sample first, then set the offset.")
        if col_rst.button("Reset to 0", use_container_width=True):
            st.session_state.hue_offset = 0.0
            st.rerun()

cfg = AnalyzeConfig(
    summary=summary,
    roi_shrink=float(roi_shrink),
    s_min=int(s_min), v_min=int(v_min), v_max=int(v_max),
    highlight_s_max=int(highlight_s_max),
    highlight_v_min=int(highlight_v_min),
    threshold_concentration=THRESHOLD_CONC,
)

# ══════════════════════════════════════════════════════════════════════════════
# STEP 1 — Upload
# ══════════════════════════════════════════════════════════════════════════════
st.divider()
st.markdown("###  Step 1 — Upload photo")

uploaded = st.file_uploader(
    "Photo of the POCT device (sample window + colour sticker visible)",
    type=["jpg", "jpeg", "png"],
)
if uploaded is None:
    st.info("Upload a photo of the front face of the device ")
    st.stop()

file_bytes = uploaded.getvalue()
arr = np.frombuffer(file_bytes, np.uint8)
img_bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
if img_bgr is None:
    st.error("Failed to decode image.")
    st.stop()

img_h, img_w = img_bgr.shape[:2]
scale     = DISPLAY_W / img_w
display_h = int(img_h * scale)
pil_disp  = Image.fromarray(bgr_to_rgb(img_bgr)).resize((DISPLAY_W, display_h), Image.LANCZOS)

# ══════════════════════════════════════════════════════════════════════════════
# STEP 2 — Confirm detection
# ══════════════════════════════════════════════════════════════════════════════
st.divider()
st.markdown("###  Step 2 — Confirm detection")

# ── Detect sticker first; predict window from physical geometry ────────────────
sticker_rect_detected = detect_sticker_rect(img_bgr)

if sticker_rect_detected:
    _sx, _sy, _sw, _ = sticker_rect_detected
    _ppm = _sw / 11.0
    _pred_h = int(round(5.3 * _ppm))
    _pred_y = max(0, int(round(_sy - 2.0 * _ppm - _pred_h)))
    auto_rect: tuple[int, int, int, int] = (_sx, _pred_y, _sw, _pred_h)
else:
    auto_rect = detect_well_rect(img_bgr)
    if auto_rect is None:
        auto_rect = (img_w // 4, img_h // 4, img_w // 2, img_h // 4)

# ── Window cropper (collapsed by default — user expands to adjust) ─────────────
with st.expander("Adjust sample window", expanded=False):
    st.caption("Drag the **green box** to cover the sample window.")

    def _win_box_algo(_img=None, **_) -> dict:
        ax, ay, aw, ah = auto_rect
        return {"left": int(ax*scale), "top": int(ay*scale),
                "width": int(aw*scale), "height": int(ah*scale)}

    box = st_cropper(
        pil_disp, realtime_update=True, box_color="#00e676", aspect_ratio=None,
        return_type="box", box_algorithm=_win_box_algo,
        should_resize_image=False, stroke_width=2, key="roi_cropper",
    )

window_rect: tuple[int, int, int, int] = (
    max(0, int(box["left"]   / scale)),
    max(0, int(box["top"]    / scale)),
    max(1, int(box["width"]  / scale)),
    max(1, int(box["height"] / scale)),
)

# Analysis zone (shrunk inward)
rx, ry, rw, rh = window_rect
shrink_x = int(rw * (1.0 - cfg.roi_shrink) / 2)
shrink_y = int(rh * (1.0 - cfg.roi_shrink) / 2)
x1 = max(0,     rx + shrink_x)
y1 = max(0,     ry + shrink_y)
x2 = min(img_w, rx + rw - shrink_x)
y2 = min(img_h, ry + rh - shrink_y)

# ── Colour correction (silent) ─────────────────────────────────────────────────
if "show_colour_ref" not in st.session_state:
    st.session_state.show_colour_ref = False
if "sticker_rect_manual" not in st.session_state:
    st.session_state.sticker_rect_manual = None

known_rgbs: list[tuple[int, int, int]] = [p.known_rgb for p in DEFAULT_PATCHES]
patch_names = [p.name for p in DEFAULT_PATCHES]

sticker_rect_for_patches = st.session_state.sticker_rect_manual or sticker_rect_detected
patch_rects = locate_sticker_patches(window_rect, (img_h, img_w), sticker_rect=sticker_rect_for_patches)

observed_rgbs: list[np.ndarray] = []
patch_errors:  list[float]      = []
sticker_ok = True

for prect, known in zip(patch_rects, known_rgbs):
    try:
        obs = measure_patch(img_bgr, prect)
        observed_rgbs.append(obs)
        patch_errors.append(delta_e_simple(obs, known))
    except Exception:
        sticker_ok = False
        break

MAX_DE_FOR_CORRECTION = 60.0
correction_matrix = None
mean_err  = None
residual  = None
img_corrected = img_bgr

if sticker_ok and len(observed_rgbs) == 6:
    mean_err = float(np.mean(patch_errors))
    if mean_err <= MAX_DE_FOR_CORRECTION:
        correction_matrix, residual = compute_correction_matrix(observed_rgbs, known_rgbs)
        img_corrected = apply_correction(img_bgr, correction_matrix)

# ── Build overlay image ────────────────────────────────────────────────────────
overlay = img_bgr.copy()
cv2.rectangle(overlay, (rx, ry), (rx+rw, ry+rh), (0, 0, 255), 2)          # red: window
cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 220, 0), 2)                # green: analysis zone
if sticker_rect_detected:
    _d = sticker_rect_detected
    cv2.rectangle(overlay, (_d[0], _d[1]), (_d[0]+_d[2], _d[1]+_d[3]), (255, 200, 0), 2)  # cyan: sticker
row_colors = [(200, 200, 200), (180, 100, 255)]
for _j, prect in enumerate(patch_rects):
    px, py, pw, ph = prect
    cv2.rectangle(overlay, (px, py), (px+pw, py+ph), row_colors[_j // 3], 1)
    cv2.putText(overlay, str(_j+1), (px+2, py+ph-3),
                cv2.FONT_HERSHEY_SIMPLEX, max(0.25, pw/60), (255, 255, 255), 1, cv2.LINE_AA)

# ── Detection overview (compact, always visible) ───────────────────────────────
ov_col, info_col = st.columns([4, 6])

with ov_col:
    st.image(bgr_to_rgb(overlay), use_container_width=True,
             caption="green = window · cyan = sticker")

with info_col:
    # Colour correction status
    if correction_matrix is not None:
        src = "manual" if st.session_state.sticker_rect_manual else ("auto" if sticker_rect_detected else "geometry")
        st.success(f"Colour correction active — dE {mean_err:.1f}  ({src})", icon="✅")
    elif mean_err is not None and mean_err > MAX_DE_FOR_CORRECTION:
        st.warning(f"Colour correction skipped — dE too high ({mean_err:.0f})", icon="⚠️")
    else:
        st.warning("Colour correction unavailable — sticker not found", icon="⚠️")

    if st.button("Adjust sticker", use_container_width=True):
        st.session_state.show_colour_ref = not st.session_state.show_colour_ref

    if not sticker_rect_detected:
        st.caption("Sticker not auto-detected. Use **Adjust sticker** to set it manually.")

# ── Sticker detail / adjustment panel ─────────────────────────────────────────
if st.session_state.show_colour_ref:
    with st.container(border=True):
        st.markdown("**Sticker adjustment**")
        st.caption("Drag the **cyan box** over the printed colour patches.")

        _si = st.session_state.sticker_rect_manual or sticker_rect_detected
        if _si is None:
            _ppm2 = rw / 11.0
            _si = (rx, int(ry + rh + 2.0 * _ppm2), int(11.0 * _ppm2), int(11.0 * _ppm2))

        def _sticker_box_algo(_img=None, **_) -> dict:
            six, siy, siw, sih = _si
            return {"left": int(six*scale), "top": int(siy*scale),
                    "width": int(siw*scale), "height": int(sih*scale)}

        sticker_box = st_cropper(
            pil_disp, realtime_update=True, box_color="#00bcd4", aspect_ratio=None,
            return_type="box", box_algorithm=_sticker_box_algo,
            should_resize_image=False, stroke_width=2, key="sticker_cropper",
        )
        st.session_state.sticker_rect_manual = (
            max(0, int(sticker_box["left"]   / scale)),
            max(0, int(sticker_box["top"]    / scale)),
            max(1, int(sticker_box["width"]  / scale)),
            max(1, int(sticker_box["height"] / scale)),
        )
        if st.button("Reset to auto-detect"):
            st.session_state.sticker_rect_manual = None
            st.rerun()

        if sticker_ok and len(observed_rgbs) == 6:
            cols = st.columns(6)
            for _k, col in enumerate(cols):
                obs_t = tuple(int(observed_rgbs[_k][c]) for c in range(3))
                with col:
                    st.caption(patch_names[_k])
                    st.image(make_swatch(known_rgbs[_k], 30), width=30)
                    st.image(make_swatch(obs_t, 30),          width=30)
                    st.caption(f"dE {patch_errors[_k]:.0f}")
            st.caption("Top = target  ·  Bottom = measured")

            ma_col, mb_col = st.columns(2)
            ma_col.metric("Mean dE before", f"{mean_err:.1f}")
            mb_col.metric("Residual after", f"{residual:.1f}" if residual is not None else "skipped")

            pa_col, pb_col = st.columns(2)
            pa_col.image(bgr_to_rgb(img_bgr[y1:y2, x1:x2]),
                         caption="Original", use_container_width=True)
            pb_col.image(bgr_to_rgb(img_corrected[y1:y2, x1:x2]),
                         caption="Corrected", use_container_width=True)
        else:
            st.warning("Could not measure all patches — check window position.")


# ══════════════════════════════════════════════════════════════════════════════
# STEP 3 — Analyze
# ══════════════════════════════════════════════════════════════════════════════
st.divider()
st.markdown("### Step 3 — Analyze")

if st.button("Analyze", type="primary", use_container_width=True):
    try:
        result = analyze_well_image(
            file_bytes=file_bytes,
            cfg=cfg,
            rect=window_rect,
            correction_matrix=correction_matrix,
        )

        if result.get("thresholds_relaxed"):
            st.warning(
                "Low colour saturation — thresholds auto-relaxed. "
                "Check that the green box is over the sample window."
            )

        # Store raw hue for device calibration
        raw_hue = result["hue_deg"]
        st.session_state.last_raw_hue = raw_hue

        # Apply device hue offset
        hue_off = float(st.session_state.get("hue_offset", 0.0))
        hue     = raw_hue + hue_off

        # Recompute concentration from offset-adjusted hue
        ln_conc = (hue - CAL_B) / CAL_M
        conc    = float(np.exp(ln_conc))
        above   = conc >= THRESHOLD_CONC
        in_range = 280 <= hue <= 302

        if conc < 0.001:
            conc_str = f"{conc:.3e}"
        elif conc < 1000:
            conc_str = f"{conc:.2f}"
        else:
            conc_str = f"{conc:.3e}"

        # ── Result card: overlay image | verdict ──────────────────────────────
        st.divider()
        r_img_col, r_res_col = st.columns([4, 6])

        with r_img_col:
            st.image(bgr_to_rgb(overlay), use_container_width=True,
                     caption="corrected" if result.get("color_corrected") else "uncorrected")

        with r_res_col:
            if not in_range:
                st.warning("# ⚠️ Out of range\nHue outside calibration range (280–302°)")
            elif above:
                st.error("# 🔴 ALZHEIMER\n## POSITIVE")
            else:
                st.success("# 🟢 ALZHEIMER\n## NEGATIVE")

            st.metric("Lactoferrin concentration", f"{conc_str} µg/mL")
            st.caption(
                f"Threshold: **{THRESHOLD_CONC:.2f} µg/mL**  ·  Hue: {hue:.1f}°"
                + (f"  ·  offset {hue_off:+.1f}°" if hue_off != 0.0 else "")
            )

        # ── Technical details ─────────────────────────────────────────────────
        with st.expander("Technical details"):
            d1, d2, d3 = st.columns(3)
            d1.metric("Hue (°)",      f"{hue:.2f}")
            d2.metric("Hue std (°)",  f'{result["hue_deg_std"]:.2f}')
            d3.metric("Pixels",       f'{result["n_pixels"]}')
            d4, d5 = st.columns(2)
            d4.metric("ln(conc)",     f"{ln_conc:.4f}")
            d5.metric("Corrected",    "Yes" if result.get("color_corrected") else "No")
            if hue_off != 0.0:
                st.caption(f"Raw hue (pre-offset): {raw_hue:.2f}°")

        with st.expander("Debug (raw result)"):
            st.json({k: v for k, v in result.items() if k not in ["img_bgr", "img_overlay_bgr"]})

    except Exception as e:
        st.error(f"Analysis failed: {e}")
