import argparse
import sys
from pathlib import Path
import cv2
import numpy as np


sys.path.insert(0, str(Path(__file__).parent / "salivaidapp"))
from detector2 import AnalyzeConfig, analyze_well_image


def main():
    parser = argparse.ArgumentParser(description="Single-well Hue detector (CLI tester)")
    parser.add_argument("image_path", type=str, help="Path to input image (jpg/png)")
    parser.add_argument("--summary", choices=["median", "trimmed_mean", "circular_mean"], default="median")
    parser.add_argument("--roi-shrink", type=float, default=0.85)

    parser.add_argument("--s-min", type=int, default=40)
    parser.add_argument("--v-min", type=int, default=40)
    parser.add_argument("--v-max", type=int, default=240)

    parser.add_argument("--highlight-s-max", type=int, default=60)
    parser.add_argument("--highlight-v-min", type=int, default=220)
    parser.add_argument("--x", type=int, default=None, help="Manual ROI left edge (pixels)")
    parser.add_argument("--y", type=int, default=None, help="Manual ROI top edge (pixels)")
    parser.add_argument("--width", type=int, default=None, help="Manual ROI width (pixels)")
    parser.add_argument("--height", type=int, default=None, help="Manual ROI height (pixels)")

    parser.add_argument("--threshold-conc", type=float, default=None, help="Decision threshold (concentration)")
    parser.add_argument("--save-overlay", type=str, default=None, help="Save ROI overlay image path")

    args = parser.parse_args()


    with open(args.image_path, "rb") as f:
        file_bytes = f.read()

    cfg = AnalyzeConfig(
        roi_shrink=args.roi_shrink,
        s_min=args.s_min,
        v_min=args.v_min,
        v_max=args.v_max,
        highlight_s_max=args.highlight_s_max,
        highlight_v_min=args.highlight_v_min,
        summary=args.summary,
        threshold_concentration=args.threshold_conc,
    )

    rect = None
    if all(v is not None for v in [args.x, args.y, args.width, args.height]):
        rect = (args.x, args.y, args.width, args.height)

    result = analyze_well_image(file_bytes=file_bytes, cfg=cfg, rect=rect)

    print("=== Hue Detector CLI ===")
    print(f"Image: {args.image_path}")
    print(f"Rect used (x,y,w,h): {result['rect_used']}")
    print(f"Hue (deg): {result['hue_deg']:.2f}")
    print(f"Hue std (deg): {result['hue_deg_std']:.2f}")
    print(f"n pixels: {result['n_pixels']}")
    print(f"ln(concentration): {result['ln_concentration_est']:.6f}")
    print(f"concentration: {result['concentration_est']:.6f}")

    thr = result.get("threshold_concentration")
    above = result.get("above_threshold")
    if thr is not None:
        print(f"threshold concentration: {thr:.6f}")
        print(f"above threshold: {above}")

    # optional overlay save
    if args.save_overlay:
        overlay = result["img_overlay_bgr"]
        cv2.imwrite(args.save_overlay, overlay)
        print(f"Saved overlay to: {args.save_overlay}")


if __name__ == "__main__":
    main()

