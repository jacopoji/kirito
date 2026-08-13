#!/usr/bin/env python3
"""Sanity-checks a new model against the shipped one on sessions with *no*
human-corrected ground truth — so unlike `compare_models.py`, this reports
raw detection counts only (no precision/recall/F1, since there's nothing to
score against). It's a red-flag check: does the new model's behavior stay
in the same ballpark everywhere, or does it start exploding with spurious
detections outside the handful of sessions you've actually corrected?

Runs both models with conf_threshold=0.0 to match production (see
compare_models.py's docstring for why).

Usage:
    python compare_raw_counts.py \\
        --frames-dir ./dataset/raw_frames \\
        --model-a ../../backend/resources/transparent_shape_nms.onnx --model-a-name shipped \\
        --model-b ./dataset/transparent_shape_nms.onnx --model-b-name new \\
        --exclude ld_yMrowLFb ld_zmahDi0v ld_zSk2iCFC ld_ZyRDRuun
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import onnxruntime as ort

from common import run_onnx_detect


def count_detections(session, frames_dir: Path, conf_threshold: float) -> tuple[int, int]:
    total_boxes = 0
    n_frames = 0
    for img_path in sorted(frames_dir.glob("*.png")):
        img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        preds = run_onnx_detect(session, img, conf_threshold=conf_threshold)
        total_boxes += len(preds)
        n_frames += 1
    return total_boxes, n_frames


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--frames-dir", required=True, type=Path)
    parser.add_argument("--model-a", required=True, type=Path)
    parser.add_argument("--model-a-name", default="A")
    parser.add_argument("--model-b", required=True, type=Path)
    parser.add_argument("--model-b-name", default="B")
    parser.add_argument("--conf-threshold", type=float, default=0.0)
    parser.add_argument("--exclude", nargs="*", default=[], help="Session names to skip (e.g. ones already corrected)")
    args = parser.parse_args()

    session_a = ort.InferenceSession(str(args.model_a), providers=["CPUExecutionProvider"])
    session_b = ort.InferenceSession(str(args.model_b), providers=["CPUExecutionProvider"])

    sessions = sorted(p for p in args.frames_dir.iterdir() if p.is_dir() and p.name not in args.exclude)

    print(f"{'session':<16} {'frames':>7} {args.model_a_name + '_boxes':>14} {args.model_b_name + '_boxes':>14} "
          f"{args.model_a_name + '/frame':>12} {args.model_b_name + '/frame':>12} {'ratio':>8}")

    grand_a = grand_b = grand_frames = 0
    for session_dir in sessions:
        boxes_a, n_frames = count_detections(session_a, session_dir, args.conf_threshold)
        boxes_b, _ = count_detections(session_b, session_dir, args.conf_threshold)
        grand_a += boxes_a
        grand_b += boxes_b
        grand_frames += n_frames

        per_a = boxes_a / n_frames if n_frames else float("nan")
        per_b = boxes_b / n_frames if n_frames else float("nan")
        ratio = boxes_b / boxes_a if boxes_a else float("nan")
        flag = "  <-- ratio > 1.5x" if ratio > 1.5 else ""
        print(f"{session_dir.name:<16} {n_frames:>7} {boxes_a:>14} {boxes_b:>14} "
              f"{per_a:>12.2f} {per_b:>12.2f} {ratio:>8.2f}{flag}")

    print("-" * 90)
    overall_ratio = grand_b / grand_a if grand_a else float("nan")
    print(f"{'TOTAL':<16} {grand_frames:>7} {grand_a:>14} {grand_b:>14} "
          f"{grand_a / grand_frames:>12.2f} {grand_b / grand_frames:>12.2f} {overall_ratio:>8.2f}")


if __name__ == "__main__":
    main()
