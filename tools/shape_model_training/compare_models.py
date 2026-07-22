#!/usr/bin/env python3
"""Objectively compares two ONNX models' detections against hand-corrected
ground truth for a session, instead of relying on eyeballing the debug UI
(which can't be observed programmatically anyway, since it's a native
OpenCV window driving real mouse input).

Runs both models with conf_threshold=0.0 to match production: detect.rs's
detect_transparent_shapes (backend/src/detect.rs:2757) applies no additional
confidence filtering beyond whatever the NMS-baked export already decided,
so filtering further here would understate/overstate accuracy relative to
what actually runs in the bot.

Matches predicted boxes to ground truth per frame via greedy IoU matching
(highest-IoU-first), then reports precision/recall/F1 and mean IoU of
matched boxes for each model side by side.

Usage:
    python compare_models.py \\
        --frames-dir ./dataset/raw_frames/ld_ZyRDRuun \\
        --labels-dir ./dataset/corrected_labels/ld_ZyRDRuun \\
        --model-a ../../backend/resources/transparent_shape_nms.onnx --model-a-name shipped \\
        --model-b ./dataset/transparent_shape_nms.onnx --model-b-name new
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import onnxruntime as ort

from common import run_onnx_detect


def load_ground_truth(label_path: Path, img_w: int, img_h: int) -> list[tuple[float, float, float, float]]:
    if not label_path.exists() or label_path.stat().st_size == 0:
        return []
    boxes = []
    for line in label_path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        _cls, cx, cy, w, h = (float(x) for x in line.split())
        cx, cy, w, h = cx * img_w, cy * img_h, w * img_w, h * img_h
        boxes.append((cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2))
    return boxes


def iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def match_greedy(preds: list[tuple[float, float, float, float]],
                  gts: list[tuple[float, float, float, float]],
                  iou_threshold: float) -> tuple[int, int, int, list[float]]:
    """Returns (true_positives, false_positives, false_negatives, matched_ious)."""
    pairs = []
    for pi, p in enumerate(preds):
        for gi, g in enumerate(gts):
            score = iou(p, g)
            if score >= iou_threshold:
                pairs.append((score, pi, gi))
    pairs.sort(reverse=True)

    matched_pred, matched_gt = set(), set()
    matched_ious = []
    for score, pi, gi in pairs:
        if pi in matched_pred or gi in matched_gt:
            continue
        matched_pred.add(pi)
        matched_gt.add(gi)
        matched_ious.append(score)

    tp = len(matched_pred)
    fp = len(preds) - tp
    fn = len(gts) - len(matched_gt)
    return tp, fp, fn, matched_ious


def evaluate_model(model_path: Path, frames_dir: Path, labels_dir: Path,
                    conf_threshold: float, iou_threshold: float) -> dict:
    session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])

    total_tp = total_fp = total_fn = 0
    all_ious = []
    for img_path in sorted(frames_dir.glob("*.png")):
        img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        h, w = img.shape[:2]
        preds = run_onnx_detect(session, img, conf_threshold=conf_threshold)
        pred_boxes = [(x1, y1, x2, y2) for x1, y1, x2, y2, _conf in preds]

        gt_boxes = load_ground_truth(labels_dir / (img_path.stem + ".txt"), w, h)

        tp, fp, fn, ious = match_greedy(pred_boxes, gt_boxes, iou_threshold)
        total_tp += tp
        total_fp += fp
        total_fn += fn
        all_ious.extend(ious)

    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) else float("nan")
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) else float("nan")
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else float("nan")
    mean_iou = sum(all_ious) / len(all_ious) if all_ious else float("nan")

    return dict(tp=total_tp, fp=total_fp, fn=total_fn,
                precision=precision, recall=recall, f1=f1, mean_iou=mean_iou)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--frames-dir", required=True, type=Path)
    parser.add_argument("--labels-dir", required=True, type=Path,
                         help="Ground truth YOLO labels (e.g. dataset/corrected_labels/<session>)")
    parser.add_argument("--model-a", required=True, type=Path)
    parser.add_argument("--model-a-name", default="A")
    parser.add_argument("--model-b", required=True, type=Path)
    parser.add_argument("--model-b-name", default="B")
    parser.add_argument("--conf-threshold", type=float, default=0.0,
                         help="0.0 matches production (detect.rs applies no extra filtering "
                              "beyond the model's own NMS-baked export)")
    parser.add_argument("--iou-threshold", type=float, default=0.5)
    args = parser.parse_args()

    for path, name in [(args.model_a, args.model_a_name), (args.model_b, args.model_b_name)]:
        result = evaluate_model(path, args.frames_dir, args.labels_dir, args.conf_threshold, args.iou_threshold)
        print(f"[{name}] {path}")
        print(f"  TP={result['tp']} FP={result['fp']} FN={result['fn']}")
        print(f"  precision={result['precision']:.3f} recall={result['recall']:.3f} "
              f"f1={result['f1']:.3f} mean_iou_matched={result['mean_iou']:.3f}")


if __name__ == "__main__":
    main()
