#!/usr/bin/env python3
"""Pre-labels extracted frames using the *current* transparent_shape_nms.onnx
model, writing standard YOLO-format .txt labels (single class: "shape").

This is the labeling accelerator: instead of drawing boxes on every frame
from scratch in CVAT, you only need to *correct* the current model's
mistakes (add missed shapes, delete false positives, tighten loose boxes) —
which also happens to directly target the current model's actual failure
modes.

Bootstrap confidence threshold defaults low (0.15, vs. the ~0.25+ you'd use
at inference time) to bias toward recall: it's much faster to delete a wrong
box in CVAT than to notice a missing shape and draw it from scratch.

Usage:
    python 02_bootstrap_labels.py \\
        --frames-dir ./dataset/raw_frames \\
        --model ../../backend/resources/transparent_shape_nms.onnx \\
        --labels-dir ./dataset/raw_labels \\
        --conf-threshold 0.15

After this, import each session's images (dataset/raw_frames/<session>/)
into a CVAT task, then upload the matching dataset/raw_labels/<session>/
folder as "YOLO 1.1" format annotations (zip labels/*.txt + obj.names
together) and correct them by hand.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import onnxruntime as ort

from common import run_onnx_detect, CLASS_NAME


def write_obj_names(labels_root: Path):
    (labels_root / "obj.names").write_text(f"{CLASS_NAME}\n")


def write_yolo_label(txt_path: Path, boxes, img_w: int, img_h: int):
    lines = []
    for x1, y1, x2, y2, _conf in boxes:
        cx = ((x1 + x2) / 2) / img_w
        cy = ((y1 + y2) / 2) / img_h
        w = (x2 - x1) / img_w
        h = (y2 - y1) / img_h
        lines.append(f"0 {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
    txt_path.write_text("\n".join(lines) + ("\n" if lines else ""))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--frames-dir", default=Path("./dataset/raw_frames"), type=Path)
    parser.add_argument("--model", default=Path("../../backend/resources/transparent_shape_nms.onnx"), type=Path,
                         help="ONNX model to bootstrap labels with (defaults to the current shipped model; "
                              "point this at your own latest export for later iteration rounds)")
    parser.add_argument("--labels-dir", default=Path("./dataset/raw_labels"), type=Path)
    parser.add_argument("--conf-threshold", type=float, default=0.15)
    args = parser.parse_args()

    if not args.model.exists():
        raise SystemExit(f"Model not found: {args.model}")

    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    session = ort.InferenceSession(str(args.model), providers=providers)
    print(f"Loaded {args.model} with providers: {session.get_providers()}")

    sessions_dirs = [p for p in sorted(args.frames_dir.iterdir()) if p.is_dir()]
    if not sessions_dirs:
        raise SystemExit(f"No session subfolders found under {args.frames_dir} (run 01_extract_frames.py first)")

    total_frames = 0
    total_boxes = 0
    for session_dir in sessions_dirs:
        out_dir = args.labels_dir / session_dir.name
        out_dir.mkdir(parents=True, exist_ok=True)
        write_obj_names(out_dir)

        frame_paths = sorted(session_dir.glob("*.png"))
        for frame_path in frame_paths:
            img = cv2.imread(str(frame_path), cv2.IMREAD_COLOR)
            if img is None:
                continue
            boxes = run_onnx_detect(session, img, conf_threshold=args.conf_threshold)
            h, w = img.shape[:2]
            write_yolo_label(out_dir / (frame_path.stem + ".txt"), boxes, w, h)
            total_boxes += len(boxes)

        total_frames += len(frame_paths)
        print(f"[{session_dir.name}] labeled {len(frame_paths)} frame(s) -> {out_dir}")

    print(f"\nDone. {total_frames} frame(s), {total_boxes} bootstrap box(es) total.")
    print("Next: import each session's images + labels into CVAT and correct them by hand,")
    print("then export the corrected YOLO annotations and run 03_prepare_dataset.py.")


if __name__ == "__main__":
    main()
