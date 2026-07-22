#!/usr/bin/env python3
"""Fine-tunes a small YOLO detector on the prepared dataset and exports it to
ONNX with the exact input/output contract backend/src/detect.rs expects
(see common.py's module docstring for the reverse-engineered contract).

Usage:
    python 04_train_and_export.py --data ./dataset/yolo_dataset/dataset.yaml

Defaults to yolo11n (small model matters: this runs every tracker tick
during live solving, on top of everything else the bot's detector does).
Override --base-model to try yolov8n.pt / yolo11s.pt / etc.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import cv2
import onnxruntime as ort
from ultralytics import YOLO

from common import run_onnx_detect, LETTERBOX_SIZE


def contract_self_check(onnx_path: Path, sample_image: Path | None):
    """Runs one sample frame through the exported model and asserts the
    output matches what backend/src/detect.rs's remap_from_yolo/
    from_output_value expect, so a broken export is caught here instead of
    silently producing a non-functional solver after the Rust-side swap.
    """
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])

    inputs = session.get_inputs()
    assert len(inputs) == 1, f"expected exactly 1 input, got {len(inputs)}"
    shape = inputs[0].shape
    assert list(shape[-2:]) == [LETTERBOX_SIZE, LETTERBOX_SIZE], (
        f"expected input spatial dims [{LETTERBOX_SIZE}, {LETTERBOX_SIZE}], got {shape}"
    )
    assert shape[-3] in (3, "3", None), f"expected 3 input channels, got shape {shape}"

    outputs = session.get_outputs()
    assert len(outputs) == 1, (
        f"expected exactly 1 output tensor (NMS-baked export), got {len(outputs)}: "
        f"{[o.name for o in outputs]}. Did you forget nms=True on export?"
    )
    out_shape = outputs[0].shape
    last_dim = out_shape[-1]
    if isinstance(last_dim, int):
        assert last_dim >= 5, (
            f"expected >= 5 output columns (x1,y1,x2,y2,conf,...), got {out_shape}"
        )

    print(f"Contract self-check (shapes): input={shape}, output={out_shape} -- OK")

    if sample_image is not None and sample_image.exists():
        img = cv2.imread(str(sample_image), cv2.IMREAD_COLOR)
        boxes = run_onnx_detect(session, img, conf_threshold=0.15)
        print(f"Contract self-check (inference): {sample_image.name} -> {len(boxes)} box(es) decoded -- OK")
    else:
        print("No sample image given/found for an end-to-end inference check; shape check only.")


def find_any_val_image(data_yaml: Path) -> Path | None:
    val_dir = data_yaml.parent / "images" / "val"
    if val_dir.is_dir():
        images = sorted(val_dir.glob("*.png"))
        if images:
            return images[0]
    return None


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", required=True, type=Path, help="dataset.yaml from 03_prepare_dataset.py")
    parser.add_argument("--base-model", default="yolo11n.pt",
                         help="Pretrained checkpoint to fine-tune from")
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--patience", type=int, default=20, help="Early-stopping patience")
    parser.add_argument("--imgsz", type=int, default=LETTERBOX_SIZE)
    parser.add_argument("--out-onnx", type=Path, default=Path("./dataset/transparent_shape_nms.onnx"),
                         help="Where to copy the final exported .onnx")
    args = parser.parse_args()

    model = YOLO(args.base_model)
    model.train(
        data=str(args.data),
        imgsz=args.imgsz,
        epochs=args.epochs,
        patience=args.patience,
        # Dial down heavy color jitter: the shapes are defined by transparency
        # and edges against the background more than by color, and the
        # existing model/game visuals are fairly consistent.
        hsv_h=0.0,
        hsv_s=0.3,
        hsv_v=0.2,
    )

    export_path = model.export(format="onnx", nms=True, imgsz=args.imgsz, simplify=True)
    export_path = Path(export_path)
    print(f"Exported: {export_path}")

    sample_image = find_any_val_image(args.data)
    contract_self_check(export_path, sample_image)

    args.out_onnx.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(export_path, args.out_onnx)
    print(f"\nCopied final model to {args.out_onnx}")
    print("Next: validate it against backend/resources/transparent_shape_test_{normal,hard}.mp4")
    print("via the debug UI (\"Test transparent shape normal/hard\") before swapping it in — see README.md.")


if __name__ == "__main__":
    main()
