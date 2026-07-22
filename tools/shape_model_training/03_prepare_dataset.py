#!/usr/bin/env python3
"""Assembles a corrected per-session frames+labels tree into an Ultralytics
YOLO dataset layout, splitting train/val by *session* (recording) so
near-duplicate consecutive frames never leak across the split.

Expects, after you've corrected the bootstrap labels in CVAT and exported
them back out (YOLO 1.1 format), a labels tree that mirrors the frames tree:

    dataset/raw_frames/<session>/frame_000000.png
    dataset/corrected_labels/<session>/frame_000000.txt   (one per image;
                                                             empty file if a
                                                             frame genuinely
                                                             has no shapes)

Usage:
    python 03_prepare_dataset.py \\
        --frames-dir ./dataset/raw_frames \\
        --labels-dir ./dataset/corrected_labels \\
        --out-dir ./dataset/yolo_dataset \\
        --val-split 0.2
"""

from __future__ import annotations

import argparse
import random
import shutil
from pathlib import Path

from common import CLASS_NAME


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--frames-dir", default=Path("./dataset/raw_frames"), type=Path)
    parser.add_argument("--labels-dir", default=Path("./dataset/corrected_labels"), type=Path)
    parser.add_argument("--out-dir", default=Path("./dataset/yolo_dataset"), type=Path)
    parser.add_argument("--val-split", type=float, default=0.2,
                         help="Fraction of *sessions* (not frames) held out for validation")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    frame_sessions = {p.name for p in args.frames_dir.iterdir() if p.is_dir()}
    label_sessions = {p.name for p in args.labels_dir.iterdir() if p.is_dir()}
    sessions = sorted(frame_sessions & label_sessions)

    missing_labels = frame_sessions - label_sessions
    if missing_labels:
        print(f"Warning: {len(missing_labels)} session(s) have frames but no corrected labels, skipping: "
              f"{sorted(missing_labels)}")
    if not sessions:
        raise SystemExit("No session present in both --frames-dir and --labels-dir.")

    rng = random.Random(args.seed)
    shuffled = sessions[:]
    rng.shuffle(shuffled)
    n_val = max(1, round(len(shuffled) * args.val_split)) if len(shuffled) > 1 else 0
    val_sessions = set(shuffled[:n_val])
    train_sessions = set(shuffled[n_val:])

    print(f"{len(sessions)} session(s): {len(train_sessions)} train, {len(val_sessions)} val")

    for split, split_sessions in (("train", train_sessions), ("val", val_sessions)):
        images_out = args.out_dir / "images" / split
        labels_out = args.out_dir / "labels" / split
        images_out.mkdir(parents=True, exist_ok=True)
        labels_out.mkdir(parents=True, exist_ok=True)

        n_images = 0
        n_with_boxes = 0
        for session in sorted(split_sessions):
            frame_dir = args.frames_dir / session
            label_dir = args.labels_dir / session
            for img_path in sorted(frame_dir.glob("*.png")):
                label_path = label_dir / (img_path.stem + ".txt")
                if not label_path.exists():
                    continue  # not yet corrected/exported for this frame, skip

                dest_stem = f"{session}_{img_path.stem}"
                shutil.copy2(img_path, images_out / f"{dest_stem}.png")
                shutil.copy2(label_path, labels_out / f"{dest_stem}.txt")

                n_images += 1
                if label_path.stat().st_size > 0:
                    n_with_boxes += 1

        print(f"  [{split}] {n_images} image(s) copied ({n_with_boxes} with at least one box)")

    yaml_path = args.out_dir / "dataset.yaml"
    yaml_path.write_text(
        f"path: {args.out_dir.resolve()}\n"
        f"train: images/train\n"
        f"val: images/val\n"
        f"nc: 1\n"
        f"names: ['{CLASS_NAME}']\n"
    )
    print(f"\nWrote {yaml_path}")
    print(f"Next: python 04_train_and_export.py --data {yaml_path}")


if __name__ == "__main__":
    main()
