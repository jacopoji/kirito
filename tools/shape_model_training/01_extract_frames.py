#!/usr/bin/env python3
"""Scans raw gameplay recordings for Lie Detector minigame footage and crops
the fixed solving region out of each active frame.

Usage:
    python 01_extract_frames.py \\
        --recordings-dir "G:/aaaNiubi/25.0.1/app-debug-mule/dataset/recordings" \\
        --resources-dir ../../backend/resources \\
        --out-dir ./dataset/raw_frames \\
        --sample-every 4

Each recording gets its own subfolder under --out-dir so later steps can
split train/val by *session* instead of by frame (adjacent frames are near-
duplicates; splitting by frame would leak near-identical images across the
split and make validation metrics meaningless).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2

from common import find_popup_region, crop_region, load_templates, REGION_SIZE

VIDEO_EXTENSIONS = {".mp4", ".mkv", ".avi", ".mov", ".flv", ".webm"}


def iter_recordings(recordings_dir: Path):
    for path in sorted(recordings_dir.rglob("*")):
        if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS:
            yield path


def extract_from_video(
    video_path: Path, templates, out_dir: Path, sample_every: int, popup_check_every: int,
) -> int:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"  [skip] could not open {video_path}")
        return 0

    out_dir.mkdir(parents=True, exist_ok=True)

    frame_idx = 0
    saved = 0
    active_region = None  # sticky once popup is found; re-checked periodically
    frames_since_check = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        # Only bother re-locating the popup every `popup_check_every` frames
        # once we have a region lock, since the popup box doesn't move once
        # the minigame starts (title stays put; only the shapes move).
        if active_region is None or frames_since_check >= popup_check_every:
            active_region = find_popup_region(frame, templates)
            frames_since_check = 0
        frames_since_check += 1

        if active_region is not None and frame_idx % sample_every == 0:
            crop = crop_region(frame, active_region)
            if crop is not None and crop.shape[1] == REGION_SIZE[0] and crop.shape[0] == REGION_SIZE[1]:
                out_path = out_dir / f"frame_{frame_idx:06d}.png"
                cv2.imwrite(str(out_path), crop)
                saved += 1

        frame_idx += 1

    cap.release()
    return saved


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--recordings-dir", required=True, type=Path)
    parser.add_argument("--resources-dir", default=Path("../../backend/resources"), type=Path,
                         help="Path to backend/resources (for the popup match templates)")
    parser.add_argument("--out-dir", default=Path("./dataset/raw_frames"), type=Path)
    parser.add_argument("--sample-every", type=int, default=4,
                         help="Keep 1 out of every N active frames (dedup near-identical frames)")
    parser.add_argument("--popup-check-every", type=int, default=15,
                         help="Re-run popup template match every N frames once locked on")
    args = parser.parse_args()

    templates = load_templates(args.resources_dir)
    recordings = list(iter_recordings(args.recordings_dir))
    if not recordings:
        raise SystemExit(f"No video files found under {args.recordings_dir}")

    print(f"Found {len(recordings)} recording(s)")
    total_saved = 0
    for video_path in recordings:
        session_name = video_path.stem
        out_dir = args.out_dir / session_name
        print(f"[{session_name}] scanning {video_path} ...")
        saved = extract_from_video(video_path, templates, out_dir, args.sample_every, args.popup_check_every)
        print(f"  saved {saved} frame(s) -> {out_dir}")
        total_saved += saved

    print(f"\nDone. {total_saved} frame(s) extracted across {len(recordings)} recording(s).")
    print(f"Next: python 02_bootstrap_labels.py --frames-dir {args.out_dir} ...")


if __name__ == "__main__":
    main()
