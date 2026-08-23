#!/usr/bin/env python3
"""Flags frames whose ground-truth boxes visually overlap/merge with each
other, so a labeling pass can specifically target the "shapes merging"
failure mode instead of only ever surfacing whatever the current model's
own bootstrap+correction cycle happens to catch.

Context: `detect_transparent_shapes`'s model is single-class ("is there a
shape here", not "which one is the target" - see shape.rs's own comment on
`Detector::detect_transparent_shapes`), so which shape is the puzzle's real
target varies round to round and isn't recorded in these YOLO labels at
all. Any two ground-truth boxes overlapping is therefore exactly the right
proxy for "a case the detector needs to still emit two separate boxes for"
- it doesn't matter which of the two happens to be the target in that
particular round, since the model can't and shouldn't condition on that.

None of the existing tooling here (prioritize_review.py, find_false_positives.py,
compare_models.py) looks at ground-truth boxes' relationship to *each other*
at all - only at how well predictions match ground truth individually. This
is the gap: a correction pass focused on "does the model see the boxes I
expect" has no reason to specifically notice that two of those boxes happen
to be touching, and `prioritize_review.py`'s own numbers show a large
fraction of frames per session were never reviewed at all, so an
overlap-heavy frame could easily have sailed through untouched.

Two overlap measures are computed per pair, since shapes can differ
noticeably in size (a small decoy fully swallowed by a larger one has low
IoU but a high overlap fraction of its own area):
  - IoU: standard intersection-over-union.
  - overlap fraction: intersection / area of the *smaller* box - closer to
    "how much of the smaller shape is obscured", regardless of the size
    ratio between the two.
A pair is flagged if either measure clears its own threshold.

Usage:
    python find_overlapping_shapes.py \\
        --labels-dir ./dataset/corrected_labels \\
        --frames-dir ./dataset/raw_frames \\
        --cvat-host http://192.168.68.92:8080 \\
        --cvat-username admin --cvat-password '...' \\
        --thumbnails-dir ./dataset/overlap_thumbnails \\
        ld_zSk2iCFC ld_zmahDi0v   # sessions to check; see --help on `sessions`
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2

from compare_models import load_ground_truth


def cvat_frame_url(cvat_host: str, task_id: int, job_id: int, frame_idx: int) -> str:
    return f"{cvat_host}/tasks/{task_id}/jobs/{job_id}?frame={frame_idx}"


def pair_overlap(
    a: tuple[float, float, float, float], b: tuple[float, float, float, float]
) -> tuple[float, float]:
    """Returns (iou, overlap_fraction_of_smaller_box) for two xyxy boxes."""
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    iou_val = inter / union if union > 0 else 0.0
    min_area = min(area_a, area_b)
    overlap_frac = inter / min_area if min_area > 0 else 0.0
    return iou_val, overlap_frac


def find_session_overlaps(
    session: str, frames_dir: Path, labels_dir: Path,
    iou_threshold: float, overlap_threshold: float, thumbnails_dir: Path | None,
    task_id: int | None, job_id: int | None, cvat_host: str,
) -> list[dict]:
    results = []

    for frame_idx, img_path in enumerate(sorted(frames_dir.glob("*.png"))):
        img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if img is None:
            continue
        h, w = img.shape[:2]
        gt_boxes = load_ground_truth(labels_dir / (img_path.stem + ".txt"), w, h)
        if len(gt_boxes) < 2:
            continue

        for i in range(len(gt_boxes)):
            for j in range(i + 1, len(gt_boxes)):
                iou_val, overlap_frac = pair_overlap(gt_boxes[i], gt_boxes[j])
                if iou_val < iou_threshold and overlap_frac < overlap_threshold:
                    continue

                entry = dict(
                    session=session, frame=img_path.name, frame_idx=frame_idx,
                    box_a=tuple(round(v) for v in gt_boxes[i]),
                    box_b=tuple(round(v) for v in gt_boxes[j]),
                    iou=round(iou_val, 3), overlap_frac=round(overlap_frac, 3),
                )
                if task_id is not None and job_id is not None:
                    entry["cvat_url"] = cvat_frame_url(cvat_host, task_id, job_id, frame_idx)

                if thumbnails_dir is not None:
                    pad = 20
                    ax1, ay1, ax2, ay2 = gt_boxes[i]
                    bx1, by1, bx2, by2 = gt_boxes[j]
                    ux1, uy1 = min(ax1, bx1), min(ay1, by1)
                    ux2, uy2 = max(ax2, bx2), max(ay2, by2)
                    cx1, cy1 = max(0, int(ux1) - pad), max(0, int(uy1) - pad)
                    cx2, cy2 = min(w, int(ux2) + pad), min(h, int(uy2) + pad)
                    crop = img[cy1:cy2, cx1:cx2].copy()
                    cv2.rectangle(
                        crop, (int(ax1) - cx1, int(ay1) - cy1), (int(ax2) - cx1, int(ay2) - cy1),
                        (0, 0, 255), 1,
                    )
                    cv2.rectangle(
                        crop, (int(bx1) - cx1, int(by1) - cy1), (int(bx2) - cx1, int(by2) - cy1),
                        (255, 0, 0), 1,
                    )
                    thumbnails_dir.mkdir(parents=True, exist_ok=True)
                    thumb_name = f"{session}_{img_path.stem}_pair{i}-{j}_iou{iou_val:.2f}.png"
                    cv2.imwrite(str(thumbnails_dir / thumb_name), crop)
                    entry["thumbnail"] = thumb_name

                results.append(entry)

    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("sessions", nargs="+",
                         help="Session names with real human-corrected ground truth. Required and "
                              "not auto-discovered on purpose - same reasoning as find_false_positives.py: "
                              "--labels-dir contains a folder for every session (cvat_export.py re-exports "
                              "untouched bootstrap boxes for sessions you haven't corrected yet), and "
                              "there's no way to tell those apart from real corrections just by looking "
                              "at the files. Passing an uncorrected session just re-surfaces whatever "
                              "overlaps the bootstrap model's own (possibly wrong) boxes happened to have, "
                              "not verified ground truth.")
    parser.add_argument("--frames-dir", default=Path("./dataset/raw_frames"), type=Path)
    parser.add_argument("--labels-dir", default=Path("./dataset/corrected_labels"), type=Path)
    parser.add_argument("--iou-threshold", type=float, default=0.1,
                         help="Flag a pair if their IoU is at least this (default 0.1 - "
                              "deliberately low since even modest overlap is enough to visually "
                              "merge two shapes, unlike prediction-vs-ground-truth matching where "
                              "0.5 means 'the same box'.)")
    parser.add_argument("--overlap-threshold", type=float, default=0.3,
                         help="Flag a pair if intersection covers at least this fraction of the "
                              "smaller box's own area (default 0.3) - catches a small shape mostly "
                              "obscured by a larger one, which can have low IoU despite being "
                              "meaningfully merged from the smaller shape's perspective.")
    parser.add_argument("--thumbnails-dir", type=Path, default=Path("./dataset/overlap_thumbnails"))
    parser.add_argument("--cvat-host", default=None, help="e.g. http://192.168.68.92:8080; omit to skip CVAT links")
    parser.add_argument("--cvat-username", default=None)
    parser.add_argument("--cvat-password", default=None)
    args = parser.parse_args()

    sessions = args.sessions

    task_ids: dict[str, tuple[int, int]] = {}
    cvat_host = None
    if args.cvat_host and args.cvat_username and args.cvat_password:
        from cvat_sdk import make_client
        cvat_host = args.cvat_host
        with make_client(host=args.cvat_host, credentials=(args.cvat_username, args.cvat_password)) as client:
            for t in client.tasks.list():
                session_name = t.bug_tracker if t.bug_tracker in sessions else t.name
                if session_name in sessions:
                    jobs = t.get_jobs()
                    if jobs:
                        task_ids[session_name] = (t.id, jobs[0].id)

    all_results = []
    for session in sessions:
        task_id, job_id = task_ids.get(session, (None, None))
        results = find_session_overlaps(
            session, args.frames_dir / session, args.labels_dir / session,
            args.iou_threshold, args.overlap_threshold, args.thumbnails_dir, task_id, job_id, cvat_host,
        )
        results.sort(key=lambda r: -max(r["iou"], r["overlap_frac"]))
        all_results.extend(results)

        if results:
            print(f"\n[{session}] {len(results)} overlapping pair(s), most-overlapping first:")
            for r in results:
                line = (
                    f"  frame={r['frame']} iou={r['iou']} overlap_frac={r['overlap_frac']} "
                    f"box_a={r['box_a']} box_b={r['box_b']}"
                )
                if "cvat_url" in r:
                    line += f"  {r['cvat_url']}"
                if "thumbnail" in r:
                    line += f"  thumb={r['thumbnail']}"
                print(line)
        else:
            print(f"[{session}] no overlapping pairs found")

    print(f"\nTotal: {len(all_results)} overlapping pair(s) across {len(sessions)} session(s)")
    if args.thumbnails_dir.exists():
        print(f"Thumbnails saved to {args.thumbnails_dir}")
    print(
        "\nNote: this dataset's labels are single-class (no target/decoy distinction - see this "
        "script's module docstring), so every flagged pair is a general shape-on-shape merge case, "
        "not specifically a target-overlaps-decoy one. That's intentional: the detector needs to "
        "keep emitting two boxes for any overlapping pair, regardless of which one is the target "
        "in a given puzzle round."
    )


if __name__ == "__main__":
    main()
