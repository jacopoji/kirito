#!/usr/bin/env python3
"""Finds a model's false positives against ground truth and makes them easy
to triage: crops a thumbnail around each one and prints a direct CVAT link
to jump straight to that frame, so you can confirm "yep, nothing there" and
correct it directly — closing the loop on precision instead of only ever
hunting for missed shapes.

Uses the same matching logic as compare_models.py (conf_threshold=0.0 to
match production, greedy IoU matching against ground truth).

Usage:
    python find_false_positives.py \\
        --frames-dir ./dataset/raw_frames \\
        --labels-dir ./dataset/corrected_labels \\
        --model ./dataset/transparent_shape_nms_v2.onnx \\
        --cvat-host http://192.168.68.92:8080 \\
        --cvat-username admin --cvat-password '...' \\
        --thumbnails-dir ./dataset/false_positive_thumbnails \\
        ld_zSk2iCFC ld_zmahDi0v   # sessions to check; omit for every session with ground truth
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import onnxruntime as ort

from common import run_onnx_detect
from compare_models import load_ground_truth, match_indices


def cvat_frame_url(cvat_host: str, task_id: int, job_id: int, frame_idx: int) -> str:
    return f"{cvat_host}/tasks/{task_id}/jobs/{job_id}?frame={frame_idx}"


def find_session_false_positives(
    session: str, model_path: Path, frames_dir: Path, labels_dir: Path,
    conf_threshold: float, iou_threshold: float, thumbnails_dir: Path | None,
    task_id: int | None, job_id: int | None, cvat_host: str,
) -> list[dict]:
    session_ort = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    results = []

    for frame_idx, img_path in enumerate(sorted(frames_dir.glob("*.png"))):
        img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        h, w = img.shape[:2]
        preds = run_onnx_detect(session_ort, img, conf_threshold=conf_threshold)
        pred_boxes = [(x1, y1, x2, y2) for x1, y1, x2, y2, _conf in preds]
        gt_boxes = load_ground_truth(labels_dir / (img_path.stem + ".txt"), w, h)

        matched_pred, _matched_gt, _ious = match_indices(pred_boxes, gt_boxes, iou_threshold)

        for pi, (x1, y1, x2, y2, conf) in enumerate(preds):
            if pi in matched_pred:
                continue
            entry = dict(session=session, frame=img_path.name, frame_idx=frame_idx,
                         box=(round(x1), round(y1), round(x2), round(y2)), conf=round(conf, 3))
            if task_id is not None and job_id is not None:
                entry["cvat_url"] = cvat_frame_url(cvat_host, task_id, job_id, frame_idx)

            if thumbnails_dir is not None:
                pad = 20
                cx1, cy1 = max(0, int(x1) - pad), max(0, int(y1) - pad)
                cx2, cy2 = min(w, int(x2) + pad), min(h, int(y2) + pad)
                crop = img[cy1:cy2, cx1:cx2].copy()
                cv2.rectangle(crop, (int(x1) - cx1, int(y1) - cy1), (int(x2) - cx1, int(y2) - cy1), (0, 0, 255), 1)
                thumbnails_dir.mkdir(parents=True, exist_ok=True)
                thumb_name = f"{session}_{img_path.stem}_conf{conf:.2f}.png"
                cv2.imwrite(str(thumbnails_dir / thumb_name), crop)
                entry["thumbnail"] = thumb_name

            results.append(entry)

    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("sessions", nargs="+",
                         help="Session names with real human-corrected ground truth. Required and "
                              "not auto-discovered on purpose: --labels-dir contains a folder for "
                              "every session (cvat_export.py re-exports untouched bootstrap boxes "
                              "for sessions you haven't corrected yet), and there's no way to tell "
                              "those apart from real corrections just by looking at the files. "
                              "Passing the wrong ones silently compares against the old model's own "
                              "unverified opinion instead of real ground truth.")
    parser.add_argument("--frames-dir", default=Path("./dataset/raw_frames"), type=Path)
    parser.add_argument("--labels-dir", default=Path("./dataset/corrected_labels"), type=Path)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--conf-threshold", type=float, default=0.0)
    parser.add_argument("--iou-threshold", type=float, default=0.5)
    parser.add_argument("--thumbnails-dir", type=Path, default=Path("./dataset/false_positive_thumbnails"))
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
        results = find_session_false_positives(
            session, args.model, args.frames_dir / session, args.labels_dir / session,
            args.conf_threshold, args.iou_threshold, args.thumbnails_dir, task_id, job_id, cvat_host,
        )
        results.sort(key=lambda r: -r["conf"])
        all_results.extend(results)

        if results:
            print(f"\n[{session}] {len(results)} false positive(s), highest-confidence first:")
            for r in results:
                line = f"  frame={r['frame']} conf={r['conf']} box={r['box']}"
                if "cvat_url" in r:
                    line += f"  {r['cvat_url']}"
                if "thumbnail" in r:
                    line += f"  thumb={r['thumbnail']}"
                print(line)
        else:
            print(f"[{session}] no false positives")

    print(f"\nTotal: {len(all_results)} false positive(s) across {len(sessions)} session(s)")
    if args.thumbnails_dir.exists():
        print(f"Thumbnails saved to {args.thumbnails_dir}")


if __name__ == "__main__":
    main()
