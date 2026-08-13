#!/usr/bin/env python3
"""Pulls corrected annotations back out of CVAT into the plain YOLO `.txt`
tree `03_prepare_dataset.py` expects.

Counterpart to `cvat_setup.py`. For each session, fetches the CVAT task's
current annotations (whatever you've corrected in the browser so far) and
writes one `.txt` per *frame in that task* under
`--corrected-labels-dir/<session>/`, converting CVAT's pixel rectangles back
to normalized YOLO `cx cy w h`. Frames with no shapes get an empty file (a
frame confirmed to have no shapes, not "not yet reviewed" — matching the
missing-file-means-uncorrected convention `03_prepare_dataset.py` uses).

Because CVAT tracks correction state per-task rather than per-frame, this
writes labels for *every* frame in a task, on the assumption you review a
session front-to-back in CVAT before calling this for it — a session you
haven't touched yet just re-exports its bootstrap boxes unchanged, which is
harmless (03_prepare_dataset.py will simply use it as-is until you go back
and correct it for real).

Usage:
    python cvat_export.py --username admin --password '...' \\
        --corrected-labels-dir ./dataset/corrected_labels \\
        [session ...]        # omit to export every task
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

from cvat_sdk import make_client

from common import CLASS_NAME, find_task_by_session


def export_session(client, session: str, out_dir: Path) -> None:
    task = find_task_by_session(client, session)
    if task is None:
        print(f"[{session}] no CVAT task with this name, skipping")
        return

    label_id = next((l.id for l in task.get_labels() if l.name == CLASS_NAME), None)
    frames = task.get_frames_info()
    ann = task.get_annotations()

    by_frame = defaultdict(list)
    n_skipped_other_label = 0
    for shape in ann.shapes:
        if str(shape.type) != "rectangle":
            continue
        if label_id is not None and shape.label_id != label_id:
            n_skipped_other_label += 1
            continue
        by_frame[shape.frame].append(shape)
    if n_skipped_other_label:
        print(f"[{session}] warning: {n_skipped_other_label} shape(s) with an unexpected label_id, skipped")

    out_dir.mkdir(parents=True, exist_ok=True)
    n_boxes = 0
    for frame_idx, frame in enumerate(frames):
        lines = []
        for shape in by_frame.get(frame_idx, []):
            xtl, ytl, xbr, ybr = shape.points
            cx = (xtl + xbr) / 2 / frame.width
            cy = (ytl + ybr) / 2 / frame.height
            w = (xbr - xtl) / frame.width
            h = (ybr - ytl) / frame.height
            lines.append(f"0 {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
        n_boxes += len(lines)

        stem = Path(frame.name).stem
        (out_dir / f"{stem}.txt").write_text("\n".join(lines) + ("\n" if lines else ""))

    print(f"[{session}] wrote {len(frames)} label file(s), {n_boxes} box(es) total, to {out_dir}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("sessions", nargs="*", help="Session names to export (default: every CVAT task)")
    parser.add_argument("--host", default="http://localhost:8080")
    parser.add_argument("--username", required=True)
    parser.add_argument("--password", required=True)
    parser.add_argument("--corrected-labels-dir", default=Path("./dataset/corrected_labels"), type=Path)
    args = parser.parse_args()

    with make_client(host=args.host, credentials=(args.username, args.password)) as client:
        sessions = args.sessions or sorted(t.bug_tracker or t.name for t in client.tasks.list())
        for session in sessions:
            export_session(client, session, args.corrected_labels_dir / session)


if __name__ == "__main__":
    main()
