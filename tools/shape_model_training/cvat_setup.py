#!/usr/bin/env python3
"""Creates CVAT tasks from the shared dataset directory and pre-loads each
with its bootstrap YOLO labels, so correcting them in CVAT needs no manual
zip/import step at all.

Requires a running local CVAT instance with `dataset/` (this toolkit's
--out-dir root, containing raw_frames/, raw_labels/, corrected_labels/)
mounted read-only into the cvat_server/cvat_worker_* containers at
/home/django/share (see docker-compose.override.yml in the CVAT checkout —
this is what makes "Connected file share" show these folders in the UI).

For each session under --frames-dir, idempotently:
1. Creates a CVAT task named after the session (skips if one already exists),
   sourcing images from the share path `raw_frames/<session>` server-side
   (no browser upload needed).
2. Parses that session's `raw_labels/<session>/*.txt` bootstrap boxes and
   pushes them directly as CVAT shapes via the API (bypassing the YOLO
   zip-import format entirely — this is just plain YOLO cx,cy,w,h converted
   to CVAT's xtl,ytl,xbr,ybr rectangle schema).

Usage:
    python cvat_setup.py --host http://localhost:8080 \\
        --username admin --password '...' \\
        --frames-dir ./dataset/raw_frames \\
        --bootstrap-labels-dir ./dataset/raw_labels \\
        --share-frames-prefix raw_frames \\
        [session ...]        # omit to process every session
"""

from __future__ import annotations

import argparse
from pathlib import Path

from cvat_sdk import make_client
from cvat_sdk.api_client import models
from cvat_sdk.core.proxies.tasks import ResourceType

from common import CLASS_NAME, find_task_by_session


def parse_yolo_labels(
    label_path: Path, img_w: int, img_h: int, label_id: int, frame_idx: int,
) -> list[models.LabeledShapeRequest]:
    if not label_path.exists() or label_path.stat().st_size == 0:
        return []
    shapes = []
    for line in label_path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        _cls, cx, cy, w, h = line.split()
        cx, cy, w, h = float(cx) * img_w, float(cy) * img_h, float(w) * img_w, float(h) * img_h
        xtl, ytl = cx - w / 2, cy - h / 2
        xbr, ybr = cx + w / 2, cy + h / 2
        xtl, ytl = max(0.0, xtl), max(0.0, ytl)
        xbr, ybr = min(float(img_w), xbr), min(float(img_h), ybr)
        if xbr <= xtl or ybr <= ytl:
            continue
        shapes.append(models.LabeledShapeRequest(
            type="rectangle", label_id=label_id, frame=frame_idx,
            points=[xtl, ytl, xbr, ybr], occluded=False, z_order=0,
        ))
    return shapes


def process_session(client, session: str, frame_dir: Path, bootstrap_dir: Path, share_prefix: str) -> None:
    existing = find_task_by_session(client, session)

    if existing is None:
        print(f"[{session}] creating task from share path '{share_prefix}/{session}' ...")
        task = client.tasks.create_from_data(
            spec=models.TaskWriteRequest(
                name=session,
                bug_tracker=session,
                labels=[models.PatchedLabelRequest(name=CLASS_NAME)],
            ),
            resources=[f"{share_prefix}/{session}"],
            resource_type=ResourceType.SHARE,
            data_params={"image_quality": 70, "sorting_method": "lexicographical"},
        )
    else:
        print(f"[{session}] task #{existing.id} already exists, reusing it")
        task = existing

    task.fetch()
    if any(True for _ in task.get_annotations().shapes):
        print(f"[{session}] task #{task.id} already has annotations, skipping label push")
        return

    label_id = next(l.id for l in task.get_labels() if l.name == CLASS_NAME)
    frames = task.get_frames_info()

    shapes = []
    n_missing_label_file = 0
    for frame_idx, frame in enumerate(frames):
        label_path = bootstrap_dir / (Path(frame.name).stem + ".txt")
        if not label_path.exists():
            n_missing_label_file += 1
            continue
        shapes.extend(parse_yolo_labels(label_path, frame.width, frame.height, label_id, frame_idx))

    if n_missing_label_file:
        print(f"[{session}] warning: {n_missing_label_file} frame(s) had no matching bootstrap label file")

    task.set_annotations(
        models.LabeledDataRequest(version=0, tags=[], tracks=[], shapes=shapes)
    )
    print(f"[{session}] pushed {len(shapes)} bootstrap box(es) across {len(frames)} frame(s)")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("sessions", nargs="*", help="Session names to process (default: all under --frames-dir)")
    parser.add_argument("--host", default="http://localhost:8080")
    parser.add_argument("--username", required=True)
    parser.add_argument("--password", required=True)
    parser.add_argument("--frames-dir", default=Path("./dataset/raw_frames"), type=Path)
    parser.add_argument("--bootstrap-labels-dir", default=Path("./dataset/raw_labels"), type=Path)
    parser.add_argument("--share-frames-prefix", default="raw_frames",
                         help="Path CVAT's mounted share sees this frames dir under, e.g. 'raw_frames'")
    args = parser.parse_args()

    all_sessions = sorted(p.name for p in args.frames_dir.iterdir() if p.is_dir())
    sessions = args.sessions or all_sessions
    unknown = set(sessions) - set(all_sessions)
    if unknown:
        raise SystemExit(f"Unknown session(s): {sorted(unknown)}")

    with make_client(host=args.host, credentials=(args.username, args.password)) as client:
        for session in sessions:
            process_session(
                client, session,
                frame_dir=args.frames_dir / session,
                bootstrap_dir=args.bootstrap_labels_dir / session,
                share_prefix=args.share_frames_prefix,
            )


if __name__ == "__main__":
    main()
