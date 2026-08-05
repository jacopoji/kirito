#!/usr/bin/env python3
"""Flags frames that were never touched during correction — i.e. every box
still matches the bootstrap output exactly — so a re-review pass can focus
on what's actually unverified instead of re-checking an entire session
frame-by-frame from scratch.

Context: some sessions were corrected with a narrower focus (getting the
moving target shape right) rather than checking every shape in the frame
(the static decoys the solver's heuristics also need to see, per
shape.rs). A frame with zero edits from bootstrap is the frame most likely
to have static-shape errors nobody has actually looked at yet — either
missed boxes or, just as importantly, false positives that were never
audited for deletion (which hasn't happened even once across this project).

A frame with *some* edits isn't necessarily fully reviewed either, but it's
at least evidence of engagement; a completely untouched frame has none.

Usage:
    python prioritize_review.py \\
        --cvat-host http://192.168.68.92:8080 \\
        --cvat-username admin --cvat-password '...' \\
        ld_DGOou7Zm ld_DvMXfvh7 ld_FsB8Fm7s ...
"""

from __future__ import annotations

import argparse
from pathlib import Path


def round_line(line: str) -> tuple[str, tuple[float, ...]]:
    parts = line.split()
    return parts[0], tuple(round(float(x), 4) for x in parts[1:])


def cvat_frame_url(cvat_host: str, task_id: int, job_id: int, frame_idx: int) -> str:
    return f"{cvat_host}/tasks/{task_id}/jobs/{job_id}?frame={frame_idx}"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("sessions", nargs="+", help="Sessions to check for unreviewed frames")
    parser.add_argument("--raw-labels-dir", default=Path("./dataset/raw_labels"), type=Path)
    parser.add_argument("--corrected-labels-dir", default=Path("./dataset/corrected_labels"), type=Path)
    parser.add_argument("--cvat-host", default=None)
    parser.add_argument("--cvat-username", default=None)
    parser.add_argument("--cvat-password", default=None)
    args = parser.parse_args()

    task_ids: dict[str, tuple[int, int]] = {}
    if args.cvat_host and args.cvat_username and args.cvat_password:
        from cvat_sdk import make_client
        with make_client(host=args.cvat_host, credentials=(args.cvat_username, args.cvat_password)) as client:
            for t in client.tasks.list():
                session_name = t.bug_tracker if t.bug_tracker in args.sessions else t.name
                if session_name in args.sessions:
                    jobs = t.get_jobs()
                    if jobs:
                        task_ids[session_name] = (t.id, jobs[0].id)

    grand_untouched = grand_total = 0
    for session in args.sessions:
        raw_dir = args.raw_labels_dir / session
        corr_dir = args.corrected_labels_dir / session
        if not raw_dir.is_dir() or not corr_dir.is_dir():
            print(f"[{session}] missing raw or corrected labels dir, skipping")
            continue

        frame_files = sorted(raw_dir.glob("*.txt"))
        untouched = []
        for frame_idx, raw_f in enumerate(frame_files):
            corr_f = corr_dir / raw_f.name
            raw_lines = {round_line(l) for l in raw_f.read_text().splitlines() if l.strip()}
            corr_lines = (
                {round_line(l) for l in corr_f.read_text().splitlines() if l.strip()}
                if corr_f.exists() else None
            )
            if corr_lines == raw_lines:
                untouched.append((frame_idx, raw_f.stem))

        grand_untouched += len(untouched)
        grand_total += len(frame_files)
        pct = 100 * len(untouched) / len(frame_files) if frame_files else 0
        print(f"\n[{session}] {len(untouched)}/{len(frame_files)} frame(s) untouched since bootstrap ({pct:.0f}%)")

        task_id, job_id = task_ids.get(session, (None, None))
        if untouched and task_id is not None:
            # Sample evenly across the untouched set rather than dumping every one —
            # if 80 frames are untouched they're likely all similar, a spread-out
            # sample is enough to judge whether the session needs a full re-pass.
            sample = untouched[::max(1, len(untouched) // 10)][:10]
            for frame_idx, stem in sample:
                print(f"  {stem}  {cvat_frame_url(args.cvat_host, task_id, job_id, frame_idx)}")
        elif untouched:
            print("  (pass --cvat-host/--cvat-username/--cvat-password for direct links)")

    if grand_total:
        print(f"\nOverall: {grand_untouched}/{grand_total} frame(s) untouched "
              f"({100 * grand_untouched / grand_total:.0f}%) across {len(args.sessions)} session(s)")


if __name__ == "__main__":
    main()
