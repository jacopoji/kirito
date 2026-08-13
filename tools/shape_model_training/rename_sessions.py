#!/usr/bin/env python3
"""Renames CVAT tasks to short sequential names (session_01, session_02, ...)
for readability in the CVAT UI, preserving the original session name in each
task's `bug_tracker` field so every other script in this toolkit can still
resolve sessions correctly (see `common.find_task_by_session`).

Safe to re-run: tasks that already have a numbered name and a `bug_tracker`
set are left alone, so this only touches tasks that haven't been renamed yet.
New tasks are numbered continuing from the current highest session_NN.

Usage:
    python rename_sessions.py --host http://192.168.68.92:8080 \\
        --username admin --password '...'
"""

from __future__ import annotations

import argparse
import re

from cvat_sdk import make_client
from cvat_sdk.api_client import models


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--host", default="http://localhost:8080")
    parser.add_argument("--username", required=True)
    parser.add_argument("--password", required=True)
    args = parser.parse_args()

    with make_client(host=args.host, credentials=(args.username, args.password)) as client:
        tasks = sorted(client.tasks.list(), key=lambda t: t.id)

        already_numbered = [int(m.group(1)) for t in tasks if (m := re.fullmatch(r"session_(\d+)", t.name))]
        next_number = max(already_numbered, default=0) + 1

        for task in tasks:
            if re.fullmatch(r"session_\d+", task.name) and task.bug_tracker:
                continue  # already renamed

            original_name = task.bug_tracker or task.name
            new_name = f"session_{next_number:02d}"
            next_number += 1

            task.update(models.PatchedTaskWriteRequest(name=new_name, bug_tracker=original_name))
            print(f"#{task.id}: {original_name!r} -> {new_name!r}")


if __name__ == "__main__":
    main()
