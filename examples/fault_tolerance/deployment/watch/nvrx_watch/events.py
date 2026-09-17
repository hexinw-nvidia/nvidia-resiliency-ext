# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Durable, opt-in handoff to a local analysis worker. No scheduler calls here.

``python -m nvrx_watch.events export --queue DIR`` returns a bounded batch.
``... ack --queue DIR ID ...`` archives events after the consumer durably saves them.
Archived records are dedup receipts and must be retained while a chain is monitored.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import tempfile
from collections import Counter
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

from .config import Config
from .types import CAP_PLATFORM, CycleRecord, Snapshot

FAILED_STATES = {"FAILED", "TIMEOUT", "NODE_FAIL", "OUT_OF_MEMORY", "BOOT_FAIL", "DEADLINE"}


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    fd, temporary = tempfile.mkstemp(prefix=".event-", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as stream:
            json.dump(value, stream, default=lambda v: v.isoformat(), sort_keys=True)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def cycle_payload(cycle: CycleRecord | None) -> dict | None:
    if cycle is None:
        return None
    value = asdict(cycle)
    value["key"] = cycle.key
    # Hint only: nonstandard layouts must be resolved from the named cycle-info file.
    value["launcher_log_dir_hint"] = (
        str(Path(cycle.path).parent.parent / "logs") if cycle.path else ""
    )
    return value


def build_event(snapshot, config, kind, previous, current, terminal=None):
    cluster = config.cluster or os.uname().nodename
    identity = [cluster, config.user, config.job_name, config.state_dir, kind]
    identity.append(current.key if current else terminal[0])
    event_id = hashlib.sha256(json.dumps(identity).encode()).hexdigest()
    return {
        "schema_version": 1,
        "slack_thread_required": bool(config.slack_bot_token_file),
        "event_id": event_id,
        "kind": kind,
        "cluster": cluster,
        "watcher_host": os.uname().nodename,
        "observed_at": snapshot.observed_at.isoformat(),
        "chain": {"user": config.user, "job_name": config.job_name, "id": config.state_dir},
        "previous_cycle": cycle_payload(previous),
        "cycle": cycle_payload(current),
        "paths": {
            "cycle_info_glob": config.resolved_cycle_info_glob,
            "checkpoint_iteration_file": config.resolved_checkpoint_file,
        },
        "checkpoint": asdict(snapshot.checkpoint),
        "scheduler": {
            "available": snapshot.has(CAP_PLATFORM),
            "cached": bool(config.cycle_job_ids),
            "generations": [
                {
                    "gen_id": g.gen_id,
                    "task0": asdict(g.task0) if g.task0 else None,
                    "task_counts": dict(Counter(t.state for t in g.tasks)),
                }
                for g in snapshot.generations
            ],
            "terminal": {"job_id": terminal[0], **asdict(terminal[1])} if terminal else None,
        },
    }


def enqueue(snapshot: Snapshot, config: Config) -> list[str]:
    if not config.event_queue_dir or config.dry_run:
        return []
    queue = Path(config.event_queue_dir)
    # Separate activation time per chain: enabling analysis does not replay old runs.
    baseline = Path(config.state_dir) / "analysis-baseline.json"
    if not baseline.exists():
        atomic_json(baseline, {"since": snapshot.observed_at.isoformat()})
        return []
    since = datetime.fromisoformat(json.loads(baseline.read_text())["since"])
    records = sorted(
        (c for c in snapshot.cycles if c.start_time is not None),
        key=lambda c: (c.start_time, c.job_id, c.attempt_index, c.cycle_number),
    )
    prior_index = next(
        (i for i, c in enumerate(records) if c.key == snapshot.prior.latest_cycle_key), None
    )
    payloads = []
    if snapshot.prior.latest_cycle_key:
        for i, cycle in enumerate(records):
            new = (
                i > prior_index
                if prior_index is not None
                else (cycle.start_time > (snapshot.prior.last_pass or since))
            )
            if not new or cycle.start_time < since:
                continue
            # Cycle zero is initial startup, even when an older job/attempt was
            # observed. A shared chain identity is not evidence of failure recovery.
            if cycle.cycle_number == 0:
                continue
            previous = next(
                (
                    c
                    for c in reversed(records[:i])
                    if (c.job_id, c.attempt_index, c.cycle_number)
                    == (cycle.job_id, cycle.attempt_index, cycle.cycle_number - 1)
                ),
                None,
            )
            payloads.append(build_event(snapshot, config, "cycle_restart", previous, cycle))
    # Reconsider cached terminal evidence each pass: accounting can arrive late.
    # Queue/archive filenames deduplicate it, independent of the cycle cursor.
    terminal = dict(snapshot.recent_endings)
    terminal.update(
        {jid: task for (jid, index), task in snapshot.terminal_info.items() if index == 0}
    )
    if snapshot.has(CAP_PLATFORM):
        for jid, task in terminal.items():
            if task.state.upper().split()[0] not in FAILED_STATES:
                continue
            if task.end_time is None or task.end_time < since:
                continue
            failed = next((c for c in reversed(records) if c.job_id == jid), None)
            payloads.append(
                build_event(snapshot, config, "terminal_failure", failed, None, (jid, task))
            )
    saved = []
    for payload in payloads:
        name = payload["event_id"] + ".json"
        if not (queue / "pending" / name).exists() and not (queue / "delivered" / name).exists():
            # Persist the parent notification before exporting the analysis event.
            # A crash here retries safely using the immutable event ID.
            if payload["slack_thread_required"]:
                atomic_json(queue / "slack-pending" / name, payload)
            atomic_json(queue / "pending" / name, payload)
        saved.append(payload["event_id"])
    return saved


def export(queue: Path, limit: int) -> list[dict]:
    if not 1 <= limit <= 100:
        raise ValueError("export limit must be between 1 and 100")
    pending = queue / "pending"
    result = []
    if pending.exists():
        with os.scandir(pending) as entries:
            for entry in entries:
                if not re.fullmatch(r"[0-9a-f]{64}\.json", entry.name):
                    continue
                if entry.is_symlink() or not entry.is_file() or entry.stat().st_size > 2_000_000:
                    raise ValueError("invalid or oversized event")
                result.append(json.loads(Path(entry.path).read_text()))
                if len(result) == limit:
                    break
    return result


def acknowledge(queue: Path, ids: list[str]) -> None:
    if len(ids) > 100 or any(not re.fullmatch(r"[0-9a-f]{64}", i) for i in ids):
        raise ValueError("invalid acknowledgement IDs")
    delivered = queue / "delivered"
    delivered.mkdir(parents=True, exist_ok=True, mode=0o700)
    for event_id in ids:
        source = queue / "pending" / (event_id + ".json")
        target = delivered / source.name
        if source.exists():
            os.replace(source, target)
        elif not target.exists():
            raise ValueError(f"unknown event {event_id}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", choices=("export", "ack", "slack-receipts"))
    parser.add_argument("--queue", required=True, type=Path)
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--ids", nargs="*", default=[])
    args = parser.parse_args()
    if args.operation == "export":
        print(json.dumps(export(args.queue, args.limit)))
    elif args.operation == "slack-receipts":
        from .slack_threads import receipts

        print(json.dumps(receipts(args.queue, args.ids)))
    else:
        acknowledge(args.queue, args.ids)


if __name__ == "__main__":
    main()
