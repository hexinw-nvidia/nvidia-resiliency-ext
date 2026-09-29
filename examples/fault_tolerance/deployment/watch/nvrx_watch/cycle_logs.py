# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Current cycle log in each Slack parent. No log content or Slurm reads."""

import glob
import hashlib
import json
import logging
import re
import time
from itertools import islice
from pathlib import Path

from .events import atomic_json, build_event, cycle_payload
from .readers import parse_cycle_file
from .types import CycleRecord

LOG = logging.getLogger("nvrx_watch")


def enqueue_start(config, key, chain, snapshot, notice):
    """Discovery parents use the Slack outbox, never the analysis queue."""
    job = str(notice["job_id"])
    cycle = next((c for c in snapshot.cycles if c.job_id == job and c.cycle_number == 0), None)
    event = build_event(snapshot, config, "job_started", None, cycle or CycleRecord(job, 0, 0))
    event["event_id"] = hashlib.sha256(f"start:{config.cluster}:{key}:{job}".encode()).hexdigest()
    event["observed_at"] = notice["observed_at"]
    event["cycle"] = (
        cycle_payload(cycle)
        if cycle
        else {
            "job_id": job,
            "attempt_index": 0,
            "cycle_number": 0,
        }
    )
    queue = Path(config.event_queue_dir)
    name = event["event_id"] + ".json"
    if (
        not (queue / "slack-receipts" / name).exists()
        and not (queue / "slack-pending" / name).exists()
    ):
        atomic_json(queue / "slack-pending" / name, event)


def enqueue(config, event):
    queue = Path(config.event_queue_dir)
    name = event["event_id"] + ".json"
    if (queue / "slack-log-receipts" / name).exists() or (
        queue / "slack-log-pending" / name
    ).exists():
        return
    parent_file = queue / "slack-receipts" / name
    parent = json.loads(parent_file.read_text()) if parent_file.exists() else {}
    state = {
        "event": event,
        "text": parent.get("text"),
        "ts": parent.get("thread_ts"),
        "target": "parent",
    }
    folder = "slack-log-receipts" if parent.get("log_complete") else "slack-log-pending"
    atomic_json(queue / folder / name, state)


def _log_path(event, cycle):
    if not cycle:
        return None
    identity = (str(cycle["job_id"]), cycle["attempt_index"], cycle["cycle_number"])
    log_file = cycle.get("log_file", "")
    if not log_file:
        paths = [cycle["path"]] if cycle.get("path") else []
        pattern = event.get("paths", {}).get("cycle_info_glob", "")
        if not paths and pattern:
            job, attempt, number = identity
            pattern = pattern.replace("/nvrx/*/", f"/nvrx/{job}/")
            pattern = str(Path(pattern).with_name(f"cycle_info.{job}.{attempt}.{number}"))
            paths = list(islice(glob.iglob(pattern), 10))
        for path in paths:
            record = parse_cycle_file(path)
            if record and (record.job_id, record.attempt_index, record.cycle_number) == identity:
                log_file = record.log_file
                break
    path = Path(log_file)
    # Do not invent a path or expose a filename before the application creates it.
    if not log_file or not path.is_absolute() or not path.is_file():
        return None
    return str(path)


def render(event):
    def escape(value):
        return (
            str(value)
            .replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace("`", "\u02cb")
        )

    cluster = event.get("cluster") or "cluster"
    if cluster.lower() in {"cmh", "aga", "hsg"}:
        cluster = cluster.upper()
    cycle = event.get("cycle") or event.get("previous_cycle")
    label = f"Cycle {cycle['cycle_number']} log" if cycle else "Cycle log"
    if cycle and cycle["attempt_index"]:
        label += f" (attempt {cycle['attempt_index']})"
    heading = f"*{label} · {escape(cluster)}*"
    path = _log_path(event, cycle)
    if not path:
        return heading + "\nLog not available yet.", False
    return heading + f"\n```\n{escape(path)}\n```", True


def flush(config, preferred=None):
    """Update at most one parent; retry missing paths on subsequent cron passes."""
    from .slack_threads import RetryLater, api_call, render_parent

    if config.dry_run or not config.slack_bot_token_file:
        return True
    queue = Path(config.event_queue_dir)
    now = time.time()
    throttle = queue / "slack-retry-after.json"
    try:
        if throttle.exists() and json.loads(throttle.read_text())["until"] > now:
            return False
        paths = list(islice((queue / "slack-log-pending").glob("*.json"), 100))
        paths.sort(key=lambda p: (p.stem != preferred, p.name))
        for path in paths:
            if not re.fullmatch(r"[0-9a-f]{64}\.json", path.name):
                continue
            done = queue / "slack-log-receipts" / path.name
            if done.exists():
                path.unlink()
                continue
            state = json.loads(path.read_text())
            if state.get("retry_at", 0) > now:
                continue
            parent_file = queue / "slack-receipts" / path.name
            if not parent_file.exists():
                continue
            parent = json.loads(parent_file.read_text())
            if parent.get("channel") != config.slack_channel_id or not re.fullmatch(
                r"[0-9]+\.[0-9]+", str(parent.get("thread_ts", ""))
            ):
                raise ValueError("invalid parent for cycle-log update")
            text, complete = render_parent(state["event"])
            state["retry_at"] = now + 180
            atomic_json(path, state)
            if state.get("target") == "parent" and text == state.get("text"):
                continue
            payload = {
                "channel": parent["channel"],
                "text": text,
                "unfurl_links": False,
                "unfurl_media": False,
            }
            payload["ts"] = parent["thread_ts"]
            try:
                value = api_call(config, "chat.update", payload)
            except RetryLater as exc:
                state["retry_at"] = now + exc.seconds
                atomic_json(path, state)
                atomic_json(throttle, {"until": now + exc.seconds})
                return False
            if value.get("channel") != parent["channel"] or not re.fullmatch(
                r"[0-9]+\.[0-9]+", str(value.get("ts", ""))
            ):
                raise ValueError("invalid cycle-log update receipt")
            if value["ts"] != parent["thread_ts"]:
                raise ValueError("cycle-log update changed parent identity")
            state.update(
                ts=value["ts"],
                text=text,
                channel=parent["channel"],
                complete=complete,
                target="parent",
            )
            atomic_json(path, state)
            if complete:
                atomic_json(done, state)
                path.unlink()
            return True
        return True
    except Exception as exc:
        LOG.warning("Cycle-log update pending (%s)", type(exc).__name__)
        return False
