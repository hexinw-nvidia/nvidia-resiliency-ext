# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Durable Slack parent messages; no scheduler queries or model calls."""

import json
import logging
import os
import re
import stat
import time
import urllib.error
import urllib.request
from pathlib import Path

from .events import atomic_json

LOG = logging.getLogger("nvrx_watch")


def receipts(queue, ids):
    if len(ids) > 100 or any(not re.fullmatch(r"[0-9a-f]{64}", i) for i in ids):
        raise ValueError("invalid receipt IDs")
    result = {}
    for event_id in ids:
        path = queue / "slack-receipts" / (event_id + ".json")
        if path.exists():
            result[event_id] = json.loads(path.read_text())
    return result


def cycle_finding_keys(config, ids):
    queue = Path(config.event_queue_dir)
    keys = set()
    for event_id in ids:
        for folder in ("pending", "delivered"):
            path = queue / folder / (event_id + ".json")
            try:
                event = json.loads(path.read_text())
            except FileNotFoundError:
                continue  # The local consumer may concurrently acknowledge it.
            if event.get("slack_thread_required") and event.get("cycle"):
                keys.add("nvrx-cycle-restart-" + event["cycle"]["key"])
            break
    return keys


class RetryLater(Exception):
    def __init__(self, seconds):
        self.seconds = max(60, min(float(seconds), 86400))


def post_parent(config, event):
    path = Path(config.slack_bot_token_file)
    if stat.S_IMODE(path.stat().st_mode) not in (0o400, 0o600):
        raise ValueError("Slack token file must have mode 400 or 600")
    token = path.read_text().strip()
    if not token.startswith("xoxb-") or any(c.isspace() for c in token):
        raise ValueError("expected a Slack bot token")
    cycle = event.get("cycle") or {}
    terminal = event.get("scheduler", {}).get("terminal") or {}
    job = cycle.get("job_id") or terminal.get("job_id", "unknown")
    description = {
        "cycle_restart": "NVRx restart cycle started",
        "generation_transition": "NVRx job/attempt transition observed",
        "terminal_failure": "NVRx terminal failure observed",
    }[event["kind"]]
    text = (
        f"[{event['observed_at']}] [{event['chain']['user']}/{event['chain']['job_name']}] "
        f"{description} for Slurm job array {job}"
    )
    if cycle:
        text += f", attempt {cycle['attempt_index']}, cycle {cycle['cycle_number']}"
    if terminal:
        text += f": {terminal['state']}"
    text += ".\nDiagnosis and validation will follow in this thread."
    request = urllib.request.Request(
        "https://slack.com/api/chat.postMessage",
        data=json.dumps(
            {
                "channel": config.slack_channel_id,
                "text": text,
                "unfurl_links": False,
                "unfurl_media": False,
            }
        ).encode(),
        headers={"Authorization": "Bearer " + token, "Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=15) as response:
            value = json.load(response)
    except urllib.error.HTTPError as exc:
        if exc.code == 429:
            raise RetryLater(exc.headers.get("Retry-After", "180")) from None
        raise RuntimeError("Slack parent delivery failed") from None
    # Slack can return HTTP 200 with ok=false. Never log the request or token.
    if not value.get("ok"):
        raise RuntimeError("Slack parent rejected")
    if value.get("channel") != config.slack_channel_id or not re.fullmatch(
        r"[0-9]+\.[0-9]+", str(value.get("ts", ""))
    ):
        raise ValueError("invalid Slack parent receipt")
    return {"event_id": event["event_id"], "channel": value["channel"], "thread_ts": value["ts"]}


def flush(config):
    """Try one parent per watcher pass, keeping failed delivery independent of analysis.

    One request bounds login-node latency and respects Slack's channel rate limit.
    Receipts survive event acknowledgement and chain retirement. A lost HTTP reply
    can still duplicate a parent; Slack posting is not an exactly-once transaction.
    """
    if not config.slack_bot_token_file or config.dry_run:
        return True
    queue = Path(config.event_queue_dir)
    pending = queue / "slack-pending"
    if not pending.exists():
        return True
    try:
        now = time.time()
        throttle = queue / "slack-retry-after.json"
        if throttle.exists() and json.loads(throttle.read_text())["until"] > now:
            return False
        candidates = []
        with os.scandir(pending) as entries:
            for index, entry in enumerate(entries):
                if index >= 100:
                    break
                if not re.fullmatch(r"[0-9a-f]{64}\.json", entry.name):
                    continue
                path = Path(entry.path)
                receipt = queue / "slack-receipts" / entry.name
                if receipt.exists():
                    path.unlink()
                    continue
                event = json.loads(path.read_text())
                candidates.append(
                    (event.get("slack_next_attempt", 0), event["observed_at"], path, event)
                )
        # Prefer unattempted/oldest due events so one rejected message cannot block others.
        for retry_at, _, path, event in sorted(candidates):
            if retry_at > now:
                continue
            receipt = queue / "slack-receipts" / path.name
            event["slack_next_attempt"] = now + 180
            atomic_json(path, event)
            try:
                value = post_parent(config, event)
            except RetryLater as exc:
                event["slack_next_attempt"] = now + exc.seconds
                atomic_json(throttle, {"until": now + exc.seconds})
                atomic_json(path, event)
                return False
            atomic_json(receipt, value)
            path.unlink()
            return True
        return True
    except Exception as exc:
        LOG.warning("Slack parent delivery pending (%s)", type(exc).__name__)
        return False
