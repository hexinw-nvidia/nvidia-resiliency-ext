# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Read attribution parents and link them into existing watcher threads.

Runs under the watcher's existing pass lock. No Slurm queries or model calls.
One shared, paginated history query and at most one link delivery per pass.
"""

import html
import json
import logging
import re
import time
import uuid
from itertools import islice
from pathlib import Path
from urllib.parse import urlsplit

from .events import atomic_json

LOG = logging.getLogger("nvrx_watch")
RETENTION = 7 * 86400


def _attribution_cycle(event):
    # A restart parent announces the successor, while attribution explains the
    # failed predecessor. Use its recorded identity; do not guess cycle N - 1.
    if event.get("kind") == "cycle_restart":
        return event.get("previous_cycle")
    return event.get("cycle")


def enqueue(config, event):
    if config.dry_run or not config.restart_agent_slack_channel_id:
        return
    if event.get("kind") not in {"cycle_restart", "job_started"} or not event.get("cycle"):
        return
    if not re.fullmatch(r"[0-9a-f]{64}", event["event_id"]):
        raise ValueError("invalid restart link event ID")
    queue = Path(config.event_queue_dir)
    name = event["event_id"] + ".json"
    if (queue / "restart-link-receipts" / name).exists() or (
        queue / "restart-link-pending" / name
    ).exists():
        return
    atomic_json(queue / "restart-link-pending" / name, {"event": event, "created": time.time()})


def _parse(message):
    text = message.get("text", "")
    ts = str(message.get("ts", ""))
    if (
        not re.fullmatch(r"[0-9]+\.[0-9]+", ts)
        or message.get("thread_ts", ts) != ts
        or not text.startswith("*NVRx attribution:*")
    ):
        return None
    job = re.search(r"\*Job ID:\*\s*`(\d+)(?:_\d+)?`", text)
    cycle = re.search(r"\*Cycle:\*\s*(\d+)(?=\s|$)", text)
    owner = re.search(r"\*Owner:\*\s*`([^`]+)`", text)
    log = re.search(r"\*Log path:\*\s*```(.*?)```", text, re.S)
    if not all((job, cycle, owner, log)):
        return None
    return dict(
        ts=ts,
        job=job[1],
        cycle=int(cycle[1]),
        owner=html.unescape(owner[1]),
        log=html.unescape(log[1].strip()),
    )


def _history(config, state, now, api):
    """Keep a small index of parsed parents, with one shared query per pass."""
    if state.get("next_query", 0) > now:
        return
    window = state.setdefault("window", {"oldest": now - 86400, "until": now})
    if not window.get("cursor") and "latest" not in window:
        window["until"] = now
        window["oldest"] = max(window["oldest"], now - RETENTION)
    payload = dict(
        channel=config.restart_agent_slack_channel_id,
        limit=100,
        # Slack timestamps are seconds.microseconds. More fractional digits can
        # silently produce an empty page rather than an invalid-argument error.
        oldest=f"{float(window['oldest']):.6f}",
        latest=f"{float(window.get('latest', window['until'])):.6f}",
    )
    if window.get("cursor"):
        payload["cursor"] = window["cursor"]
    state["next_query"] = now + 180

    def save():
        atomic_json(Path(config.event_queue_dir) / "restart-link-index.json", state)

    save()  # Persist the query budget before contacting Slack.
    try:
        result = api("conversations.history", payload)
    except Exception as exc:
        from .slack_threads import RetryLater

        state["next_query"] = now + (exc.seconds if isinstance(exc, RetryLater) else 600)
        # Slack cursors can expire during an outage. Restart this same time
        # window after the backoff; indexed timestamps make replay harmless.
        window.pop("cursor", None)
        window.pop("latest", None)
        if isinstance(exc, RetryLater):
            state["retry_after"] = now + exc.seconds
        save()
        raise
    messages = state.setdefault("messages", {})
    for raw in result.get("messages", [])[:100]:
        parsed = _parse(raw)
        if parsed:
            messages[parsed["ts"]] = parsed
    if result.get("has_more") or result.get("response_metadata", {}).get("next_cursor"):
        cursor = result.get("response_metadata", {}).get("next_cursor")
        if cursor:
            window["cursor"] = cursor
        elif result.get("messages"):
            window.pop("cursor", None)
            window["latest"] = result["messages"][-1]["ts"]
        else:
            raise ValueError("incomplete Slack history pagination")
    else:
        # Overlap catches arrivals close to a page boundary without duplicates.
        state["window"] = {"oldest": window["until"] - 60, "until": now}
    state["messages"] = {
        ts: messages[ts]
        for ts in sorted(messages, key=float, reverse=True)[:2000]
        if float(ts) >= now - RETENTION
    }
    save()


def _preview(config, source, api):
    """Read the exact current parent rather than inventing a verdict from a link."""
    result = api(
        "conversations.history",
        dict(
            channel=config.restart_agent_slack_channel_id,
            oldest=source["ts"],
            latest=source["ts"],
            inclusive="true",
            limit=1,
        ),
    )
    messages = [m for m in result.get("messages", []) if m.get("ts") == source["ts"]]
    if len(messages) != 1:
        raise ValueError("attribution parent unavailable for preview")
    current = _parse(messages[0])
    fields = ("ts", "job", "cycle", "owner", "log")
    if current is None or any(current[key] != source[key] for key in fields):
        raise ValueError("attribution parent identity changed before preview")
    text = messages[0]["text"]
    verdict = re.match(r"^\*NVRx attribution:\*\s*`(STOP|RESTART|CONTINUE|UNKNOWN|TIMEOUT)`", text)
    if not verdict:
        raise ValueError("attribution verdict unavailable for preview")
    verdict = verdict[1]
    preview = f"*Restart-agent verdict: {verdict}*"
    reason = re.search(r"(?m)^\*Why " + verdict + r":\*[ \t]*(.*)$", text)
    if reason:
        reason = " ".join(html.unescape(reason[1]).replace("`", "").split())
        if len(reason) > 300:
            reason = reason[:299] + "…"
        if reason:
            # Source prose must not turn into another channel/user notification.
            reason = reason.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
            preview += "\n> " + reason
    return verdict, preview


def _render_reply(event, preview, permalink):
    if event.get("kind") == "cycle_restart":
        failed, resumed = _attribution_cycle(event), event.get("cycle")
        if failed and resumed:
            header, separator, reason = preview.partition("\n")
            preview = (
                header + f"\nFailed cycle {failed['cycle_number']} → "
                f"recovery cycle {resumed['cycle_number']}" + separator + reason
            )
    return preview + f"\n<{permalink}|Restart-agent attribution and analysis thread>"


def _link(config, event, parent, source, api):
    """Check manual/lost-response duplicates before posting in the watcher thread."""
    channel, ts = config.restart_agent_slack_channel_id, source["ts"]
    permalink = api("chat.getPermalink", {"channel": channel, "message_ts": ts})["permalink"]
    url = urlsplit(permalink)
    expected = f"/archives/{channel}/p{ts.replace('.', '')}"
    if (
        url.scheme != "https"
        or not (url.hostname or "").endswith(".slack.com")
        or url.path != expected
    ):
        raise ValueError("invalid attribution permalink")
    replies = api(
        "conversations.replies", dict(channel=parent["channel"], ts=parent["thread_ts"], limit=100)
    )
    pattern = r"https://[^/\s<>|]+\.slack\.com" + re.escape(expected) + r"(?=[?\s<>|/#]|$)"
    if any(re.search(pattern, m.get("text", "")) for m in replies.get("messages", [])):
        return dict(source_channel=channel, source_ts=ts, permalink=permalink, existing=True)
    if replies.get("has_more") or replies.get("response_metadata", {}).get("next_cursor"):
        raise ValueError("watcher thread exceeds duplicate-check page bound")
    verdict, preview = _preview(config, source, api)
    text = _render_reply(event, preview, permalink)
    value = api(
        "chat.postMessage",
        {
            "channel": parent["channel"],
            "thread_ts": parent["thread_ts"],
            "text": text,
            "unfurl_links": False,
            "unfurl_media": False,
            "client_msg_id": str(uuid.uuid5(uuid.NAMESPACE_URL, event["event_id"] + ":" + ts)),
        },
    )
    if value.get("channel") != parent["channel"] or not re.fullmatch(
        r"[0-9]+\.[0-9]+", str(value.get("ts", ""))
    ):
        raise ValueError("invalid restart-link receipt")
    return dict(
        source_channel=channel,
        source_ts=ts,
        permalink=permalink,
        reply_ts=value["ts"],
        verdict=verdict,
        text=text,
    )


def flush(config):
    if config.dry_run or not config.restart_agent_slack_channel_id:
        return
    queue = Path(config.event_queue_dir)
    try:
        pending = sorted(islice((queue / "restart-link-pending").glob("*.json"), 10000))
        if not pending:
            return
        now = time.time()
        throttle = queue / "slack-retry-after.json"
        if throttle.exists() and json.loads(throttle.read_text())["until"] > now:
            return
        index_path = queue / "restart-link-index.json"
        index = json.loads(index_path.read_text()) if index_path.exists() else {}
        channel = config.restart_agent_slack_channel_id
        if index.get("channel", channel) != channel:
            raise ValueError("restart-agent source channel changed; use a separate index")
        index["channel"] = channel
        if index.get("retry_after", 0) > now:
            return
        cursor = index.get("pending_cursor", "")
        pending = (
            [p for p in pending if p.name > cursor] + [p for p in pending if p.name <= cursor]
        )[:100]
        index["pending_cursor"] = pending[-1].name
        from .slack_threads import RetryLater, api_call

        def api(method, payload):
            return api_call(config, method, payload, timeout=5)

        _history(config, index, now, api)
        if index.get("window", {}).get("cursor") or "latest" in index.get("window", {}):
            return  # Complete the history window before deciding a match is unique.
        atomic_json(index_path, index)
        states = [(path, json.loads(path.read_text())) for path in pending]
        for path, state in sorted(states, key=lambda item: item[1].get("retry_at", 0)):
            event = state["event"]
            done = queue / "restart-link-receipts" / path.name
            if done.exists():
                path.unlink()
                continue
            if state.get("retry_at", 0) > now:
                continue
            if now - state["created"] > RETENTION:
                atomic_json(done, {"event_id": event["event_id"], "expired": True})
                path.unlink()
                continue
            parent_path = queue / "slack-receipts" / path.name
            if not parent_path.exists():
                continue
            parent = json.loads(parent_path.read_text())
            if (
                parent.get("event_id") != event["event_id"]
                or parent.get("channel") != config.slack_channel_id
                or not re.fullmatch(r"[0-9]+\.[0-9]+", str(parent.get("thread_ts", "")))
            ):
                raise ValueError("invalid watcher parent identity for restart link")
            cycle = _attribution_cycle(event)
            if not cycle:
                continue  # Missing predecessor is not permission to match the successor.
            log = cycle.get("log_file")
            if not log:
                from .cycle_logs import _log_path

                log = _log_path(event, cycle)
            matches = [
                m
                for m in index.get("messages", {}).values()
                if (m["job"], m["cycle"], m["owner"], m["log"])
                == (str(cycle["job_id"]), cycle["cycle_number"], event["chain"]["user"], log)
            ]
            if len(matches) != 1 or not log or not Path(log).is_absolute():
                continue
            state["retry_at"] = now + 180
            atomic_json(path, state)
            try:
                receipt = _link(config, event, parent, matches[0], api)
            except RetryLater as exc:
                state["retry_at"] = now + exc.seconds
                atomic_json(path, state)
                index["retry_after"] = now + exc.seconds
                atomic_json(index_path, index)
                raise
            atomic_json(done, {"event_id": event["event_id"], **receipt})
            path.unlink()
            return
    except Exception as exc:
        LOG.warning("Restart-agent thread link pending (%s)", type(exc).__name__)
