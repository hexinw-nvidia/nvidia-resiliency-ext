# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Slack delivery and durable receipt races, without real Slack or Slurm."""

import io
import json
import sys
import urllib.error
from dataclasses import replace
from pathlib import Path

import pytest

sys.path.insert(
    0, str(Path(__file__).resolve().parents[3] / "examples/fault_tolerance/deployment/watch")
)
from nvrx_watch import events, runner, slack_threads, types
from nvrx_watch.config import Config

from . import test_nvrx_watch_events as event_fixtures

cycle = event_fixtures.cycle
setup = event_fixtures.setup


def enable(config, tmp_path):
    token = tmp_path / "token"
    token.write_text("xoxb-test-secret")
    token.chmod(0o600)
    return replace(config, slack_bot_token_file=str(token), slack_channel_id="C123")


def accepted(config, event):
    return {"event_id": event["event_id"], "channel": "C123", "thread_ts": "123.456"}


def test_ack_before_parent_and_crash_after_receipt(setup, tmp_path, monkeypatch):
    config, snapshot, queue = setup
    config = enable(config, tmp_path)
    ids = events.enqueue(replace(snapshot, cycles=(cycle(0), cycle(1))), config)
    assert events.export(queue, 20)[0]["slack_thread_required"]
    events.acknowledge(queue, ids)
    assert slack_threads.receipts(queue, ids) == {}
    calls = []

    def post(c, e):
        calls.append(e["event_id"])
        return accepted(c, e)

    monkeypatch.setattr(slack_threads, "post_parent", post)
    # Crash after persisting the receipt but before removing the pending intent.
    saved = (queue / "slack-pending" / (ids[0] + ".json")).read_text()
    assert slack_threads.flush(config)
    (queue / "slack-pending" / (ids[0] + ".json")).write_text(saved)
    assert slack_threads.flush(config)
    assert calls == ids
    assert slack_threads.receipts(queue, ids)[ids[0]]["thread_ts"] == "123.456"
    assert slack_threads.cycle_finding_keys(config, ids) == {"nvrx-cycle-restart-100.0.1"}


def test_failed_parent_does_not_starve_other_events(setup, tmp_path, monkeypatch):
    config, snapshot, queue = setup
    config = enable(config, tmp_path)
    ids = events.enqueue(replace(snapshot, cycles=(cycle(0), cycle(1), cycle(2))), config)
    calls = []

    def post(c, e):
        calls.append(e["event_id"])
        if len(calls) == 1:
            raise RuntimeError("Slack unavailable")
        return accepted(c, e)

    monkeypatch.setattr(slack_threads, "post_parent", post)
    assert not slack_threads.flush(config)
    assert slack_threads.flush(config)
    assert len(calls) == 2 and len(set(calls)) == 2
    assert set(slack_threads.receipts(queue, ids)) == {calls[1]}


def test_slack_http_200_error_is_failure_and_token_not_logged(setup, tmp_path, monkeypatch, caplog):
    config, snapshot, queue = setup
    config = enable(config, tmp_path)
    events.enqueue(replace(snapshot, cycles=(cycle(0), cycle(1))), config)

    def rejected(req, **kw):
        assert req.headers["Authorization"] == "Bearer xoxb-test-secret"
        return io.BytesIO(b'{"ok":false,"error":"invalid_auth"}')

    monkeypatch.setattr(slack_threads.urllib.request, "urlopen", rejected)
    assert not slack_threads.flush(config)
    assert "xoxb" not in caplog.text
    assert not (queue / "slack-receipts").exists()


def test_rate_limit_and_dry_run(setup, tmp_path, monkeypatch):
    config, snapshot, queue = setup
    config = enable(config, tmp_path)
    ids = events.enqueue(replace(snapshot, cycles=(cycle(0), cycle(1))), config)

    def limited(*a, **kw):
        raise urllib.error.HTTPError(
            "https://slack.com", 429, "limited", {"Retry-After": "600"}, None
        )

    monkeypatch.setattr(slack_threads.urllib.request, "urlopen", limited)
    monkeypatch.setattr(slack_threads.time, "time", lambda: 1000)
    assert slack_threads.flush(replace(config, dry_run=True))
    assert not slack_threads.flush(config)
    pending = json.loads((queue / "slack-pending" / (ids[0] + ".json")).read_text())
    assert pending["slack_next_attempt"] == 1600
    assert not slack_threads.flush(config)  # Still throttled; no new request.


def test_only_matching_webhook_finding_is_suppressed(tmp_path):
    config = Config(state_dir=str(tmp_path))
    calls = []

    class Sink:
        def __init__(self, name):
            self.name = name

        def emit(self, finding):
            calls.append((self.name, finding.key))
            return True

    finding = types.Finding(key="cycle", detector="cycle_restart", severity="info", summary="test")
    runner.report([finding], config, [Sink("webhook"), Sink("log"), Sink("pagerduty")], {"cycle"})
    assert calls == [("log", "cycle"), ("pagerduty", "cycle")]


def test_config_and_receipt_validation(tmp_path):
    with pytest.raises(ValueError):
        Config(slack_channel_id="C123")
    with pytest.raises(ValueError):
        slack_threads.receipts(tmp_path, ["../secret"])


def test_slack_success_and_private_token_permissions(setup, tmp_path, monkeypatch):
    config, snapshot, queue = setup
    config = enable(config, tmp_path)
    ids = events.enqueue(replace(snapshot, cycles=(cycle(0), cycle(1))), config)
    ev = events.export(queue, 1)[0]

    def success(req, **kw):
        payload = json.loads(req.data)
        assert payload["channel"] == "C123"
        assert ev["event_id"][:12] not in payload["text"]
        return io.BytesIO(b'{"ok":true,"channel":"C123","ts":"123.456"}')

    monkeypatch.setattr(slack_threads.urllib.request, "urlopen", success)
    assert slack_threads.post_parent(config, ev) == accepted(config, ev)
    Path(config.slack_bot_token_file).chmod(0o644)
    with pytest.raises(ValueError):
        slack_threads.post_parent(config, ev)
