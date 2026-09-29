# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Cycle-log parent updates: real queues/files, mocked Slack, no Slurm calls."""

import json
import sys
from dataclasses import replace
from pathlib import Path

import pytest

sys.path.insert(
    0, str(Path(__file__).resolve().parents[3] / "examples/fault_tolerance/deployment/watch")
)
from nvrx_watch import cycle_logs, discovery_notice, events, slack_threads, types

from . import test_nvrx_watch_events as fixtures

setup = fixtures.setup


@pytest.fixture
def delivery(setup, tmp_path, monkeypatch):
    cfg, snapshot, queue = setup
    cfg = replace(cfg, slack_bot_token_file=str(tmp_path / "token"), slack_channel_id="C123")
    calls = []
    clock = [1000]

    def api(config, method, payload):
        calls.append((method, payload.copy()))
        return {"ok": True, "channel": "C123", "ts": payload.get("ts", f"{len(calls)}.000001")}

    monkeypatch.setattr(slack_threads, "api_call", api)
    monkeypatch.setattr(slack_threads.time, "sleep", lambda seconds: None)
    monkeypatch.setattr(cycle_logs.time, "time", lambda: clock[0])
    return cfg, snapshot, queue, calls, clock


def recorded(tmp_path, number, exists=True, job="100", attempt=0):
    log = tmp_path / "logs" / f"train_{job}_attempt{attempt}_cycle{number}.log"
    log.parent.mkdir(exist_ok=True)
    if exists:
        log.touch()
    return replace(fixtures.cycle(number, job, attempt), log_file=str(log))


def test_restart_posts_current_log_in_parent_only(delivery, tmp_path):
    cfg, snapshot, queue, calls, clock = delivery
    cycles = (recorded(tmp_path, 0), recorded(tmp_path, 1))
    ids = events.enqueue(replace(snapshot, cycles=cycles), cfg)
    assert slack_threads.flush(cfg)
    assert len(calls) == 1
    method, parent = calls[0]
    assert method == "chat.postMessage" and "thread_ts" not in parent
    assert "Previous" not in parent["text"]
    assert "train_100_attempt0_cycle0.log" not in parent["text"]
    assert "\n\n*Cycle 1 log · CMH*\n```\n" in parent["text"]
    assert cycles[1].log_file in parent["text"]
    assert (queue / "slack-log-receipts" / (ids[0] + ".json")).exists()
    clock[0] += 180
    assert slack_threads.flush(cfg)
    assert len(calls) == 1


def test_late_file_updates_same_parent_after_restart(delivery, tmp_path):
    cfg, snapshot, queue, calls, clock = delivery
    current = recorded(tmp_path, 1, exists=False)
    events.enqueue(replace(snapshot, cycles=(recorded(tmp_path, 0), current)), cfg)
    assert slack_threads.flush(cfg)
    assert "Log not available yet." in calls[-1][1]["text"]
    clock[0] += 180
    assert slack_threads.flush(cfg)  # Unchanged placeholder is not reposted.
    assert len(calls) == 1
    Path(current.log_file).touch()
    clock[0] += 180
    assert slack_threads.flush(cfg)
    assert calls[-1][0] == "chat.update"
    assert calls[-1][1]["ts"] == "1.000001"
    assert "not available" not in calls[-1][1]["text"]
    assert len(calls) == 2


def test_initial_notice_waits_for_cycle_info_without_analysis(delivery, tmp_path):
    cfg, snapshot, queue, calls, clock = delivery
    cfg = replace(cfg, cycle_info_glob=str(tmp_path / "nvrx/*/cycle_infos/cycle_info.*"))
    snapshot = replace(
        snapshot,
        cycles=(),
        generations=(types.ChainGeneration("100", (types.TaskInfo(0, "RUNNING"),)),),
    )
    chain = {"user": "alice", "name": "train", "first_id": "100"}
    discovery_notice.notify(cfg, "chain", chain, snapshot, [], lambda: None)
    assert slack_threads.flush(cfg)
    assert "InJob started" in calls[0][1]["text"]
    assert "Diagnosis" not in calls[0][1]["text"]
    assert "Log not available yet." in calls[0][1]["text"]
    assert events.export(queue, 20) == []
    current = recorded(tmp_path, 0)
    info = tmp_path / "nvrx/100/cycle_infos/cycle_info.100.0.0"
    info.parent.mkdir(parents=True)
    info.write_text(
        json.dumps({"job_id": "100", "cycle_number": 0, "cycle_log_file": current.log_file})
    )
    clock[0] += 180
    assert slack_threads.flush(cfg)
    assert calls[-1][0] == "chat.update"
    assert "train_100_attempt0_cycle0.log" in calls[-1][1]["text"]
    discovery_notice.notify(cfg, "chain", chain, snapshot, [], lambda: None)
    assert slack_threads.flush(cfg)
    assert len(calls) == 2


def test_initial_successor_and_legacy_receipts_do_not_duplicate(delivery):
    cfg, snapshot, queue, calls, clock = delivery
    chain = {
        "user": "alice",
        "name": "train",
        "first_id": "100",
        "discovery_notice": {"job_id": "100", "sent": ["webhook"]},
    }
    snapshot = replace(
        snapshot, generations=(types.ChainGeneration("100", (types.TaskInfo(0, "RUNNING"),)),)
    )
    discovery_notice.notify(cfg, "chain", chain, snapshot, [], lambda: None)
    assert slack_threads.flush(cfg)
    assert calls == []
    successor = replace(
        snapshot, generations=(types.ChainGeneration("101", (types.TaskInfo(0, "RUNNING"),)),)
    )
    discovery_notice.notify(cfg, "chain", chain, successor, [], lambda: None)
    assert slack_threads.flush(cfg)
    assert len(calls) == 1 and "array 101" in calls[0][1]["text"]


def test_update_failure_retries_without_reposting_parent(delivery, tmp_path, monkeypatch):
    cfg, snapshot, queue, calls, clock = delivery
    current = recorded(tmp_path, 1, exists=False)
    event_ids = events.enqueue(replace(snapshot, cycles=(recorded(tmp_path, 0), current)), cfg)
    assert slack_threads.flush(cfg)
    Path(current.log_file).touch()
    clock[0] += 180
    original = slack_threads.api_call

    def fail_update(c, method, payload):
        if method == "chat.update":
            raise slack_threads.RetryLater(600)
        return original(c, method, payload)

    monkeypatch.setattr(slack_threads, "api_call", fail_update)
    assert not slack_threads.flush(cfg)
    assert event_ids[0] in slack_threads.receipts(queue, event_ids)
    assert not slack_threads.flush(cfg)
    assert len(calls) == 1
    clock[0] += 601
    monkeypatch.setattr(slack_threads, "api_call", original)
    assert slack_threads.flush(cfg)
    assert len(calls) == 2 and calls[1][1]["ts"] == "1.000001"
    assert "thread_ts" not in calls[1][1]


def test_exact_current_attempt_without_predecessor(delivery, tmp_path):
    cfg, snapshot, queue, calls, clock = delivery
    current = recorded(tmp_path, 1, attempt=2, exists=False)
    other = recorded(tmp_path, 1, attempt=1)
    event = events.build_event(snapshot, cfg, "cycle_restart", None, current)
    event["cycle"]["log_file"] = ""
    event["cycle"]["path"] = ""
    event["paths"]["cycle_info_glob"] = str(tmp_path / "nvrx/*/cycle_infos/cycle_info.*")
    info = tmp_path / "nvrx/100/cycle_infos/cycle_info.100.2.1"
    info.parent.mkdir(parents=True)
    record = {
        "job_id": "100",
        "attempt_index": 1,
        "cycle_number": 1,
        "cycle_log_file": other.log_file,
    }
    info.write_text(json.dumps(record))
    text, complete = cycle_logs.render(event)
    assert not complete and "Cycle 1 log (attempt 2)" in text
    assert Path(other.log_file).name not in text
    Path(current.log_file).touch()
    info.write_text(json.dumps({**record, "attempt_index": 2, "cycle_log_file": current.log_file}))
    text, complete = cycle_logs.render(event)
    assert complete and current.log_file in text and "Previous" not in text


def test_legacy_pending_reply_updates_parent(delivery, tmp_path):
    cfg, snapshot, queue, calls, clock = delivery
    current = recorded(tmp_path, 1)
    event = events.build_event(snapshot, cfg, "cycle_restart", None, current)
    name = event["event_id"] + ".json"
    events.atomic_json(
        queue / "slack-receipts" / name, {"channel": "C123", "thread_ts": "42.000001"}
    )
    events.atomic_json(
        queue / "slack-log-pending" / name, {"event": event, "ts": "43.000001", "text": "old reply"}
    )
    assert slack_threads.flush(cfg)
    assert len(calls) == 1 and calls[0][0] == "chat.update"
    assert calls[0][1]["ts"] == "42.000001"
    assert "thread_ts" not in calls[0][1]


def test_dry_run_has_no_reply_mutation(delivery, tmp_path):
    cfg, snapshot, queue, calls, clock = delivery
    events.enqueue(replace(snapshot, cycles=(recorded(tmp_path, 0), recorded(tmp_path, 1))), cfg)
    assert slack_threads.flush(replace(cfg, dry_run=True))
    assert not calls and not (queue / "slack-log-pending").exists()


def test_receipts_survive_crash_and_analysis_ack(delivery, tmp_path):
    cfg, snapshot, queue, calls, clock = delivery
    ids = events.enqueue(
        replace(snapshot, cycles=(recorded(tmp_path, 0), recorded(tmp_path, 1))), cfg
    )
    events.acknowledge(queue, ids)
    parent_intent = (queue / "slack-pending" / (ids[0] + ".json")).read_text()
    assert slack_threads.flush(cfg)
    # Simulate crashes after the durable receipts but before the intent unlinks.
    (queue / "slack-pending" / (ids[0] + ".json")).write_text(parent_intent)
    reply_receipt = queue / "slack-log-receipts" / (ids[0] + ".json")
    events.atomic_json(
        queue / "slack-log-pending" / reply_receipt.name, json.loads(reply_receipt.read_text())
    )
    clock[0] += 180
    assert slack_threads.flush(cfg)
    assert len(calls) == 1
