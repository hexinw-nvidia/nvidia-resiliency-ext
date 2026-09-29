# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Cycle-log replies: real queues/files, mocked Slack, no Slurm calls."""

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


def test_restart_parent_then_one_grouped_reply(delivery, tmp_path):
    cfg, snapshot, queue, calls, clock = delivery
    cycles = (recorded(tmp_path, 0), recorded(tmp_path, 1))
    ids = events.enqueue(replace(snapshot, cycles=cycles), cfg)
    assert slack_threads.flush(cfg)
    assert len(calls) == 2
    parent, reply = [p for _, p in calls]
    assert "Directory:" not in parent["text"]
    assert reply["thread_ts"] == "1.000001"
    assert reply["text"].count(str(tmp_path / "logs")) == 1
    assert "Previous cycle 0:" in reply["text"] and "New cycle 1:" in reply["text"]
    assert "train_100_attempt0_cycle0.log" in reply["text"]
    assert "train_100_attempt0_cycle1.log" in reply["text"]
    assert (queue / "slack-log-receipts" / (ids[0] + ".json")).exists()
    clock[0] += 180
    assert slack_threads.flush(cfg)
    assert len(calls) == 2


def test_late_file_updates_same_reply_after_restart(delivery, tmp_path):
    cfg, snapshot, queue, calls, clock = delivery
    current = recorded(tmp_path, 1, exists=False)
    events.enqueue(replace(snapshot, cycles=(recorded(tmp_path, 0), current)), cfg)
    assert slack_threads.flush(cfg)
    assert "New cycle 1: log not available yet." in calls[-1][1]["text"]
    clock[0] += 180
    assert slack_threads.flush(cfg)  # Unchanged placeholder is not reposted.
    assert len(calls) == 2
    Path(current.log_file).touch()
    clock[0] += 180
    assert slack_threads.flush(cfg)
    assert calls[-1][0] == "chat.update"
    assert calls[-1][1]["ts"] == "2.000001"
    assert "not available" not in calls[-1][1]["text"]
    assert len(calls) == 3


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
    assert "Cycle 0: log not available yet." in calls[1][1]["text"]
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
    assert len(calls) == 3


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
    assert len(calls) == 2 and "array 101" in calls[0][1]["text"]


def test_reply_failure_retries_without_reposting_parent(delivery, tmp_path, monkeypatch):
    cfg, snapshot, queue, calls, clock = delivery
    event_ids = events.enqueue(
        replace(snapshot, cycles=(recorded(tmp_path, 0), recorded(tmp_path, 1))), cfg
    )
    original = slack_threads.api_call

    def fail_reply(c, method, payload):
        if "thread_ts" in payload:
            raise slack_threads.RetryLater(600)
        return original(c, method, payload)

    monkeypatch.setattr(slack_threads, "api_call", fail_reply)
    assert not slack_threads.flush(cfg)
    assert event_ids[0] in slack_threads.receipts(queue, event_ids)
    assert not slack_threads.flush(cfg)
    assert len(calls) == 1
    clock[0] += 601
    monkeypatch.setattr(slack_threads, "api_call", original)
    assert slack_threads.flush(cfg)
    assert len(calls) == 2 and calls[1][1]["thread_ts"] == "1.000001"


def test_exact_attempt_missing_predecessor_and_split_directories(delivery, tmp_path):
    cfg, snapshot, queue, calls, clock = delivery
    current = recorded(tmp_path, 1, attempt=2)
    other = recorded(tmp_path, 0, attempt=1)
    event = events.build_event(snapshot, cfg, "cycle_restart", None, current)
    event["paths"]["cycle_info_glob"] = str(tmp_path / "nvrx/*/cycle_infos/cycle_info.*")
    info = tmp_path / "nvrx/100/cycle_infos/cycle_info.100.2.0"
    info.parent.mkdir(parents=True)
    info.write_text(
        json.dumps(
            {
                "job_id": "100",
                "attempt_index": 1,
                "cycle_number": 0,
                "cycle_log_file": other.log_file,
            }
        )
    )
    text, complete = cycle_logs.render(event)
    assert not complete and "Previous cycle 0 (attempt 2): log not available" in text
    assert Path(other.log_file).name not in text
    other_dir = tmp_path / "different"
    other_dir.mkdir()
    previous = recorded(other_dir, 0, attempt=2)
    event["previous_cycle"] = events.cycle_payload(previous)
    text, complete = cycle_logs.render(event)
    assert complete and text.count("Directory:") == 2


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
    (queue / "slack-log-pending" / reply_receipt.name).write_text(reply_receipt.read_text())
    clock[0] += 180
    assert slack_threads.flush(cfg)
    assert len(calls) == 2
