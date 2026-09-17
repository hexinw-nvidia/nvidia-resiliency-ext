# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Durable event handoff: real files, no scheduler or network."""

import sys
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

WATCH_DIR = Path(__file__).resolve().parents[3] / "examples/fault_tolerance/deployment/watch"
sys.path.insert(0, str(WATCH_DIR))

from nvrx_watch import discovery, events, persistence, runner, sinks, types  # noqa: E402
from nvrx_watch.config import Config  # noqa: E402
from nvrx_watch.platform import NullPlatform  # noqa: E402

NOW = datetime(2026, 9, 16, tzinfo=timezone.utc)


def cycle(n, job="100", attempt=0):
    return types.CycleRecord(
        job,
        attempt,
        n,
        NOW + timedelta(minutes=n),
        log_file=f"/run/{job}/cycle{n}.log",
        path=f"/run/{job}/cycle_infos/{n}",
    )


@pytest.fixture
def setup(tmp_path):
    config = Config(
        state_dir=str(tmp_path / "state"),
        event_queue_dir=str(tmp_path / "outbox"),
        cluster="cmh",
        user="alice",
        job_name="train",
    )
    snapshot = types.Snapshot(
        observed_at=NOW,
        cycles=(cycle(0),),
        capabilities=frozenset({types.CAP_PLATFORM, types.CAP_CYCLES}),
    )
    assert events.enqueue(snapshot, config) == []
    snapshot = replace(
        snapshot, prior=types.PriorState(latest_cycle_key=cycle(0).key, last_pass=NOW)
    )
    return config, snapshot, Path(config.event_queue_dir)


def test_all_intervening_cycles_and_archived_dedup(setup):
    config, snapshot, queue = setup
    snapshot = replace(snapshot, cycles=(cycle(0), cycle(1), cycle(2)))
    ids = events.enqueue(snapshot, config)
    assert len(ids) == 2
    records = sorted(events.export(queue, 20), key=lambda e: e["cycle"]["cycle_number"])
    assert [(e["previous_cycle"]["key"], e["cycle"]["key"]) for e in records] == [
        ("100.0.0", "100.0.1"),
        ("100.0.1", "100.0.2"),
    ]
    assert records[0]["chain"]["user"] == "alice"
    events.acknowledge(queue, ids)
    events.acknowledge(queue, ids)  # retry an acknowledgement after lost SSH reply
    events.enqueue(snapshot, config)
    assert events.export(queue, 20) == []
    assert len(list((queue / "delivered").glob("*.json"))) == 2


def test_first_observation_no_historical_replay(tmp_path):
    config = Config(state_dir=str(tmp_path), event_queue_dir=str(tmp_path / "queue"))
    snapshot = types.Snapshot(
        observed_at=NOW + timedelta(hours=1),
        cycles=(cycle(0), cycle(1)),
        prior=types.PriorState(latest_cycle_key=cycle(0).key, last_pass=NOW),
    )
    events.enqueue(snapshot, config)
    events.enqueue(snapshot, config)
    assert events.export(Path(config.event_queue_dir), 20) == []


def test_new_array_and_attempt_cycle_zero_do_not_trigger_analysis(setup):
    config, snapshot, queue = setup
    successor = replace(cycle(0, "101"), start_time=NOW + timedelta(minutes=1))
    events.enqueue(replace(snapshot, cycles=(cycle(0), successor)), config)
    assert events.export(queue, 1) == []
    events.enqueue(
        replace(snapshot, cycles=(cycle(0), replace(successor, job_id="100", attempt_index=1))),
        config,
    )
    assert events.export(queue, 20) == []


def test_restart_pairs_only_same_job_attempt_predecessor(setup):
    config, snapshot, queue = setup
    other = replace(cycle(3), start_time=NOW + timedelta(seconds=70))
    initial = replace(cycle(0, "101"), start_time=NOW + timedelta(seconds=60))
    restart = replace(cycle(1, "101"), start_time=NOW + timedelta(seconds=90))
    events.enqueue(replace(snapshot, cycles=(cycle(0), initial, other, restart)), config)
    record = next(e for e in events.export(queue, 20) if e["cycle"]["job_id"] == "101")
    assert record["kind"] == "cycle_restart"
    assert record["previous_cycle"]["key"] == "101.0.0"


def test_missing_predecessor_never_uses_another_job(setup):
    config, snapshot, queue = setup
    restart = replace(cycle(1, "101"), start_time=NOW + timedelta(minutes=2))
    events.enqueue(replace(snapshot, cycles=(cycle(0), restart)), config)
    assert events.export(queue, 1)[0]["previous_cycle"] is None


def test_terminal_without_successor_and_delayed_accounting(setup):
    config, snapshot, queue = setup
    task = types.TaskInfo(0, "NODE_FAIL", 1, NOW + timedelta(minutes=1))
    # Accounting evidence arrives after the observation cursor passed the end time.
    snapshot = replace(
        snapshot,
        observed_at=NOW + timedelta(minutes=20),
        prior=replace(snapshot.prior, last_pass=NOW + timedelta(minutes=10)),
        recent_endings=(("100", task),),
    )
    events.enqueue(snapshot, config)
    event = events.export(queue, 20)[0]
    assert event["kind"] == "terminal_failure" and event["cycle"] is None
    assert event["previous_cycle"]["key"] == "100.0.0"
    assert event["scheduler"]["terminal"]["state"] == "NODE_FAIL"
    events.enqueue(snapshot, config)
    assert len(events.export(queue, 20)) == 1


@pytest.mark.parametrize("state", ["COMPLETED", "CANCELLED", "RUNNING"])
def test_not_a_terminal_failure(setup, state):
    config, snapshot, queue = setup
    task = types.TaskInfo(0, state, 0, NOW + timedelta(minutes=1))
    events.enqueue(replace(snapshot, recent_endings=(("100", task),)), config)
    assert events.export(queue, 20) == []


def test_outbox_failure_preserves_cursor_and_still_reports(setup, monkeypatch):
    config, snapshot, queue = setup
    snapshot = replace(snapshot, cycles=(cycle(0), cycle(1)))
    persistence.save(config.state_file, snapshot.prior, {})
    monkeypatch.setattr(runner, "gather", lambda *a: (snapshot, []))
    monkeypatch.setattr(events, "atomic_json", lambda *a: (_ for _ in ()).throw(OSError("full")))
    calls = []
    monkeypatch.setattr(runner, "report", lambda *a: calls.append(True))
    result = runner.run_once(config, NullPlatform(), [])
    assert result.degraded and calls == [True]
    assert persistence.load(config.state_file)[0].latest_cycle_key == "100.0.0"


def test_dry_run_does_not_create_outbox_or_baseline(tmp_path):
    config = Config(
        state_dir=str(tmp_path / "state"), event_queue_dir=str(tmp_path / "queue"), dry_run=True
    )
    events.enqueue(types.Snapshot(), config)
    assert not list(tmp_path.iterdir())


def test_export_limit_and_ack_path_validation(setup):
    config, snapshot, queue = setup
    events.enqueue(replace(snapshot, cycles=tuple(cycle(i) for i in range(4))), config)
    assert len(events.export(queue, 1)) == 1
    with pytest.raises(ValueError):
        events.acknowledge(queue, ["../outside"])
    with pytest.raises(ValueError):
        events.export(queue, 101)
    assert all(p.stat().st_mode & 0o777 == 0o600 for p in (queue / "pending").glob("*.json"))


def test_analysis_independent_of_slack_opt_in(setup, monkeypatch):
    config, snapshot, queue = setup
    assert not config.notify_cycle_restarts
    snapshot = replace(snapshot, cycles=(cycle(0), cycle(1)))
    monkeypatch.setattr(runner, "gather", lambda *a: (snapshot, []))
    runner.run_once(config, NullPlatform(), [])
    assert len(events.export(queue, 20)) == 1


def test_retirement_waits_for_delayed_terminal_event(tmp_path, monkeypatch):
    cfg = Config(
        discover_users=("alice",),
        state_dir=str(tmp_path / "state"),
        event_queue_dir=str(tmp_path / "queue"),
        discovery_retire_seconds=600,
    )
    now = NOW.timestamp()
    registry = {
        "users": {
            "alice": {
                "next_query": now + 600,
                "observed": now,
                "arrays": {},
                "terminal": {
                    "100": {
                        "state": "FAILED",
                        "code": "1",
                        "end": (NOW - timedelta(hours=12)).isoformat(),
                    }
                },
            }
        },
        "chains": {
            "chain": {
                "user": "alice",
                "name": "train",
                "ids": ["100"],
                "first_id": "100",
                "absent_since": now - 1200,
                "config": {
                    "cycle_info_glob": str(tmp_path / "missing/*"),
                    "checkpoint_iteration_file": "",
                },
            }
        },
    }
    baseline = Path(cfg.state_dir) / "chains/chain-100/analysis-baseline.json"
    events.atomic_json(baseline, {"since": (NOW - timedelta(days=1)).isoformat()})
    monkeypatch.setattr(sinks, "build", lambda cfg: [])
    monkeypatch.setattr(types, "utcnow", lambda: NOW)
    monkeypatch.setattr(runner, "utcnow", lambda: NOW)
    original = events.atomic_json
    monkeypatch.setattr(events, "atomic_json", lambda *a: (_ for _ in ()).throw(OSError("full")))
    assert discovery._pass(cfg, registry, now, lambda: None) == 1
    assert registry["chains"]  # not retired before its terminal event is durable
    monkeypatch.setattr(events, "atomic_json", original)
    discovery._pass(cfg, registry, now, lambda: None)
    assert not registry["chains"]
    pending = events.export(Path(cfg.event_queue_dir), 20)
    assert len(pending) == 1 and pending[0]["kind"] == "terminal_failure"
