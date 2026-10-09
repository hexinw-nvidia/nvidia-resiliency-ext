# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Regression tests for checkpoint-silent but actively training cycles."""
import json
import os
import sys
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

WATCH = Path(__file__).resolve().parents[3] / "examples/fault_tolerance/deployment/watch"
sys.path.insert(0, str(WATCH))
from nvrx_watch import detectors, persistence, progress, runner, types  # noqa: E402
from nvrx_watch.config import Config  # noqa: E402
from nvrx_watch.platform import NullPlatform  # noqa: E402

NOW = datetime(2026, 10, 9, 20, 12, 1, tzinfo=timezone.utc)


def line(iteration, at=NOW, skipped=0):
    return (
        f"4159: [{at.replace(tzinfo=None)}] iteration {iteration}/ 601014 | "
        "elapsed time per iteration (ms): 17000.0 | "
        f"number of skipped iterations: {skipped} | number of nan iterations: 0 |\n"
    )


@pytest.fixture
def context(tmp_path):
    path = tmp_path / "cycle0.log"
    cycle = types.CycleRecord(
        job_id="4331635",
        attempt_index=0,
        cycle_number=0,
        start_time=NOW - timedelta(hours=3),
        log_file=str(path),
    )
    cfg = Config(
        platform="none",
        work_dir=str(tmp_path),
        state_dir=str(tmp_path / "state"),
        application_log_timezone="UTC",
    )
    snapshot = types.Snapshot(
        observed_at=NOW,
        capabilities=frozenset((types.CAP_CYCLES, types.CAP_CHECKPOINT)),
        cycles=(cycle,),
        checkpoint=types.CheckpointProgress(
            value=11400, mtime=NOW - timedelta(minutes=61, seconds=11)
        ),
    )
    return path, cfg, snapshot


def observe(path, cfg, snap, text=None):
    if text is not None:
        with path.open("a") as f:
            f.write(text)
    return replace(snap, training=progress.read(snap, cfg))


def next_pass(snap, minutes=3):
    prior = persistence.advance(
        snap.prior, snap.checkpoint, snap.latest_cycle, snap.observed_at, snap.training
    )
    return replace(snap, prior=prior, observed_at=snap.observed_at + timedelta(minutes=minutes))


def test_second_real_alert_during_save_is_suppressed(context):
    path, cfg, snap = context
    snap = observe(
        path,
        cfg,
        snap,
        "0: save_interval ........ 200\n"
        + line(11600, NOW - timedelta(seconds=73))
        + "0: [2026-10-09 20:11:04.444163] saving checkpoint at iteration 11600 to /ckpt\n",
    )
    assert detectors.cycle_stalled(snap, cfg) == []
    assert detectors.checkpoint_overdue(snap, cfg) == []
    assert snap.training.save_interval == 200
    snap = next_pass(snap)
    snap = replace(
        snap, checkpoint=types.CheckpointProgress(value=11600, mtime=NOW + timedelta(seconds=54))
    )
    snap = observe(path, cfg, snap)
    assert snap.training.checkpoint_due_at is None


def test_first_real_alert_startup_plus_200_iterations(context):
    path, cfg, snap = context
    at = datetime(2026, 10, 9, 19, 0, 1, tzinfo=timezone.utc)
    snap = replace(
        snap,
        observed_at=at,
        cycles=(replace(snap.latest_cycle, start_time=at - timedelta(minutes=63)),),
        checkpoint=types.CheckpointProgress(value=11200, mtime=at - timedelta(hours=2)),
    )
    snap = observe(path, cfg, snap, line(11370, at - timedelta(seconds=5)))
    assert detectors.cycle_stalled(snap, cfg) == []


def test_frozen_iteration_and_noisy_logs_cannot_reset_timer(context):
    path, cfg, snap = context
    snap = observe(path, cfg, snap, line(11500))
    snap = next_pass(snap, minutes=61)
    snap = observe(path, cfg, snap, "WARNING retrying forever\n" + line(11500, snap.observed_at))
    assert detectors.cycle_stalled(snap, cfg)[0].severity == types.CRITICAL
    assert snap.training.advanced_at == NOW


def test_skipped_iterations_are_not_progress(context):
    path, cfg, snap = context
    snap = observe(path, cfg, snap, line(11500))
    snap = next_pass(snap, minutes=61)
    snap = observe(path, cfg, snap, line(11510, snap.observed_at, skipped=10))
    assert snap.training.advanced_at == NOW
    assert detectors.cycle_stalled(snap, cfg)[0].severity == types.CRITICAL
    snap = next_pass(snap)
    snap = observe(path, cfg, snap, line(11511, snap.observed_at, skipped=10))
    assert detectors.cycle_stalled(snap, cfg) == []


@pytest.mark.parametrize("kind", ["absent", "unrecognized", "startup", "fifo"])
def test_unavailable_progress_only_warns(context, kind):
    path, cfg, snap = context
    if kind == "unrecognized":
        path.write_text("INFO doing something\n")
    elif kind == "startup":
        path.write_text("0: save_interval ........ 200\n")
    elif kind == "fifo":
        os.mkfifo(path)
    snap = observe(path, cfg, snap)
    findings = detectors.cycle_stalled(snap, cfg)
    assert len(findings) == 1 and findings[0].severity == types.WARNING
    assert "unconfirmed" in findings[0].summary


def test_cycle_rollback_resets_highwater(context):
    path, cfg, snap = context
    snap = observe(path, cfg, snap, line(11650))
    snap = next_pass(snap)
    other = path.with_name("cycle1.log")
    other.write_text(line(11601, snap.observed_at))
    snap = replace(snap, cycles=(replace(snap.latest_cycle, cycle_number=1, log_file=str(other)),))
    snap = observe(other, cfg, snap)
    assert snap.training.iteration == 11601
    assert snap.training.advanced_at == snap.observed_at


@pytest.mark.parametrize("rotation", [False, True])
def test_truncation_or_rotation_cannot_rejuvenate_old_iteration(context, rotation):
    path, cfg, snap = context
    snap = observe(path, cfg, snap, line(11500) + "padding " * 100 + "\n")
    snap = next_pass(snap, minutes=61)
    if rotation:
        path.rename(path.with_suffix(".old"))
    path.write_text(line(11500, snap.observed_at))
    snap = observe(path, cfg, snap)
    assert snap.training.advanced_at == NOW
    assert detectors.cycle_stalled(snap, cfg)[0].severity == types.CRITICAL


def test_partial_line_and_state_roundtrip(context):
    path, cfg, snap = context
    text = line(11500)
    path.write_text(text[:-20])
    snap = observe(path, cfg, snap)
    assert not snap.training.available
    snap = next_pass(snap)
    persistence.save(cfg.state_file, snap.prior, {})
    prior, _ = persistence.load(cfg.state_file)
    snap = replace(snap, prior=prior)
    snap = observe(path, cfg, snap, text[-20:])
    assert snap.training.iteration == 11500
    assert snap.training.advanced_at == NOW


def test_bounded_bootstrap_and_backlog_unknown(context):
    path, cfg, snap = context
    cfg.progress_read_bytes = cfg.progress_header_bytes = 1024
    path.write_text("0: save_interval ........ 200\n" + "noise\n" * 1000 + line(11500))
    snap = observe(path, cfg, snap)
    assert snap.training.available and snap.training.save_interval == 200
    snap = next_pass(snap, minutes=61)
    snap = observe(path, cfg, snap, "unrecognized\n" * 1000)
    assert not snap.training.available
    assert detectors.cycle_stalled(snap, cfg)[0].severity == types.WARNING


def test_checkpoint_overdue_independent_of_training(context):
    path, cfg, snap = context
    snap = observe(path, cfg, snap, "0: save_interval ........ 200\n" + line(11600))
    snap = next_pass(snap, minutes=11)
    snap = observe(path, cfg, snap, line(11640, snap.observed_at))
    assert detectors.cycle_stalled(snap, cfg) == []
    found = detectors.checkpoint_overdue(snap, cfg)
    assert len(found) == 1 and found[0].severity == types.WARNING
    assert "11600" in found[0].summary
    snap = next_pass(snap)
    snap = observe(
        path,
        cfg,
        snap,
        f"0: [{snap.observed_at.replace(tzinfo=None)}] saving checkpoint at iteration 11600 to /ckpt\n",
    )
    assert snap.training.checkpoint_due_at == NOW
    assert detectors.checkpoint_overdue(snap, cfg)


def test_unchanged_checkpoint_touch_cannot_reset_stall(context):
    path, cfg, snap = context
    snap = observe(path, cfg, snap, line(11500))
    snap = next_pass(snap, minutes=61)
    snap = replace(snap, checkpoint=replace(snap.checkpoint, mtime=snap.observed_at))
    snap = observe(path, cfg, snap)
    assert detectors.cycle_stalled(snap, cfg)[0].severity == types.CRITICAL


def test_pdt_timestamps_are_normalized(context):
    path, cfg, snap = context
    cfg.application_log_timezone = "America/Los_Angeles"
    path.write_text(line(11600, NOW - timedelta(hours=7, seconds=73)))
    snap = observe(path, cfg, snap)
    assert snap.training.advanced_at == NOW - timedelta(seconds=73)
    assert detectors.cycle_stalled(snap, cfg) == []


def test_run_once_persists_reader_state_and_dry_run_does_not(context, monkeypatch):
    path, cfg, snap = context
    path.write_text(line(11600))
    cycle_file = path.with_name("cycle_info.4331635.0.0")
    cycle_file.write_text(
        json.dumps(
            {
                "job_id": "4331635",
                "cycle_start_time": snap.latest_cycle.start_time.isoformat(),
                "cycle_log_file": str(path),
            }
        )
    )
    cfg.cycle_info_glob = str(cycle_file)
    monkeypatch.setattr(runner, "utcnow", lambda: NOW)
    monkeypatch.setattr(persistence, "utcnow", lambda: NOW)
    result = runner.run_once(cfg, NullPlatform(), sink_list=[])
    assert result.snapshot.training.iteration == 11600
    prior, _ = persistence.load(cfg.state_file)
    assert prior.training.iteration == 11600
    before = Path(cfg.state_file).read_bytes()
    cfg.dry_run = True
    runner.run_once(cfg, NullPlatform(), sink_list=[])
    assert Path(cfg.state_file).read_bytes() == before


def test_checkpoint_deadline_uses_first_crossing_in_read_batch(context):
    path, cfg, snap = context
    snap = observe(
        path,
        cfg,
        snap,
        "0: save_interval ........ 200\n"
        + line(11600, NOW - timedelta(minutes=12))
        + line(11640, NOW),
    )
    assert snap.training.checkpoint_due_at == NOW - timedelta(minutes=12)
    assert detectors.checkpoint_overdue(snap, cfg)


def test_measured_save_duration_extends_checkpoint_grace(context):
    path, cfg, snap = context
    snap = observe(
        path,
        cfg,
        snap,
        "0: save_interval ........ 200\n"
        "0: [2026-10-09 19:00:00] saving checkpoint at iteration 11400 to /ckpt\n"
        "0: [2026-10-09 19:05:00] successfully saved checkpoint from iteration 11400 to /ckpt\n"
        + line(11600, NOW - timedelta(minutes=11)),
    )
    assert snap.training.save_seconds == 300
    assert detectors.checkpoint_overdue(snap, cfg) == []
    snap = next_pass(snap, minutes=5)
    snap = observe(path, cfg, snap, line(11640, snap.observed_at))
    assert detectors.checkpoint_overdue(snap, cfg)


def test_stale_or_future_iteration_does_not_suppress_warning(context):
    path, cfg, snap = context
    snap = observe(
        path,
        cfg,
        snap,
        line(11600, NOW - timedelta(hours=4)) + line(11610, NOW + timedelta(hours=7)),
    )
    assert not snap.training.available
    assert detectors.cycle_stalled(snap, cfg)[0].severity == types.WARNING


def test_read_volume_is_bounded(context, monkeypatch):
    path, cfg, snap = context
    cfg.progress_read_bytes = 1024
    cfg.progress_header_bytes = 2048
    path.write_text("noise\n" * 10000 + line(11600))
    sizes = []
    original = progress._lines

    def track(fh, start, budget, end):
        before = fh.tell()
        result = original(fh, start, budget, end)
        sizes.append(fh.tell() - start)
        assert fh.tell() >= before or start < before
        return result

    monkeypatch.setattr(progress, "_lines", track)
    snap = observe(path, cfg, snap)
    assert sum(sizes) <= 3072
    assert snap.training.available


def test_dead_generation_never_reports_checkpoint_overdue(context):
    path, cfg, snap = context
    snap = observe(
        path,
        cfg,
        snap,
        "0: save_interval ........ 200\n" + line(11600, NOW - timedelta(minutes=20)),
    )
    snap = replace(snap, capabilities=snap.capabilities | {types.CAP_PLATFORM})
    assert detectors.checkpoint_overdue(snap, cfg) == []
