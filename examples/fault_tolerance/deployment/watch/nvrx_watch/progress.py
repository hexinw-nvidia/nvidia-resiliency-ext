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

"""Bounded, incremental Megatron progress observation; never use log mtime as progress.

Application timestamps are local to the training cluster by default. A watcher running
elsewhere must set application_log_timezone. Reader state is scoped to a cycle, so a
checkpoint rollback in a new cycle does not inherit the previous iteration high-water.
"""

from __future__ import annotations

import os
import re
import stat
from dataclasses import replace
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

from .config import Config
from .types import Snapshot, TrainingProgress

STAMP = re.compile(r"\[(\d{4}-\d\d-\d\d [\d:.]+)\]")
ITERATION = re.compile(r"\biteration\s+(\d+)\s*/\s*\d+\s*\|")
SKIPPED = re.compile(r"number of skipped iterations:\s*(\d+)")
STEP = re.compile(r"elapsed time per iteration \(ms\):\s*([\d.]+)")
INTERVAL = re.compile(r"\bsave_interval\s+\.+\s+(\d+)\s*$")
LOADED = re.compile(r"successfully loaded checkpoint.*at iteration\s+(\d+)")
SAVE_START = re.compile(r"(?<!successfully )saving checkpoint at iteration\s+(\d+)")
SAVE_END = re.compile(r"successfully saved checkpoint from iteration\s+(\d+)")


def _timestamp(line: str, config: Config, snapshot: Snapshot) -> datetime | None:
    match = STAMP.search(line)
    if not match:
        return None
    try:
        value = datetime.fromisoformat(match[1])
        if config.application_log_timezone:
            value = value.replace(tzinfo=ZoneInfo(config.application_log_timezone))
        value = value.astimezone(timezone.utc)
    except ValueError:
        return None
    cycle = snapshot.latest_cycle
    if value > snapshot.observed_at or (cycle.start_time and value < cycle.start_time):
        return None
    return value


def _lines(fh, start: int, budget: int, end: int) -> tuple[list[str], int]:
    """Read at most budget bytes. Revisit a partial final line on the next pass."""
    fh.seek(start)
    raw = fh.read(min(budget, end - start))
    complete = raw.rfind(b"\n") + 1
    lines = raw[:complete].decode("utf-8", errors="replace").splitlines()
    if start:
        # The start may be inside a line after bootstrap/backlog skipping. The caller
        # supplies start-1, so discarding this line also preserves aligned records.
        lines = lines[1:]
    return lines, start + complete


def read(snapshot: Snapshot, config: Config) -> TrainingProgress:
    cycle = snapshot.latest_cycle
    if cycle is None or not cycle.is_open:
        return TrainingProgress()
    old = snapshot.prior.training
    current = old if old.cycle_key == cycle.key else TrainingProgress(cycle_key=cycle.key)
    current = replace(current, path=cycle.log_file, available=False)
    if not cycle.log_file:
        return replace(current, reason="cycle has no application log path")
    try:
        # O_NONBLOCK avoids hanging the watcher on a FIFO masquerading as a log.
        fd = os.open(cycle.log_file, os.O_RDONLY | os.O_NONBLOCK)
        with os.fdopen(fd, "rb") as fh:
            info = os.fstat(fh.fileno())
            if not stat.S_ISREG(info.st_mode):
                return replace(current, reason="application log is not a regular file")
            identity = f"{info.st_dev}:{info.st_ino}"
            reset = current.identity != identity or info.st_size < current.offset
            lines = []
            if reset:
                header, _ = _lines(fh, 0, config.progress_header_bytes, info.st_size)
                # Bootstrap the argument/checkpoint metadata without interpreting
                # historical iterations in the header as fresh progress.
                lines.extend(
                    line for line in header if INTERVAL.search(line) or LOADED.search(line)
                )
            start = 0 if reset else current.offset
            skipped_bytes = info.st_size - start > config.progress_read_bytes
            if skipped_bytes:
                start = info.st_size - config.progress_read_bytes + 1
            # One byte of overlap tells _lines where the next full line begins.
            block, offset = _lines(fh, max(0, start - 1), config.progress_read_bytes, info.st_size)
            lines.extend(block)
    except OSError as exc:
        return replace(current, reason=f"application log unreadable: {exc.strerror}")

    current = replace(current, identity=identity, offset=offset)
    found_iteration = False
    advancements = []
    for line in lines:
        match = INTERVAL.search(line)
        if match:
            current = replace(current, save_interval=int(match[1]) or None)
        match = LOADED.search(line)
        if match:
            current = replace(current, loaded_iteration=int(match[1]))
        timestamp = _timestamp(line, config, snapshot)
        if timestamp is None:
            continue
        match = ITERATION.search(line)
        skipped = SKIPPED.search(line)
        if match and skipped:
            found_iteration = True
            iteration = int(match[1])
            completed = iteration - int(skipped[1])
            if current.completed is None or completed > current.completed:
                # Timestamps must also advance: reordered or replayed log messages
                # cannot extend the stall timer.
                when = max(timestamp, current.advanced_at) if current.advanced_at else timestamp
                advancements.append((iteration, timestamp))
                step = STEP.search(line)
                current = replace(
                    current,
                    iteration=iteration,
                    completed=completed,
                    advanced_at=when,
                    step_seconds=float(step[1]) / 1000 if step else current.step_seconds,
                )
        match = SAVE_START.search(line)
        if match:
            iteration = int(match[1])
            if current.save_iteration != iteration:
                current = replace(current, save_iteration=iteration, save_started_at=timestamp)
        match = SAVE_END.search(line)
        if match and current.save_iteration == int(match[1]) and current.save_started_at:
            duration = (timestamp - current.save_started_at).total_seconds()
            if duration >= 0:
                current = replace(current, save_seconds=duration, save_started_at=None)

    # A missing region may contain progress. Do not claim a stall from an incomplete
    # read; preserve unavailable status until recognized evidence is seen again.
    available = found_iteration or (
        not reset and not skipped_bytes and old.available and current.iteration is not None
    )
    current = replace(
        current,
        available=available,
        reason="" if available else "no recognized iteration evidence in bounded log read",
    )
    baseline = max(
        (v for v in (snapshot.checkpoint.value, current.loaded_iteration) if v is not None),
        default=None,
    )
    due = None
    if baseline is not None and current.save_interval:
        due = (baseline // current.save_interval + 1) * current.save_interval
    due_at = current.checkpoint_due_at if due == current.checkpoint_due_iteration else None
    if due is not None and current.iteration is not None and current.iteration >= due:
        due_at = due_at or min(
            (at for iteration, at in advancements if iteration >= due),
            default=current.advanced_at,
        )
    current = replace(current, checkpoint_due_iteration=due, checkpoint_due_at=due_at)
    return current
