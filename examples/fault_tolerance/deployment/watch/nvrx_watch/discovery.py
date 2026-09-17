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

"""Persistent, rate-limited discovery of explicitly tagged singleton arrays.

One invocation queries at most one user, and performs at most one batched accounting
query. All chain runners consume that persisted snapshot, never a live Slurm client.
The registry lock covers discovery, state advancement and notification deduplication.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import logging
import math
import os
import re
import shlex
import tempfile
import time
from dataclasses import replace
from datetime import datetime
from pathlib import Path

from . import discovery_notice, parsing, runner, sinks
from .config import Config
from .platform import NullPlatform, PlatformError, SlurmPlatform, _parse_slurm_time
from .types import WARNING, ChainGeneration, Finding, TaskInfo

logger = logging.getLogger("nvrx_watch")
MAX_TASKS = 100_000
MAX_NEW_SCRIPTS = 8
MAX_ACCOUNTING_JOBS = 128


def validate(config: Config) -> None:
    if config.platform != "slurm" or config.job_id or config.user:
        raise ValueError(
            "--discover-users requires Slurm and cannot be combined with job_id/--user"
        )
    if not config.observe_only:
        raise ValueError("discovery supports observe-only monitoring; omit --act")
    if any(
        not isinstance(u, str) or not re.fullmatch(r"[A-Za-z0-9_][A-Za-z0-9_.-]*", u)
        for u in config.discover_users
    ):
        raise ValueError("discover_users must contain individual user names")
    for field in ("discovery_interval", "discovery_retry_seconds", "discovery_retire_seconds"):
        value = getattr(config, field)
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"{field} must be finite and positive")
    if config.work_dir or config.cycle_info_glob or config.checkpoint_iteration_file:
        raise ValueError("discovery resolves paths per chain; omit global runtime path overrides")


def injob_marker(comment: str) -> bool:
    try:
        data = json.loads(comment)
    except (ValueError, TypeError):
        return False
    return (
        isinstance(data, dict)
        and isinstance(data.get("APS"), dict)
        and data["APS"].get("nvrx") == "enabled"
    )


def singleton_directive(script: str) -> bool:
    """Read only the SBATCH header. Last dependency wins; never source the script."""
    dependency = None
    for line in script.splitlines():
        stripped = line.strip()
        if stripped and not stripped.startswith("#"):
            break  # Slurm stops processing directives at the first executable line
        match = re.match(r"^#SBATCH(?:\s+)(.*)$", stripped)
        if not match:
            continue
        try:
            args = shlex.split(match[1], comments=True)
        except ValueError:
            return False
        for i, arg in enumerate(args):
            if arg.startswith("--dependency="):
                dependency = arg.partition("=")[2]
            elif arg in ("--dependency", "-d"):
                dependency = args[i + 1] if i + 1 < len(args) else ""
            elif arg.startswith("-d"):
                dependency = arg[2:]
    return dependency is not None and "singleton" in re.split(r"[,?]", dependency)


def _tasks(raw: str) -> list[int]:
    raw = raw.strip().strip("[]").split("%", 1)[0]
    if not raw or raw in ("N/A", "(null)"):
        return []  # non-array jobs are outside this discovery mode
    tasks = []
    for item in raw.split(","):
        match = re.fullmatch(r"(\d+)(?:-(\d+)(?::(\d+))?)?", item)
        if not match:
            raise PlatformError("invalid or truncated array task range in discovery snapshot")
        lo = int(match[1])
        hi = int(match[2] or lo)
        step = int(match[3] or 1)
        if step < 1 or hi < lo or (hi - lo) // step + 1 + len(tasks) > MAX_TASKS:
            raise PlatformError("discovery array exceeds task range bound")
        tasks.extend(range(lo, hi + 1, step))
    return tasks


def queue_snapshot(platform: SlurmPlatform, user: str) -> dict:
    # Comment is last, so pipes inside JSON values are preserved by split(maxsplit).
    # No -r: pending arrays remain compact; expand only tagged candidates locally.
    output = platform._run(
        [
            "squeue",
            "--local",
            "-h",
            "-u",
            user,
            "-o",
            "%F|%K|%u|%j|%T|%o|%Z|%k",
        ]
    )
    arrays = {}
    if output and not output.endswith("\n"):
        raise PlatformError("incomplete discovery snapshot")
    for line in output.splitlines():
        fields = line.split("|", 7)
        if len(fields) != 8:
            raise PlatformError("malformed discovery row")
        jid, indexes, owner, name, state, command, workdir, comment = (x.strip() for x in fields)
        if owner != user or not jid.isdecimal():
            raise PlatformError("unexpected owner or array id in discovery snapshot")
        if not injob_marker(comment):
            continue
        tasks = _tasks(indexes)
        if not tasks:
            continue
        identity = dict(user=owner, name=name, command=command, workdir=workdir)
        entry = arrays.setdefault(jid, {**identity, "tasks": {}})
        if any(entry[k] != value for k, value in identity.items()):
            raise PlatformError("inconsistent array metadata in discovery snapshot")
        for task in tasks:
            key = str(task)
            if key in entry["tasks"]:
                raise PlatformError("duplicate array task in discovery snapshot")
            entry["tasks"][key] = state
    return arrays


def _read_script(path: str) -> str | None:
    try:
        with open(path) as handle:
            content = handle.read(1_000_001)
        return content if len(content) <= 1_000_000 else None
    except OSError:
        return None


def resolve_metadata(row: dict) -> tuple[dict | None, str]:
    command, workdir = row["command"], row["workdir"]
    if not os.path.isabs(command):
        if not os.path.isabs(workdir):
            return None, "relative Command without an absolute WorkDir"
        command = os.path.normpath(os.path.join(workdir, command))
    script = _read_script(command)
    if script is None:
        return None, "batch script unreadable or larger than 1 MB"
    if not singleton_directive(script):
        return None, "no active SBATCH singleton dependency"
    reads = 0

    def read_include(path):
        nonlocal reads
        reads += 1
        if reads > 32:
            return None
        if not os.path.isabs(path):
            path = os.path.join(workdir, path)
        return _read_script(path)

    cyc, ckpt = parsing.resolve_ft_launcher_paths(
        script, script_path=command, read_file=read_include
    )
    if not cyc or not os.path.isabs(cyc):
        return None, "could not resolve an absolute cycle-info path"
    if ckpt and not os.path.isabs(ckpt):
        return None, "checkpoint path is not absolute"
    # A missing/unresolved checkpoint cannot safely be called 'never checkpointed'.
    # The corresponding detectors are disabled for this chain until it is resolved.
    budget = re.search(r"--max-restarts[= ]+(\d+)(?=\s|[\\\"']|$)", script)
    return (
        dict(
            cycle_info_glob=cyc,
            checkpoint_iteration_file=ckpt or "",
            max_restarts=int(budget[1]) if budget else None,
        ),
        "",
    )


def _atomic_save(path: Path, data: dict) -> None:
    fd, tmp = tempfile.mkstemp(prefix=".discovery-", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as handle:
            json.dump(data, handle)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def _warning(key: str, message: str) -> Finding:
    return Finding(key=key, detector="observer", severity=WARNING, summary=message)


def _refresh(config, registry, user, platform, now):
    entry = registry["users"][user]
    try:
        arrays = queue_snapshot(platform, user)
    except PlatformError as exc:
        entry["failures"] = min(entry.get("failures", 0) + 1, 4)
        entry["next_query"] = now + config.discovery_interval * 2 ** entry["failures"]
        entry["error"] = str(exc)
        return
    entry.update(
        arrays=arrays,
        observed=now,
        failures=0,
        error="",
        next_query=now + config.discovery_interval,
    )


def _register(config, registry, now):
    findings = []
    budget = MAX_NEW_SCRIPTS
    for user in config.discover_users:
        entry = registry["users"][user]
        if entry.get("error"):
            continue
        cache = entry.setdefault("metadata", {})
        arrays = entry.get("arrays", {})
        # Bound metadata storage; known chains retain their own resolved configuration.
        for jid in set(cache) - set(arrays):
            del cache[jid]
        for jid, row in sorted(arrays.items(), key=lambda item: int(item[0])):
            prior = cache.get(jid, {})
            signature = [row["name"], row["command"], row["workdir"]]
            if prior.get("signature") != signature:
                prior = {}
            if not prior.get("config") and now >= prior.get("retry", 0) and budget:
                budget -= 1
                metadata, reason = resolve_metadata(row)
                prior = cache[jid] = dict(
                    config=metadata,
                    reason=reason,
                    signature=signature,
                    retry=now + config.discovery_retry_seconds,
                )
            if not prior.get("config"):
                reason = prior.get("reason", "metadata inspection deferred by per-pass budget")
                findings.append(
                    _warning(
                        f"discovery-unverified-{user}-{jid}",
                        f"{user}/{row['name']} array {jid}: {reason}",
                    )
                )
                continue
            metadata = prior["config"]
            identity = [
                user,
                row["name"],
                metadata["cycle_info_glob"],
                metadata["checkpoint_iteration_file"],
            ]
            key = hashlib.sha256(json.dumps(identity).encode()).hexdigest()[:24]
            chain = registry["chains"].get(key)
            if chain is None:
                chain = registry["chains"][key] = dict(
                    user=user,
                    name=row["name"],
                    config=metadata,
                    ids=[],
                    first_id=jid,
                    absent_since=None,
                )
            if jid not in chain["ids"]:
                chain["ids"].append(jid)
                chain["config"] = metadata
    return findings


def _accounting(config, registry, user, platform, now):
    """At most one sacct, only for known arrays whose task 0 needs terminal evidence."""
    entry = registry["users"][user]
    terminal = entry.setdefault("terminal", {})
    if now < entry.get("next_accounting", 0):
        return
    wanted = set()
    for chain in registry["chains"].values():
        if chain["user"] != user:
            continue
        for jid in chain["ids"]:
            tasks = entry["arrays"].get(jid, {}).get("tasks", {})
            task0 = TaskInfo(0, tasks.get("0", ""))
            if not task0.is_live and jid not in terminal:
                wanted.add(jid)
    if not wanted:
        entry["accounting_error"] = ""
        return
    selected = sorted(wanted, key=int)[:MAX_ACCOUNTING_JOBS]
    try:
        output = platform._run(
            [
                "sacct",
                "-X",
                "-n",
                "-P",
                "-u",
                user,
                "-j",
                ",".join(f"{jid}_0" for jid in selected),
                "-o",
                "JobID,State,End,ExitCode",
            ]
        )
        for line in output.splitlines():
            parts = line.split("|")
            if len(parts) < 4:
                raise PlatformError("malformed discovery accounting response")
            jid, sep, task = parts[0].strip().partition("_")
            if jid not in wanted or task != "0":
                continue
            state = parts[1].strip().split()[0] if parts[1].strip() else ""
            end = _parse_slurm_time(parts[2])
            if state and not TaskInfo(0, state).is_live and end:
                terminal[jid] = dict(state=state, end=end.isoformat(), code=parts[3].split(":")[0])
        missing = wanted - set(terminal)
        entry["accounting_error"] = (
            f"terminal accounting unavailable for {len(missing)} array(s)" if missing else ""
        )
        entry["accounting_failures"] = 0
    except PlatformError as exc:
        entry["accounting_error"] = str(exc)
        entry["accounting_failures"] = min(entry.get("accounting_failures", 0) + 1, 4)
        entry["next_accounting"] = (
            now + config.discovery_interval * 2 ** entry["accounting_failures"]
        )


class CachedPlatform(NullPlatform):
    name = "slurm"

    def __init__(self, entry, chain, now, ttl):
        self.entry, self.chain, self.now, self.ttl = entry, chain, now, ttl

    def list_generations(self, job_name):
        if self.entry.get("error") or self.now - self.entry.get("observed", 0) > self.ttl:
            raise PlatformError(self.entry.get("error") or "discovery queue snapshot expired")
        return [
            ChainGeneration(
                jid, tuple(TaskInfo(int(task), state) for task, state in row["tasks"].items())
            )
            for jid, row in self.entry.get("arrays", {}).items()
            if jid in self.chain["ids"]
        ]

    def terminal_info(self, gen_id, task):
        raw = self.entry.get("terminal", {}).get(gen_id)
        if raw is None:
            raise PlatformError(
                self.entry.get("accounting_error") or f"terminal state of {gen_id} is unknown"
            )
        return TaskInfo(
            task,
            raw["state"],
            int(raw["code"]) if raw["code"].isdigit() else None,
            datetime.fromisoformat(raw["end"]),
        )

    @property
    def cached_endings(self):
        """Known terminal records, including evidence delayed beyond the churn window.

        The analysis outbox needs these until retirement. This property never queries
        Slurm and preserves successful records even if other accounting rows failed.
        """
        return [
            (jid, self.terminal_info(jid, 0))
            for jid in self.chain["ids"]
            if jid in self.entry.get("terminal", {})
        ]

    def recent_endings(self, job_name, since_seconds):
        if self.entry.get("accounting_error"):
            raise PlatformError(self.entry["accounting_error"])
        result = []
        for jid in self.chain["ids"]:
            if jid in self.entry.get("terminal", {}):
                task = self.terminal_info(jid, 0)
                if self.now - task.end_time.timestamp() <= since_seconds:
                    result.append((jid, task))
        return result


class ChainSink:
    """Keep alert routing/dedup intact while identifying the owner and chain."""

    def __init__(self, sink, user, name):
        self.sink, self.user, self.chain_name = sink, user, name
        self.name = sink.name

    def emit(self, finding):
        return self.sink.emit(
            replace(finding, summary=f"[{self.user}/{self.chain_name}] {finding.summary}")
        )


def _pass(config, registry, now, save):
    users = registry.setdefault("users", {})
    registry.setdefault("chains", {})
    for i, user in enumerate(config.discover_users):
        users.setdefault(
            user, dict(next_query=now + i * config.discovery_interval / len(config.discover_users))
        )
    # One user per invocation, also after downtime. No catch-up burst.
    due = [u for u in config.discover_users if users[u]["next_query"] <= now]
    selected = min(due, key=lambda u: users[u]["next_query"]) if due else None
    if selected:
        # Persist the budget before contacting Slurm. An interrupted process or
        # stalled filesystem read must not cause another query at the next cron tick.
        users[selected]["next_query"] = now + config.discovery_interval
        save()
        platform = SlurmPlatform(timeout=config.command_timeout, user=selected)
        _refresh(config, registry, selected, platform, now)
        save()
    findings = _register(config, registry, now)
    if selected and not users[selected].get("error"):
        _accounting(config, registry, selected, platform, now)
        save()
    for user in config.discover_users:
        entry = users[user]
        if entry.get("error"):
            findings.append(
                _warning(
                    f"discovery-blind-{user}", f"Discovery cannot observe {user}: {entry['error']}"
                )
            )
        if entry.get("accounting_error"):
            findings.append(
                _warning(
                    f"discovery-accounting-{user}",
                    f"Discovery accounting incomplete for {user}: {entry['accounting_error']}",
                )
            )
    runner.report(findings, config, sinks.build(config))
    exit_code = 1 if findings else 0
    for key, chain in list(registry["chains"].items()):
        user = chain["user"]
        if user not in config.discover_users:
            continue
        entry = users[user]
        known = set(chain["ids"]) & set(entry.get("arrays", {}))
        healthy = (
            not entry.get("error")
            and now - entry.get("observed", 0) <= config.discovery_interval * 2
        )
        retire = False
        if healthy:
            if known:
                chain["absent_since"] = None
            elif chain["absent_since"] is None:
                chain["absent_since"] = entry["observed"]
            elif entry["observed"] - chain[
                "absent_since"
            ] >= config.discovery_retire_seconds and all(
                jid in entry.get("terminal", {}) for jid in chain["ids"]
            ):
                retire = True
        state = str(Path(config.state_dir) / "chains" / f"{key}-{chain['first_id']}")
        disabled = set(config.disable)
        if not chain["config"]["checkpoint_iteration_file"]:
            disabled.update(("stalled_progress", "cycle_stalled"))
            logger.warning(
                "%s/%s: checkpoint path unresolved; checkpoint stall checks disabled",
                user,
                chain["name"],
            )
        cfg = replace(
            config,
            **chain["config"],
            discover_users=(),
            user=user,
            job_id=chain["first_id"],
            job_name=chain["name"],
            cycle_job_ids=tuple(chain["ids"]),
            state_dir=state,
            log_file=str(Path(state) / "watch.log"),
            expect_file=str(Path(state) / "expected"),
            heartbeat_url="",
            disable=tuple(disabled),
        )
        cached = CachedPlatform(entry, chain, now, config.discovery_interval * 2)
        try:
            chain_sinks = [ChainSink(s, user, chain["name"]) for s in sinks.build(cfg)]
            result = runner.run_once(cfg, cached, sink_list=chain_sinks)
            if not retire and not result.degraded:
                discovery_notice.notify(cfg, key, chain, result.snapshot, chain_sinks, save)
            if retire and not result.degraded:
                logger.info("retiring discovery chain %s/%s", user, chain["name"])
                del registry["chains"][key]
            exit_code = 1 if result.degraded or exit_code == 1 else max(exit_code, result.exit_code)
        except Exception:
            logger.exception("watch failed for %s/%s", user, chain["name"])
            exit_code = 1
    if (
        exit_code != 1
        and not config.dry_run
        and all(
            users[u].get("observed")
            and now - users[u]["observed"] <= config.discovery_interval * 2
            and not users[u].get("error")
            for u in config.discover_users
        )
    ):
        sinks.heartbeat(config.heartbeat_url)
    return exit_code


def run_once(config: Config) -> int:
    validate(config)
    directory = Path(config.state_dir)
    directory.mkdir(parents=True, exist_ok=True, mode=0o700)
    with (directory / "discovery.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            logger.info("discovery pass already running; skipped")
            return 0
        path = directory / "discovery.json"
        try:
            registry = json.loads(path.read_text()) if path.exists() else {}
            if not isinstance(registry, dict):
                raise ValueError("registry must be an object")
        except (OSError, ValueError):
            logger.exception("cannot read discovery registry; refusing to discard query budgets")
            return 1

        def save():
            if not config.dry_run:
                _atomic_save(path, registry)

        result = _pass(config, registry, time.time(), save)
        save()
        return result
