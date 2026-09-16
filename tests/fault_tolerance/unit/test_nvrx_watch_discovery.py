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

"""Discovery regressions; fake Slurm replies, real registry and chain state files."""

import fcntl
import json
import sys
from pathlib import Path

import pytest

WATCH_DIR = Path(__file__).resolve().parents[3] / "examples/fault_tolerance/deployment/watch"
sys.path.insert(0, str(WATCH_DIR))

from nvrx_watch import discovery, readers, sinks  # noqa: E402
from nvrx_watch.__main__ import main  # noqa: E402
from nvrx_watch.config import Config, load  # noqa: E402
from nvrx_watch.platform import PlatformError, SlurmPlatform  # noqa: E402


@pytest.mark.parametrize(
    "comment,expected",
    [
        ('{"APS":{"nvrx":"enabled"}}', True),
        ('{"other":"a|b", "APS": {"environment":"prod", "nvrx": "enabled"}}', True),
        ('{"APS":{"nvrx":"disabled"}}', False),
        ('{"APS":{"nvrx":true}}', False),
        ('{"nvrx":"enabled"}', False),
        ('{"APS":"enabled"}', False),
        ('null', False),
        ('[]', False),
        ('(null)', False),
        ('{bad', False),
    ],
)
def test_marker(comment, expected):
    assert discovery.injob_marker(comment) is expected


@pytest.mark.parametrize(
    "script,expected",
    [
        ('#!/bin/bash\n#SBATCH --dependency=singleton\necho hi', True),
        ('#SBATCH --dependency singleton', True),
        ('#SBATCH -d singleton', True),
        ('#SBATCH --dependency=afterany:123,singleton', True),
        ('#SBATCH --dependency=singletonish', False),
        ('##SBATCH --dependency=singleton', False),
        ('# example: #SBATCH --dependency=singleton', False),
        ('echo hi\n#SBATCH --dependency=singleton', False),
        ('#SBATCH --dependency=singleton\n#SBATCH --dependency=afterok:123', False),
        ('#SBATCH --dependency="singleton', False),
    ],
)
def test_singleton_header(script, expected):
    assert discovery.singleton_directive(script) is expected


@pytest.fixture
def harness(tmp_path, monkeypatch):
    class Harness:
        now = 1_800_000_000.0
        calls = []
        queues = {}
        accounting = ""
        messages = []
        beats = []

        def config(self, **kwargs):
            return Config(
                discover_users=("alice", "bob"),
                state_dir=str(tmp_path / "state"),
                notify_cycle_restarts=True,
                **kwargs,
            )

        def row(self, user="alice", jid="100", name="train", task="0", state="RUNNING", root=None):
            root = root or tmp_path / user / name
            root.mkdir(parents=True, exist_ok=True)
            script = root / "run.sh"
            if not script.exists():
                script.write_text(
                    '#!/bin/bash\n#SBATCH --dependency=singleton\n'
                    f'ROOT="{root}"\nARGS="--ft-cycle-info-dir=${{ROOT}}/nvrx/${{SLURM_ARRAY_JOB_ID}}/cycle_infos '
                    '--ft-checkpoint-iteration-file=${ROOT}/checkpoints/latest_checkpointed_iteration.txt '
                    '--max-restarts 100"\n'
                )
            return f'{jid}|{task}|{user}|{name}|{state}|../run.sh|{root}/slurm_out|{{"APS":{{"nvrx":"enabled"}}}}\n'

        def run(self, config=None):
            return discovery.run_once(config or self.config())

        def registry(self):
            return json.loads((tmp_path / "state/discovery.json").read_text())

    h = Harness()

    def query(self, args):
        h.calls.append(args)
        if args[0] == "squeue":
            result = h.queues.get(args[args.index("-u") + 1], "")
        elif args[0] == "sacct":
            result = h.accounting
        else:
            pytest.fail(f"unexpected scheduler command: {args}")
        if isinstance(result, Exception):
            raise result
        return result

    class Sink:
        name = "test"

        def emit(self, finding):
            h.messages.append(finding)
            return True

    monkeypatch.setattr(SlurmPlatform, "_run", query)
    monkeypatch.setattr(discovery.time, "time", lambda: h.now)
    monkeypatch.setattr(sinks, "build", lambda config: [Sink()])
    monkeypatch.setattr(sinks, "heartbeat", h.beats.append)
    return h


def test_stagger_persist_and_reuse_for_all_chains(harness):
    h = harness
    h.queues["alice"] = h.row(name="one") + h.row(jid="101", name="two")
    h.queues["bob"] = h.row(user="bob", jid="200")
    assert h.run() == 0
    assert len(h.calls) == 1
    assert len(h.registry()["chains"]) == 2
    h.now += 180
    assert h.run() == 0
    assert len(h.calls) == 1  # cycle-file checks do not query Slurm
    h.now += 180
    assert h.run() == 0
    assert len(h.calls) == 2 and h.calls[-1][h.calls[-1].index("-u") + 1] == "bob"
    assert len(h.registry()["chains"]) == 3
    h.now += 10_000
    h.run()
    assert len(h.calls) == 3  # no catch-up fan-out after downtime


def test_cache_script_and_join_successor_without_seed_lookup(harness, monkeypatch):
    h = harness
    h.queues["alice"] = h.row()
    h.run()
    chain = next(iter(h.registry()["chains"].values()))
    first_id = chain["first_id"]
    original = discovery.resolve_metadata
    calls = []
    monkeypatch.setattr(
        discovery, "resolve_metadata", lambda row: (calls.append(row) or original(row))
    )
    h.now += 600
    h.run()  # bob was due first
    h.now += 1
    h.run()  # alice refresh, no script read
    assert not calls
    h.queues["alice"] += h.row(jid="101", task="[0-5%2]", state="PENDING")
    h.now += 600
    h.run()
    h.now += 1
    h.run()
    chain = next(iter(h.registry()["chains"].values()))
    assert chain["ids"] == ["100", "101"] and chain["first_id"] == first_id
    assert len(calls) == 1
    assert all(c[0] == "squeue" for c in h.calls)


def test_compact_array_states_and_comment_pipe(harness):
    h = harness
    h.queues["alice"] = h.row(task="0") + h.row(task="[1-5%2]", state="PENDING")
    h.queues["alice"] = h.queues["alice"].replace('"APS":', '"note":"a|b","APS":')
    assert h.run() == 0
    tasks = h.registry()["users"]["alice"]["arrays"]["100"]["tasks"]
    assert len(tasks) == 6 and tasks["5"] == "PENDING"


@pytest.mark.parametrize("bad", ["0-999999999", "1-5...", "5-1", "1-3:0"])
def test_bad_ranges_are_not_empty_success(bad):
    with pytest.raises(PlatformError):
        discovery._tasks(bad)


def test_unverified_script_not_enrolled_and_retry_is_slow(harness):
    h = harness
    h.queues["alice"] = h.row()
    script = Path(h.queues["alice"].split("|")[6]).parent / "run.sh"
    script.write_text('echo hello\n#SBATCH --dependency=singleton\n')
    assert h.run() == 1
    assert not h.registry()["chains"]
    assert any("no active SBATCH" in f.summary for f in h.messages)
    script.unlink()
    h.now += 180
    h.run()
    assert len(h.messages) == 1  # persistent notification cooldown
    assert not h.registry()["chains"]


def test_failure_preserves_chains_and_backs_off(harness):
    h = harness
    h.queues["alice"] = h.row()
    h.run()
    h.queues["alice"] = PlatformError("timeout")
    h.now += 600
    h.run()
    h.now += 1
    assert h.run() == 1
    reg = h.registry()
    entry = reg["users"]["alice"]
    assert entry["next_query"] == h.now + 1200
    assert len(reg["chains"]) == 1
    assert next(iter(reg["chains"].values()))["absent_since"] is None
    assert not any(f.detector == "spares_exhausted" for f in h.messages)
    calls = len(h.calls)
    h.now += 180
    h.run()
    assert len(h.calls) == calls


def test_terminal_query_batched_and_chain_retires(harness):
    h = harness
    cfg = h.config()
    cfg.discover_users = ("alice",)
    h.queues["alice"] = h.row() + h.row(jid="101")
    h.run(cfg)
    h.queues["alice"] = ""
    h.accounting = (
        "100_0|COMPLETED|2027-01-15T07:00:00|0:0|\n101_0|CANCELLED|2027-01-15T07:00:00|0:0|\n"
    )
    h.now += 600
    h.run(cfg)
    sacct = [c for c in h.calls if c[0] == "sacct"]
    assert len(sacct) == 1 and sacct[0][sacct[0].index("-j") + 1] == "100_0,101_0"
    h.now += 600
    h.run(cfg)
    h.now += 600
    h.run(cfg)
    assert not h.registry()["chains"]
    assert len([c for c in h.calls if c[0] == "sacct"]) == 1


def test_dry_run_no_registry_or_notification(harness):
    h = harness
    h.queues["alice"] = h.row()
    cfg = h.config(dry_run=True)
    h.run(cfg)
    assert not Path(cfg.state_dir, "discovery.json").exists()
    assert not h.messages and not h.beats


def test_lock_and_corrupt_registry_do_not_query(harness):
    h = harness
    cfg = h.config()
    Path(cfg.state_dir).mkdir()
    with open(Path(cfg.state_dir, "discovery.lock"), "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        assert h.run(cfg) == 0 and not h.calls
    Path(cfg.state_dir, "discovery.json").write_text("broken")
    assert h.run(cfg) == 1 and not h.calls


def test_cli_routes_discovery_and_rejects_act(harness):
    h = harness
    assert main(["--discover-users", "alice,bob", "--state-dir", h.config().state_dir]) == 0
    before = len(h.calls)
    assert main(["--discover-users", "alice", "--act"]) == 1
    assert main(["123", "--discover-users", "alice"]) == 1
    assert len(h.calls) == before


def test_config_user_list():
    assert load(env={"NVRX_WATCH_DISCOVER_USERS": "alice, bob,alice"}).discover_users == (
        "alice",
        "bob",
    )
    with pytest.raises(ValueError):
        discovery.validate(Config(discover_users=("alice",), discovery_interval=float("nan")))


def test_old_cycle_history_is_not_read(tmp_path, monkeypatch):
    (tmp_path / "cycle_info.100.0.0").write_text('{}')
    (tmp_path / "cycle_info.200.0.0").write_text('{}')
    read = []
    monkeypatch.setattr(readers, "parse_cycle_file", lambda p: read.append(p))
    readers.read_cycles(str(tmp_path / "cycle_info.*"), ("200",))
    assert len(read) == 1 and "200" in read[0]


def test_unknown_accounting_preserves_chain_and_withholds_heartbeat(harness):
    h = harness
    cfg = h.config()
    cfg.discover_users = ("alice",)
    h.queues["alice"] = h.row()
    h.run(cfg)
    h.beats.clear()
    h.queues["alice"] = ""
    for _ in range(4):
        h.now += 600
        assert h.run(cfg) == 1
    assert h.registry()["chains"]
    assert not any(h.beats)
    assert any("accounting incomplete" in f.summary for f in h.messages)


def test_accounting_failure_backoff(harness):
    h = harness
    cfg = h.config()
    cfg.discover_users = ("alice",)
    h.queues["alice"] = h.row()
    h.run(cfg)
    h.queues["alice"] = ""
    h.accounting = PlatformError("db timeout")
    h.now += 600
    h.run(cfg)
    calls = len([c for c in h.calls if c[0] == "sacct"])
    h.now += 600
    h.run(cfg)
    assert len([c for c in h.calls if c[0] == "sacct"]) == calls


def test_scope_change_does_not_query_removed_user(harness):
    h = harness
    h.queues["alice"] = h.row()
    h.run()
    cfg = h.config()
    cfg.discover_users = ("bob",)
    h.now += 600
    h.run(cfg)
    assert h.calls[-1][h.calls[-1].index("-u") + 1] == "bob"
    assert len(h.registry()["chains"]) == 1  # historical state is retained


def test_retired_chain_reuse_gets_new_state(harness):
    h = harness
    cfg = h.config()
    cfg.discover_users = ("alice",)
    h.queues["alice"] = h.row()
    h.run(cfg)
    h.queues["alice"] = ""
    h.accounting = "100_0|COMPLETED|2027-01-15T07:00:00|0:0|\n"
    for _ in range(3):
        h.now += 600
        h.run(cfg)
    assert not h.registry()["chains"]
    h.queues["alice"] = h.row(jid="200")
    h.now += 600
    h.run(cfg)
    chain = next(iter(h.registry()["chains"].values()))
    assert chain["first_id"] == "200" and chain["ids"] == ["200"]


def test_query_budget_survives_interrupted_metadata_read(harness, monkeypatch):
    h = harness
    h.queues["alice"] = h.row()

    def interrupted(row):
        raise KeyboardInterrupt()

    monkeypatch.setattr(discovery, "resolve_metadata", interrupted)
    with pytest.raises(KeyboardInterrupt):
        h.run()
    assert h.registry()["users"]["alice"]["next_query"] == h.now + 600
    h.now += 180
    with pytest.raises(KeyboardInterrupt):
        h.run()
    assert len(h.calls) == 1


def test_marker_absent_skips_script_reads(harness, monkeypatch):
    h = harness
    h.queues["alice"] = h.row().replace('"enabled"', '"disabled"')
    monkeypatch.setattr(discovery, "resolve_metadata", lambda row: pytest.fail("read untagged job"))
    assert h.run() == 0 and not h.registry()["chains"]


def test_same_user_name_different_runtime_stays_separate(harness, tmp_path):
    h = harness
    h.queues["alice"] = h.row(root=tmp_path / "run1") + h.row(jid="101", root=tmp_path / "run2")
    assert h.run() == 0
    assert len(h.registry()["chains"]) == 2


def test_unresolved_paths_do_not_run_shell(tmp_path):
    script = tmp_path / "run.sh"
    touched = tmp_path / "executed"
    script.write_text(
        '#SBATCH --dependency=singleton\n'
        f'ROOT=$(touch {touched})\n'
        'ARGS="--ft-cycle-info-dir=${ROOT}/cycles"\n'
    )
    metadata, reason = discovery.resolve_metadata(dict(command=str(script), workdir=str(tmp_path)))
    assert metadata is None and "could not resolve" in reason
    assert not touched.exists()


def test_discovery_notice_once_across_successors_and_cooldown(harness):
    h = harness
    cfg = h.config()
    cfg.discover_users = ("alice",)
    h.queues["alice"] = h.row(jid="100", state="PENDING") + h.row(jid="101")
    assert h.run(cfg) == 0
    notices = [f for f in h.messages if f.detector == "injob_discovered"]
    assert len(notices) == 1
    assert notices[0].summary == (
        "[alice/train] Discovered InJob; monitoring started for Slurm job array 101."
    )
    assert len(h.calls) == 1  # enrollment adds no scheduler queries
    for _ in range(2):
        h.now += 86400
        h.queues["alice"] = h.row(jid="101") + h.row(jid="102", state="PENDING")
        h.run(cfg)
    assert len([f for f in h.messages if f.detector == "injob_discovered"]) == 1


def test_discovery_notice_retries_only_failed_sink(harness, monkeypatch):
    h = harness
    h.queues["alice"] = h.row()
    deliveries = {"good": [], "retry": []}

    class Sink:
        def __init__(self, name):
            self.name = name

        def emit(self, finding):
            deliveries[self.name].append(finding)
            return self.name == "good" or len(deliveries[self.name]) > 1

    monkeypatch.setattr(sinks, "build", lambda config: [Sink("good"), Sink("retry")])
    assert h.run() == 0
    chain = next(iter(h.registry()["chains"].values()))
    assert chain["discovery_notice"]["sent"] == ["good"]
    h.now += 180
    assert h.run() == 0
    assert len(deliveries["good"]) == 1
    assert len(deliveries["retry"]) == 2
    assert deliveries["retry"][0] == deliveries["retry"][1]
    assert len(h.calls) == 1  # retry uses cached data, with no scheduler query
    chain = next(iter(h.registry()["chains"].values()))
    assert chain["discovery_notice"]["sent"] == ["good", "retry"]


def test_existing_chain_gets_one_discovery_notice_after_upgrade(harness):
    h = harness
    h.queues["alice"] = h.row()
    h.run()
    registry = h.registry()
    for chain in registry["chains"].values():
        chain.pop("discovery_notice")
    path = Path(h.config().state_dir) / "discovery.json"
    path.write_text(json.dumps(registry))
    h.messages.clear()
    h.now += 180
    h.run()
    h.run()
    assert len([f for f in h.messages if f.detector == "injob_discovered"]) == 1
