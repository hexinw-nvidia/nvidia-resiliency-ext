# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Link independently posted attribution alerts to the exact watcher cycle thread."""

import json
import re
import sys
from dataclasses import replace
from pathlib import Path

import pytest

sys.path.insert(
    0, str(Path(__file__).resolve().parents[3] / "examples/fault_tolerance/deployment/watch")
)
from nvrx_watch import events, restart_links, slack_threads
from nvrx_watch.config import Config


@pytest.fixture
def bridge(tmp_path, monkeypatch):
    class Bridge:
        now = 100000.0
        calls = []
        messages = []
        replies = []
        history = None
        source_message = None
        post_error = None
        config = Config(
            event_queue_dir=str(tmp_path),
            slack_bot_token_file=str(tmp_path / "token"),
            slack_channel_id="CWATCH",
            restart_agent_slack_channel_id="CAGENT",
        )
        event = {
            "event_id": "a" * 64,
            "kind": "cycle_restart",
            "chain": {"user": "alice", "job_name": "train"},
            "cycle": {
                "job_id": "100",
                "attempt_index": 0,
                "cycle_number": 4,
                "log_file": "/run/train_100_date_start_cycle4.log",
            },
        }
        parent = {"event_id": "a" * 64, "channel": "CWATCH", "thread_ts": "100.123"}

        def message(self, path=None, job="100", cycle=4, owner="alice", ts="101.123"):
            return {
                "ts": ts,
                "text": (
                    "*NVRx attribution:* `STOP`\n"
                    f"*Job ID:* `{job}` | *Cycle:* {cycle} | *Owner:* `{owner}`\n"
                    "*Log path:*\n```" + (path or self.event["cycle"]["log_file"]) + "```"
                ),
            }

        def enqueue(self):
            events.atomic_json(
                tmp_path / "slack-receipts" / (self.event["event_id"] + ".json"), self.parent
            )
            restart_links.enqueue(self.config, self.event)

    b = Bridge()

    def api(config, method, payload, **kwargs):
        b.calls.append((method, payload))
        if method == "conversations.history":
            if payload.get("inclusive") == "true":
                return {"messages": [b.source_message or b.message()]}
            if isinstance(b.history, Exception):
                raise b.history
            return b.history or {"messages": b.messages, "has_more": False}
        if method == "conversations.replies":
            return {"messages": b.replies, "has_more": False}
        if method == "chat.getPermalink":
            assert payload == {"channel": "CAGENT", "message_ts": "101.123"}
            return {"permalink": "https://team.slack.com/archives/CAGENT/p101123"}
        if method == "chat.postMessage":
            if b.post_error:
                raise b.post_error
            return {"channel": "CWATCH", "ts": "102.123"}
        pytest.fail("unexpected API method: " + method)

    monkeypatch.setattr(slack_threads, "api_call", api)
    monkeypatch.setattr(restart_links.time, "time", lambda: b.now)
    return b


def test_exact_match_links_into_watcher_thread_and_deduplicates(bridge):
    b = bridge
    b.messages = [b.message()]
    b.enqueue()
    restart_links.flush(b.config)
    posts = [p for m, p in b.calls if m == "chat.postMessage"]
    assert len(posts) == 1
    assert posts[0]["channel"] == "CWATCH" and posts[0]["thread_ts"] == "100.123"
    assert "https://team.slack.com/archives/CAGENT/p101123" in posts[0]["text"]
    assert all(p.get("channel") != "CAGENT" for m, p in b.calls if m == "chat.postMessage")
    b.enqueue()  # Replay after a crash between receipt save and pending removal.
    b.now += 180
    restart_links.flush(b.config)
    assert len([m for m, _ in b.calls if m == "chat.postMessage"]) == 1


@pytest.mark.parametrize(
    "changes",
    [
        {"path": "/other/train_100_date_start_cycle4.log"},
        {"job": "1000"},
        {"cycle": 40},
        {"owner": "bob"},
    ],
)
def test_different_log_job_cycle_or_owner_is_never_linked(bridge, changes):
    b = bridge
    b.messages = [b.message(**changes)]
    b.enqueue()
    restart_links.flush(b.config)
    assert [m for m, _ in b.calls] == ["conversations.history"]


def test_ambiguous_sources_are_not_linked(bridge):
    b = bridge
    b.messages = [b.message(), b.message(ts="101.124")]
    b.enqueue()
    restart_links.flush(b.config)
    assert [m for m, _ in b.calls] == ["conversations.history"]


def test_manual_link_and_lost_post_response_do_not_duplicate(bridge):
    b = bridge
    b.messages = [b.message()]
    b.enqueue()
    b.post_error = RuntimeError("lost HTTP response")
    restart_links.flush(b.config)
    b.replies = [
        {"ts": "102.123", "text": "<https://team.enterprise.slack.com/archives/CAGENT/p101123>"}
    ]
    b.now += 180
    b.post_error = None
    restart_links.flush(b.config)
    assert len([m for m, _ in b.calls if m == "chat.postMessage"]) == 1


def test_delayed_source_and_persistent_history_cursor(bridge):
    b = bridge
    b.enqueue()
    b.history = {"messages": [], "has_more": True, "response_metadata": {"next_cursor": "page2"}}
    restart_links.flush(b.config)
    restart_links.flush(b.config)
    assert len(b.calls) == 1
    b.now += 180
    b.history = {"messages": [b.message()], "has_more": False}
    restart_links.flush(b.config)
    assert b.calls[1][1]["cursor"] == "page2"
    assert any(m == "chat.postMessage" for m, _ in b.calls)


def test_one_shared_history_query_and_one_link_per_pass_for_multiple_events(bridge):
    b = bridge
    b.enqueue()
    b.event = {**b.event, "event_id": "b" * 64}
    b.parent = {**b.parent, "event_id": "b" * 64, "thread_ts": "200.123"}
    b.enqueue()
    b.messages = [b.message()]
    restart_links.flush(b.config)
    assert sum(m == "conversations.history" and "inclusive" not in p for m, p in b.calls) == 1
    assert sum(m == "conversations.history" and "inclusive" in p for m, p in b.calls) == 1
    assert sum(m == "chat.postMessage" for m, _ in b.calls) == 1


def test_access_failure_backoff_and_dry_run(bridge):
    b = bridge
    b.enqueue()
    restart_links.flush(replace(b.config, dry_run=True))
    assert not b.calls
    b.history = RuntimeError("not_in_channel")
    restart_links.flush(b.config)
    restart_links.flush(b.config)
    assert len(b.calls) == 1
    b.now += 600
    b.history = None
    b.messages = [b.message()]
    restart_links.flush(b.config)
    assert any(m == "chat.postMessage" for m, _ in b.calls)


def test_rate_limit_and_bridge_failure_do_not_break_monitoring(bridge):
    b = bridge
    b.enqueue()
    b.history = slack_threads.RetryLater(900)
    assert slack_threads.flush(b.config)
    b.now += 600
    restart_links.flush(b.config)
    assert len(b.calls) == 1
    b.now += 300
    b.history = None
    restart_links.flush(b.config)
    assert len(b.calls) == 2


def test_config_requires_separate_source_and_watcher_bot():
    with pytest.raises(ValueError):
        Config(restart_agent_slack_channel_id="CAGENT")
    with pytest.raises(ValueError):
        Config(
            event_queue_dir="/private",
            slack_bot_token_file="/token",
            slack_channel_id="C1",
            restart_agent_slack_channel_id="C1",
        )


def test_partial_history_cannot_establish_a_unique_match(bridge):
    b = bridge
    b.enqueue()
    b.history = {
        "messages": [b.message()],
        "has_more": True,
        "response_metadata": {"next_cursor": "page2"},
    }
    restart_links.flush(b.config)
    assert [m for m, _ in b.calls] == ["conversations.history"]
    b.now += 180
    b.history = {"messages": [b.message(ts="101.124")], "has_more": False}
    restart_links.flush(b.config)
    assert [m for m, _ in b.calls] == ["conversations.history", "conversations.history"]


def test_wrong_watcher_parent_channel_is_rejected(bridge):
    b = bridge
    b.messages = [b.message()]
    b.parent = {**b.parent, "channel": "COTHER"}
    b.enqueue()
    restart_links.flush(b.config)
    assert not any(m == "chat.postMessage" for m, _ in b.calls)


def test_expired_unmatched_event_is_retained_as_receipt(bridge):
    b = bridge
    b.enqueue()
    b.now += 8 * 86400
    restart_links.flush(b.config)
    receipt = (
        Path(b.config.event_queue_dir) / "restart-link-receipts" / (b.event["event_id"] + ".json")
    )
    assert json.loads(receipt.read_text())["expired"]
    assert not any(m == "chat.postMessage" for m, _ in b.calls)


def test_source_thread_reply_is_not_treated_as_an_attribution_parent(bridge):
    b = bridge
    b.messages = [{**b.message(), "thread_ts": "100.000"}]
    b.enqueue()
    restart_links.flush(b.config)
    assert [m for m, _ in b.calls] == ["conversations.history"]


def test_parent_crash_replay_preserves_link_intent(bridge, monkeypatch):
    b = bridge
    parent_pending = (
        Path(b.config.event_queue_dir) / "slack-pending" / (b.event["event_id"] + ".json")
    )
    events.atomic_json(parent_pending, b.event)
    events.atomic_json(
        parent_pending.parent.parent / "slack-receipts" / parent_pending.name, b.parent
    )
    from nvrx_watch import cycle_logs

    monkeypatch.setattr(cycle_logs, "enqueue", lambda *args: None)
    monkeypatch.setattr(cycle_logs, "flush", lambda *args: True)
    monkeypatch.setattr(restart_links, "flush", lambda *args: None)
    assert slack_threads.flush(b.config)
    assert not parent_pending.exists()
    assert (parent_pending.parent.parent / "restart-link-pending" / parent_pending.name).exists()


def test_pending_rotation_does_not_starve_new_matches(bridge):
    b = bridge
    original_event = b.event
    original_parent = b.parent
    for i in range(100):
        event_id = format(i, "064x")
        b.event = {
            **original_event,
            "event_id": event_id,
            "cycle": {**original_event["cycle"], "log_file": f"/other/log{i}"},
        }
        b.parent = {**original_parent, "event_id": event_id}
        b.enqueue()
    b.event = {**original_event, "event_id": "f" * 64}
    b.parent = {**original_parent, "event_id": "f" * 64}
    b.enqueue()
    b.messages = [b.message()]
    restart_links.flush(b.config)
    assert not any(m == "chat.postMessage" for m, _ in b.calls)
    b.now += 180
    restart_links.flush(b.config)
    assert any(m == "chat.postMessage" for m, _ in b.calls)


def test_cursor_is_restarted_after_an_outage(bridge):
    b = bridge
    b.enqueue()
    b.history = {"messages": [], "has_more": True, "response_metadata": {"next_cursor": "expires"}}
    restart_links.flush(b.config)
    b.now += 180
    b.history = RuntimeError("invalid_cursor")
    restart_links.flush(b.config)
    b.now += 600
    b.history = None
    b.messages = [b.message()]
    restart_links.flush(b.config)
    history = [p for m, p in b.calls if m == "conversations.history"]
    assert history[1]["cursor"] == "expires" and "cursor" not in history[2]
    assert history[0]["oldest"] == history[2]["oldest"]
    assert any(m == "chat.postMessage" for m, _ in b.calls)


def test_history_timestamps_have_exactly_microsecond_precision(bridge):
    b = bridge
    b.now = 100000.1234567
    b.enqueue()
    restart_links.flush(b.config)
    params = next(p for m, p in b.calls if m == "conversations.history")
    assert re.fullmatch(r"\d+\.\d{6}", params["oldest"])
    assert re.fullmatch(r"\d+\.\d{6}", params["latest"])


@pytest.mark.parametrize("verdict", ["STOP", "RESTART"])
def test_reply_includes_explicit_source_verdict_and_short_reason(bridge, verdict):
    b = bridge
    message = b.message()
    message["text"] = message["text"].replace("`STOP`", f"`{verdict}`")
    message["text"] += f"\n*Why {verdict}:* retry budget exhausted"
    b.messages = [message]
    b.source_message = message
    b.enqueue()
    restart_links.flush(b.config)
    post = next(p for m, p in b.calls if m == "chat.postMessage")
    assert f"*Restart-agent verdict: {verdict}*" in post["text"]
    assert "> retry budget exhausted" in post["text"]
    assert "https://team.slack.com/archives/CAGENT/p101123" in post["text"]


def test_preview_uses_latest_source_verdict_and_escapes_mentions(bridge):
    b = bridge
    b.messages = [b.message()]
    updated = b.message()
    updated["text"] = updated["text"].replace("`STOP`", "`RESTART`")
    updated["text"] += "\n*Why RESTART:* recovered <@U123> <!channel> & " + "x" * 600
    b.source_message = updated
    b.enqueue()
    restart_links.flush(b.config)
    text = next(p["text"] for m, p in b.calls if m == "chat.postMessage")
    assert "*Restart-agent verdict: RESTART*" in text
    assert "*Restart-agent verdict: STOP*" not in text
    assert "<@U123>" not in text and "<!channel>" not in text
    assert "&lt;@U123&gt;" in text and "&amp;" in text
    assert len(text) < 550


def test_source_changed_to_another_cycle_is_not_forwarded(bridge):
    b = bridge
    b.messages = [b.message()]
    b.source_message = b.message(cycle=5)
    b.enqueue()
    restart_links.flush(b.config)
    assert not any(m == "chat.postMessage" for m, _ in b.calls)


def test_verdict_without_reason_does_not_invent_a_reason(bridge):
    b = bridge
    b.messages = [b.message()]
    b.enqueue()
    restart_links.flush(b.config)
    text = next(p["text"] for m, p in b.calls if m == "chat.postMessage")
    assert "*Restart-agent verdict: STOP*" in text
    assert "> " not in text
