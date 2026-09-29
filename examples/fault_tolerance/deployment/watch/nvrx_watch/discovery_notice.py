# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""One enrollment notification per chain and sink, with durable delivery receipts."""

from __future__ import annotations

import logging

from .types import INFO, Finding

logger = logging.getLogger("nvrx_watch")


def notify(config, key, chain, snapshot, sink_list, save):
    if config.dry_run:
        logger.info(
            "[dry-run] would announce discovered InJob: %s/%s", chain["user"], chain["name"]
        )
        return

    if config.slack_bot_token_file:
        return _notify_threaded(config, key, chain, snapshot, sink_list, save)

    notice = chain.get("discovery_notice")
    if notice is None:
        running = next(
            (
                g.gen_id
                for g in snapshot.generations
                if any(t.is_live and t.state != "PENDING" for t in g.tasks)
            ),
            None,
        )
        notice = chain["discovery_notice"] = {
            "job_id": running or snapshot.current_job_id or chain["first_id"],
            "observed_at": snapshot.observed_at.isoformat(),
            "sent": [],
        }
        # Keep the same message and identity if delivery needs another cron pass.
        save()

    finding = Finding(
        key=f"nvrx-discovered-{key}-{chain['first_id']}",
        detector="injob_discovered",
        severity=INFO,
        summary=f"Discovered InJob; monitoring started for Slurm job array {notice['job_id']}.",
        detail=(
            f"Observed at {notice['observed_at']}; InJob marker and singleton directive verified. "
            "Monitoring this singleton chain, including successor arrays."
        ),
    )
    for sink in sink_list:
        if sink.name in notice["sent"]:
            continue
        if sink.emit(finding):
            notice["sent"].append(sink.name)
            # Save each successful sink separately; failed sinks retry on the next pass.
            save()


def _notify_threaded(config, key, chain, snapshot, sink_list, save):
    """Give each newly observed active array its own parent and cycle-log reply."""
    from .cycle_logs import enqueue_start

    notices = chain.setdefault("discovery_notices", {})
    legacy = chain.get("discovery_notice", {})
    for generation in snapshot.generations:
        if not any(t.is_live and t.state != "PENDING" for t in generation.tasks):
            continue
        job = generation.gen_id
        if job not in notices:
            notices[job] = {
                "job_id": job,
                "observed_at": snapshot.observed_at.isoformat(),
                "sent": list(legacy.get("sent", [])) if legacy.get("job_id") == job else [],
            }
            save()
    for notice in notices.values():
        finding = Finding(
            key=f"nvrx-discovered-{key}-{notice['job_id']}",
            detector="injob_discovered",
            severity=INFO,
            summary=f"InJob started; monitoring active for Slurm job array {notice['job_id']}.",
        )
        # Slack delivery has its own durable outbox; no webhook duplicate.
        if "webhook" not in notice["sent"]:
            enqueue_start(config, key, chain, snapshot, notice)
            notice["sent"].append("webhook")
            save()
        for sink in sink_list:
            if sink.name in notice["sent"]:
                continue
            if sink.emit(finding):
                notice["sent"].append(sink.name)
                save()
