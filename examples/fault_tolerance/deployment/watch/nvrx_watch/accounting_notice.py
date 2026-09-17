# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""One durable accounting incident per user, with grace and recovery notification."""

from .types import INFO, WARNING, Finding

REMINDER_SECONDS = 6 * 3600
INCOMPLETE_CHECKS = 2


def notify(config, user, entry, sink_list, now, save):
    if config.dry_run:
        return
    error = entry.get("accounting_error")
    notice = entry.get("accounting_notice")
    if error:
        # Count actual accounting queries, not the intervening cron ticks.
        if entry.get("accounting_incomplete_checks", 0) < INCOMPLETE_CHECKS:
            return
        if notice is None:
            notice = entry["accounting_notice"] = {"sent": {}, "recovered": []}
            save()
        ids = entry.get("accounting_missing", [])
        finding = Finding(
            key=f"discovery-accounting-{user}",
            detector="observer",
            severity=WARNING,
            summary=f"Accounting incomplete for {user}: terminal state unavailable for "
            f"{len(ids)} Slurm job array(s) ({', '.join(ids[:12])}{', …' if len(ids) > 12 else ''}). "
            "Cycle monitoring continues; terminal-state checks are limited.",
            detail=error,
        )
        for sink in sink_list:
            last = notice["sent"].get(sink.name)
            if sink.name == "log":
                sink.emit(finding)
            elif (
                last is None
                or sink.name in notice["recovered"]
                or now - last >= max(config.alert_cooldown, REMINDER_SECONDS)
            ):
                if sink.emit(finding):
                    notice["sent"][sink.name] = now
                    if sink.name in notice["recovered"]:
                        notice["recovered"].remove(sink.name)
                    save()
    elif notice is not None:
        finding = Finding(
            key=f"discovery-accounting-recovered-{user}",
            detector="observer",
            severity=INFO,
            summary=f"Accounting restored for {user}; terminal-state checks are available again.",
        )
        for sink in sink_list:
            if sink.name in notice["sent"] and sink.name not in notice["recovered"]:
                if sink.emit(finding):
                    notice["recovered"].append(sink.name)
                    save()
        names = {s.name for s in sink_list}
        if set(notice["sent"]) & names <= set(notice["recovered"]):
            entry.pop("accounting_notice")
            save()
