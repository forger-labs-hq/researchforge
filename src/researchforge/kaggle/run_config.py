"""A project's run configuration: one place to read it, one setter per field.

Everything setup asked — where rounds run, the metric, budgets, what the person
knows — plus what was decided since (driver, auto-run, notifications) is read by
:func:`config_view` and changed through :func:`set_field`. The CLI commands
(``kaggle compute``, ``settings``, ``driver``, ``auto-run``) and the dashboard's
Run configs view call the same setters, so a change made anywhere lands in the
same files and in ``config_changes.jsonl`` with when, what and from where.

Changes apply from the next round, launch or proposal; a round already running
keeps the settings it started with.
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any

from researchforge.kaggle.workspace import atomic_write_json, read_json

CHANGES_FILE = "config_changes.jsonl"
EDITABLE = (
    "compute",
    "kaggle_gpu_daily_cap",
    "metric_direction",
    "time_budget_hours",
    "max_experiments",
    "notification_level",
    "submission_policy",
    "ntfy_topic",
    "ntfy_machine_topic",
    "round_time_limit_minutes",
)
# How long one round's full run may take; the old fixed limit was 30 minutes.
DEFAULT_ROUND_MINUTES = 30
ROUND_MINUTES_RANGE = (5, 1440)
# What each notification level sends, in words, for Run configs.
ALERT_LABELS = {
    "result": "every round's result",
    "improvement": "a round that beats the best score",
    "ideas": "new ideas in the pool",
    "kernel": "a better public notebook",
    "merge": "two rounds worth merging",
    "candidate": "a round waiting for your approval to submit",
    "interaction": "questions and replies waiting for you",
    "briefing": "the morning briefing",
    "alert": "failures that need you",
}


def change_via() -> str:
    """Where a change is being made from, for the change log."""
    if os.environ.get("RESEARCHFORGE_AUTO_RUN") == "1":
        return "auto-run"
    from researchforge.kaggle.auto_run import detect_host

    host = detect_host()[0]
    return {"claude": "Claude Code chat", "cursor": "Cursor chat"}.get(host, "terminal")


def record_change(d: Path, field: str, before: Any, after: Any, via: str = "") -> None:
    if before == after:
        return
    line = {
        "at": datetime.now().isoformat(timespec="seconds"),
        "field": field,
        "from": before,
        "to": after,
        "via": via or change_via(),
    }
    with (d / CHANGES_FILE).open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(line) + "\n")


def change_log(d: Path, limit: int = 30) -> list[dict[str, Any]]:
    path = d / CHANGES_FILE
    if not path.is_file():
        return []
    out = []
    for raw in path.read_text(encoding="utf-8").splitlines()[-limit:]:
        try:
            out.append(json.loads(raw))
        except ValueError:
            continue
    return list(reversed(out))


def _setup(d: Path) -> dict[str, Any]:
    setup = read_json(d / "overnight_setup.json", None)
    if not isinstance(setup, dict):
        raise ValueError("this project has no overnight setup")
    return setup


# ---------------------------------------------------------------------------
# Setters (each returns the previous value and records the change)
# ---------------------------------------------------------------------------


def set_compute(d: Path, target: str, via: str = "") -> tuple[str, str]:
    """Switch where rounds run. Refuses what the data policy forbids, and ssh /
    cloud until they are configured."""
    from researchforge.kaggle.compute_policy import ComputePolicy
    from researchforge.kaggle.orchestrator import OvernightError, OvernightOrchestrator

    slug = d.name.removeprefix(".overnight_")
    orch = OvernightOrchestrator.__new__(OvernightOrchestrator)
    orch.slug, orch.artifacts_dir, orch.data_dir = slug, d, d / "data"
    try:
        backend = orch.forced_compute(target)
    except OvernightError as exc:
        raise ValueError(str(exc)) from None
    if backend is None:
        raise ValueError("name a compute target")
    policy = ComputePolicy.load(d)
    if backend == "cpu" and policy is not None and not policy.data.can_download_locally:
        raise ValueError(
            f"the data policy keeps data at '{policy.data.location}'; running on this "
            "machine would download it — pick a remote target"
        )
    setup = _setup(d)
    before = str(setup.get("selected_gpu") or "cpu")
    setup["selected_gpu"] = str(backend)
    atomic_write_json(d / "overnight_setup.json", setup)
    record_change(d, "compute", before, str(backend), via)
    return before, str(backend)


def set_metric_direction(d: Path, direction: str, via: str = "") -> tuple[str, str]:
    """Point every reader of the direction the same way: the loop reads the setup's
    contract, the dashboard and prompts read competition_metadata.json."""
    from researchforge.kaggle.hypothesis_pool import setup_direction

    direction = direction.strip().lower()
    if direction not in ("maximize", "minimize"):
        raise ValueError("metric direction must be maximize or minimize")
    before = setup_direction(d)
    setup = _setup(d)
    metric = (setup.get("contract") or {}).get("objective", {}).get("primary_metric")
    if isinstance(metric, dict):
        metric["direction"] = direction
        atomic_write_json(d / "overnight_setup.json", setup)
    meta = read_json(d / "competition_metadata.json", None)
    if isinstance(meta, dict):
        meta["metric_direction"] = direction
        meta["metric_direction_source"] = "user"
        atomic_write_json(d / "competition_metadata.json", meta)
    record_change(d, "metric_direction", before, direction, via)
    return before, direction


def set_setup_number(d: Path, field: str, value: Any, via: str = "") -> tuple[Any, Any]:
    limits = {"time_budget_hours": (1, 168), "max_experiments": (1, 1000)}
    if field not in limits:
        raise ValueError(f"unknown field {field}")
    try:
        number = int(value)
    except (TypeError, ValueError):
        raise ValueError(f"{field} must be a whole number") from None
    lo, hi = limits[field]
    if not lo <= number <= hi:
        raise ValueError(f"{field} must be between {lo} and {hi}")
    setup = _setup(d)
    before = setup.get(field)
    setup[field] = number
    atomic_write_json(d / "overnight_setup.json", setup)
    record_change(d, field, before, number, via)
    return before, number


def round_time_limit_seconds(d: Path) -> float:
    """How long one round's full run may take on this project's compute."""
    setup = read_json(d / "overnight_setup.json", {}) or {}
    try:
        minutes = int(setup.get("round_time_limit_minutes") or DEFAULT_ROUND_MINUTES)
    except (TypeError, ValueError):
        minutes = DEFAULT_ROUND_MINUTES
    lo, hi = ROUND_MINUTES_RANGE
    return float(min(max(minutes, lo), hi) * 60)


def set_round_time_limit(d: Path, value: Any, via: str = "") -> tuple[Any, int]:
    try:
        minutes = int(value)
    except (TypeError, ValueError):
        raise ValueError("the round time limit must be a whole number of minutes") from None
    lo, hi = ROUND_MINUTES_RANGE
    if not lo <= minutes <= hi:
        raise ValueError(f"the round time limit must be between {lo} and {hi} minutes")
    setup = _setup(d)
    before = setup.get("round_time_limit_minutes") or DEFAULT_ROUND_MINUTES
    setup["round_time_limit_minutes"] = minutes
    atomic_write_json(d / "overnight_setup.json", setup)
    record_change(d, "round_time_limit_minutes", before, minutes, via)
    return before, minutes


def set_machine_topic(d: Path, topic: str, via: str = "") -> tuple[str, str]:
    """The ntfy topic for every project without its own (``researchforge notify setup``)."""
    from researchforge.notifications.service import _load_notify_config, save_notify_config

    before = str(_load_notify_config().get("ntfy_topic") or "")
    save_notify_config(topic=topic.strip(), clear_topic=not topic.strip())
    record_change(d, "ntfy_machine_topic", before, topic.strip(), via)
    return before, topic.strip()


def set_ntfy_topic(d: Path, topic: str, via: str = "") -> tuple[str, str]:
    setup = _setup(d)
    before = str(setup.get("ntfy_topic") or "")
    setup["ntfy_topic"] = topic.strip()
    atomic_write_json(d / "overnight_setup.json", setup)
    record_change(d, "ntfy_topic", before, topic.strip(), via)
    return before, topic.strip()


def set_control(d: Path, field: str, value: Any, via: str = "") -> tuple[Any, Any]:
    from researchforge.kaggle import control

    ctl = control.LoopControl.load(d)
    if field == "notification_level":
        if value not in control.NOTIFICATION_LEVELS:
            raise ValueError(f"notification level must be one of {control.NOTIFICATION_LEVELS}")
    elif field == "submission_policy":
        if value not in control.SUBMISSION_POLICIES:
            raise ValueError(f"submission policy must be one of {control.SUBMISSION_POLICIES}")
    elif field == "kaggle_gpu_daily_cap":
        try:
            value = int(value)
        except (TypeError, ValueError):
            raise ValueError("the Kaggle GPU cap must be a whole number (0 = no cap)") from None
        if value < 0:
            raise ValueError("the Kaggle GPU cap must be 0 or more")
    else:
        raise ValueError(f"unknown field {field}")
    before = getattr(ctl, field)
    setattr(ctl, field, value)
    ctl.save(d)
    record_change(d, field, before, value, via)
    return before, value


def set_answer(d: Path, key: str, value: str, via: str = "") -> tuple[str, str]:
    """Change something the person told setup; proposals read it from both copies."""
    from researchforge.kaggle.human_questions import UserAnswers

    answers = UserAnswers.load(d)
    before = str(answers.answers.get(key, ""))
    answers.answers[key] = value
    answers.save(d)
    setup = read_json(d / "overnight_setup.json", None)
    if isinstance(setup, dict):
        ua = setup.get("user_answers") or {"answers": {}, "asked_at": ""}
        ua.setdefault("answers", {})[key] = value
        setup["user_answers"] = ua
        atomic_write_json(d / "overnight_setup.json", setup)
    record_change(d, f"answer:{key}", before, value, via)
    return before, value


def set_auto_run(d: Path, field: str, value: Any, via: str = "") -> tuple[Any, Any]:
    """Driver and auto-run limits (auto_run.json)."""
    from researchforge.kaggle import auto_run as ar

    if field == "driver":
        before = ar.AutoRunConfig.load(d).driver
        ar.set_driver(d, str(value))  # records its own change
        return before, str(value)
    cfg = ar.AutoRunConfig.load(d)
    key = field.removeprefix("auto_run_")
    if key in ("rounds_per_launch", "max_rounds_per_day", "max_minutes"):
        try:
            value = int(value)
        except (TypeError, ValueError):
            raise ValueError(f"{key} must be a whole number") from None
    elif key == "back_to_back":
        value = (
            value if isinstance(value, bool) else str(value).lower() in ("1", "true", "yes", "on")
        )
    else:
        raise ValueError(f"unknown field {field}")
    before = getattr(cfg, key)
    setattr(cfg, key, value)
    cfg.save(d)
    after = getattr(ar.AutoRunConfig.load(d), key)  # clamped
    record_change(d, field, before, after, via)
    return before, after


def set_colab(d: Path, field: str, value: Any, via: str = "") -> tuple[Any, Any]:
    from researchforge.kaggle import colab

    before = colab.ColabConfig.load(d)
    key = field.removeprefix("colab_")
    if key == "plan":
        colab.set_config(d, plan=str(value), via=via)
    elif key == "gpu":
        colab.set_config(d, gpu=str(value), via=via)
    elif key == "high_mem":
        colab.set_config(d, high_mem=str(value).lower() in ("1", "true", "yes", "on"), via=via)
    else:
        raise ValueError(f"unknown field {field}")
    return getattr(before, key), getattr(colab.ColabConfig.load(d), key)


def set_field(d: Path, field: str, value: Any, via: str = "") -> tuple[Any, Any]:
    """Change one field by name (the dashboard's single entry point)."""
    if field.startswith("colab_"):
        return set_colab(d, field, value, via)
    if field == "driver" or field.startswith("auto_run_"):
        return set_auto_run(d, field, value, via)
    if field == "compute":
        return set_compute(d, str(value), via)
    if field == "metric_direction":
        return set_metric_direction(d, str(value), via)
    if field in ("time_budget_hours", "max_experiments"):
        return set_setup_number(d, field, value, via)
    if field == "ntfy_topic":
        return set_ntfy_topic(d, str(value), via)
    if field == "ntfy_machine_topic":
        return set_machine_topic(d, str(value), via)
    if field == "round_time_limit_minutes":
        return set_round_time_limit(d, value, via)
    if field in ("notification_level", "submission_policy", "kaggle_gpu_daily_cap"):
        return set_control(d, field, value, via)
    if field.startswith("answer:"):
        return set_answer(d, field.removeprefix("answer:"), str(value), via)
    raise ValueError(f"{field} cannot be changed here")


# ---------------------------------------------------------------------------
# View
# ---------------------------------------------------------------------------


def config_view(d: Path) -> dict[str, Any]:
    """Everything that shapes how this project runs, for the Run configs view."""
    from researchforge.kaggle import control
    from researchforge.kaggle.compute_policy import ComputePolicy
    from researchforge.kaggle.gpu_detector import COMPUTE_CHOICES
    from researchforge.kaggle.human_questions import UserAnswers

    setup = read_json(d / "overnight_setup.json", {}) or {}
    meta = read_json(d / "competition_metadata.json", {}) or {}
    metric = ((setup.get("contract") or {}).get("objective") or {}).get("primary_metric") or {}
    ctl = control.LoopControl.load(d)
    policy = ComputePolicy.load(d)
    questions = {
        str(q.get("id")): str(q.get("text") or "")
        for q in (read_json(d / "mid_loop_questions.json", []) or [])
        if isinstance(q, dict) and q.get("id")
    }
    answers = UserAnswers.load(d).answers
    from researchforge.kaggle import auto_run as ar

    auto = ar.AutoRunConfig.load(d)
    from researchforge.kaggle import colab

    ccfg = colab.ColabConfig.load(d)
    on_colab = str(setup.get("selected_gpu") or "") == "colab_gpu"
    return {
        "colab": {
            "plan": ccfg.plan,
            "gpu": ccfg.gpu,
            "high_mem": ccfg.high_mem,
            "plans": [[k, colab.PLAN_LABELS[k]] for k in colab.PLANS],
            "plan_gpus": {k: list(v) for k, v in colab.PLAN_GPUS.items()},
            "high_mem_plans": sorted(colab.PLAN_HIGH_MEM),
            "cli": colab.cached("status") if on_colab else colab.status(check_login=False),
            "usage": colab.cached("usage") if on_colab else {},
        },
        "driver": auto.driver,
        "auto_run": {
            "rounds_per_launch": auto.rounds_per_launch,
            "max_rounds_per_day": auto.max_rounds_per_day,
            "max_minutes": auto.max_minutes,
            "back_to_back": auto.back_to_back,
        },
        "compute": {
            "value": str(setup.get("selected_gpu") or "cpu"),
            "choices": [c.strip() for c in COMPUTE_CHOICES.split(",")],
            "data_location": policy.data.location if policy is not None else "",
            "local_allowed": policy.data.can_download_locally if policy is not None else True,
        },
        "kaggle_gpu_daily_cap": ctl.kaggle_gpu_daily_cap,
        "metric": {
            "name": str(metric.get("name") or meta.get("metric_name") or ""),
            "direction": str(metric.get("direction") or meta.get("metric_direction") or ""),
            "source": str(meta.get("metric_direction_source") or ""),
        },
        "time_budget_hours": setup.get("time_budget_hours"),
        "max_experiments": setup.get("max_experiments"),
        "notification_level": ctl.notification_level,
        "notification_levels": list(control.NOTIFICATION_LEVELS),
        "submission_policy": ctl.submission_policy,
        "submission_policies": list(control.SUBMISSION_POLICIES),
        "ntfy_topic": str(setup.get("ntfy_topic") or ""),
        "notify": notify_view(d),
        "round_time_limit_minutes": int(round_time_limit_seconds(d) // 60),
        "answers": [
            {"key": k, "question": questions.get(k, ""), "value": str(v)}
            for k, v in sorted(answers.items())
        ],
        "baseline": (setup.get("baseline_result") or {}).get("cv_score"),
        "baseline_kernel": (setup.get("baseline_result") or {}).get("kernel_ref") or "",
        "changes": change_log(d),
    }


def notify_view(d: Path) -> dict[str, Any]:
    """Where this project's alerts go and what each level sends, for Run configs."""
    from researchforge.kaggle import control
    from researchforge.notifications.service import _load_notify_config, _ntfy_server

    topic, where = control.notify_topic(d)
    server = _ntfy_server()
    return {
        "topic": topic,
        "where": where,
        "machine_topic": str(_load_notify_config().get("ntfy_topic") or ""),
        "server": server,
        "subscribe_url": f"{server}/{topic}" if topic else "",
        "level_sends": {
            level: [ALERT_LABELS[c] for c in ALERT_LABELS if c in allowed]
            for level, allowed in control._LEVEL_ALLOWS.items()
        },
    }


def send_test_alert(d: Path) -> dict[str, Any]:
    """Send one alert to this project's topic now, whatever the level, to check the phone."""
    from researchforge.kaggle import control
    from researchforge.notifications.service import Channel, Notification, send_notification

    topic, where = control.notify_topic(d)
    if not topic:
        raise ValueError("no ntfy topic is set — add one first")
    slug = d.name.removeprefix(".overnight_")
    old = os.environ.get("NTFY_TOPIC")
    os.environ["NTFY_TOPIC"] = topic
    try:
        sent = send_notification(
            Notification(
                title=f"ResearchForge: {slug}",
                message="Test alert from Run configs. Questions, approvals and submission "
                "offers for this competition will arrive here.",
                channel=Channel.NTFY,
                tags=["bell"],
            )
        )
    finally:
        if old is None:
            os.environ.pop("NTFY_TOPIC", None)
        else:
            os.environ["NTFY_TOPIC"] = old
    if not sent:
        raise ValueError(f"ntfy did not accept the alert for topic {topic} — check the internet")
    return {"sent": True, "topic": topic, "where": where}
