"""Stalls recover by themselves, rounds can be archived, settings change in one place."""

from __future__ import annotations

import json
import os
import sqlite3
import time
from datetime import datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest
from typer.testing import CliRunner

from researchforge.kaggle import auto_run as ar
from researchforge.kaggle import dashboard_state, run_config
from researchforge.kaggle.experiment_loop import (
    OvernightExperimentLoop,
    archived_rounds,
    delete_round,
    load_rounds,
    recover_interrupted_round,
    set_archived,
)

SLUG = "comp"


@pytest.fixture(autouse=True)
def _isolated_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: home))
    for key in (
        "RESEARCHFORGE_OVERNIGHT_DIR",
        "RESEARCHFORGE_SCAN_ROOTS",
        "CLAUDECODE",
        "AI_AGENT",
        "CLAUDE_CODE_EXECPATH",
        "CURSOR_AGENT",
        "CURSOR_TRACE_ID",
        "RESEARCHFORGE_AUTO_RUN",
    ):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("RESEARCHFORGE_NO_DASHBOARD", "1")
    monkeypatch.setattr(ar.shutil, "which", lambda name: None)
    dashboard_state._LAUNCHD_CACHE.clear()
    dashboard_state._READY_CACHE.clear()
    return home


def _project(root: Path, rounds: list[tuple[int, float]] = ()) -> Path:
    d = root / f".overnight_{SLUG}"
    d.mkdir(parents=True)
    contract = {"objective": {"primary_metric": {"name": "score", "direction": "minimize"}}}
    (d / "overnight_setup.json").write_text(
        json.dumps(
            {
                "competition_slug": SLUG,
                "selected_gpu": "kaggle_gpu",
                "contract": contract,
                "time_budget_hours": 8,
                "max_experiments": 20,
            }
        )
    )
    (d / "pending_hypotheses.json").write_text(json.dumps([{"idea": "a"}]))
    loop = OvernightExperimentLoop.__new__(OvernightExperimentLoop)
    loop.db_path = d / "experiment_log.db"
    loop._init_database()
    with sqlite3.connect(loop.db_path) as c:
        for num, metric in rounds:
            c.execute(
                "INSERT INTO experiments (round_num, patch_description, patch_code, full_passed, "
                "full_metric, full_improved, started_at, completed_at) VALUES (?,?,'',1,?,1,?,?)",
                (num, f"r{num}", metric, "2026-11-06T10:00:00", "2026-11-06T10:30:00"),
            )
    return d


class TestStalls:
    def test_a_dead_rounds_lock_is_released_and_the_round_recorded(self, tmp_path: Path) -> None:
        d = _project(tmp_path, [(6, 22.24)])
        dead = {
            "pid": 99999999,
            "round_num": 7,
            "title": "neighbour transfer",
            "stage": "full",
            "parent_round_num": 6,
            "started_at": "2026-11-06T18:09:49",
        }
        (d / "execute_round.lock").write_text(json.dumps(dead))
        (d / "inflight_rounds.json").write_text(json.dumps({"7": dead}))
        assert ar.round_in_flight(d) is False  # recovered on the way
        assert not (d / "execute_round.lock").exists()
        assert json.loads((d / "inflight_rounds.json").read_text()) == {}
        r7 = load_rounds(d / "experiment_log.db")[-1]
        assert r7.round_num == 7 and r7.rejected_reason.startswith("interrupted: process 99999999")

        (d / "execute_round.lock").write_text(json.dumps({"pid": os.getpid(), "round_num": 8}))
        assert recover_interrupted_round(d) is None and ar.round_in_flight(d)

    def test_the_monitor_waits_on_the_wall_clock(self) -> None:
        from researchforge.kaggle.cli import _wait_until_due

        t0 = time.monotonic()
        _wait_until_due(datetime.now() - timedelta(hours=16))  # overdue: no wait at all
        _wait_until_due(datetime.now() + timedelta(seconds=0.3), step_seconds=0.05)
        assert time.monotonic() - t0 < 2

    def test_submissions_are_announced_when_listed_scored_or_failed(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from researchforge.kaggle import kernel_watcher
        from researchforge.kaggle.cli import SUBMISSION_POLL_MINUTES, _submission_check

        d = _project(tmp_path)
        kernel_watcher.WatcherConfig(competition_slug=SLUG, interval_hours=6).save(d)
        rows = [
            {"ref": "1", "status": "SubmissionStatus.COMPLETE", "publicscore": "22.25"},
            {"ref": "2", "status": "SubmissionStatus.ERROR", "publicscore": ""},
        ]
        monkeypatch.setattr(kernel_watcher, "fetch_submissions", lambda slug: (rows, ""))
        sent: list[tuple[str, str]] = []
        monkeypatch.setattr(
            kernel_watcher,
            "_notify_submission_event",
            lambda s, sub, a, event: sent.append((sub["ref"], event)),
        )
        monkeypatch.setattr(
            kernel_watcher,
            "_notify_submission_complete",
            lambda s, sub, a: sent.append((sub["ref"], "scored")),
        )
        _submission_check(SLUG, d)
        assert sent == [("1", "scored")]  # the first look leaves an old failure quiet
        rows.append({"ref": "3", "status": "SubmissionStatus.PENDING", "publicscore": ""})
        assert _submission_check(SLUG, d) == (
            "ok",
            "3 submissions listed, 0 newly complete, 1 still scoring",
        )
        _submission_check(SLUG, d)
        rows.append({"ref": "4", "status": "SubmissionStatus.PENDING", "publicscore": ""})
        rows[2]["status"] = "SubmissionStatus.COMPLETE"
        rows[3]["status"] = "SubmissionStatus.ERROR"  # failed before we saw it scoring
        _submission_check(SLUG, d)
        _submission_check(SLUG, d)
        assert sent == [("1", "scored"), ("3", "listed"), ("3", "scored"), ("4", "failed")]
        assert SUBMISSION_POLL_MINUTES <= 15

    def test_headless_claude_gets_a_command_limit_as_long_as_a_launch(self, tmp_path: Path) -> None:
        d = _project(tmp_path)
        ar.AutoRunConfig(driver="claude", max_minutes=240).save(d)
        env = ar.child_env(d, "claude")
        assert env["BASH_DEFAULT_TIMEOUT_MS"] == env["BASH_MAX_TIMEOUT_MS"] == str(240 * 60 * 1000)
        prompt = ar.agent_prompt(SLUG, 2)
        assert "foreground" in prompt and f"pull-kernel {SLUG}" in prompt


class TestArchive:
    def test_archived_rounds_leave_history_views_and_best(self, tmp_path: Path) -> None:
        d = _project(tmp_path, [(1, 22.30), (2, 22.10), (3, 22.20)])
        set_archived(d, 2, True, "bad merge")
        assert archived_rounds(d) == {2}
        assert [r.round_num for r in load_rounds(d / "experiment_log.db")] == [1, 3]
        assert len(load_rounds(d / "experiment_log.db", include_archived=True)) == 3
        p = next(p for p in dashboard_state.build_state(tmp_path)["projects"])
        assert [e["id"] for e in p["experiments"]] == ["R1", "R3"]
        assert [e["id"] for e in p["overnight"]["archived_experiments"]] == ["R2"]
        set_archived(d, 2, False)
        assert archived_rounds(d) == set()
        assert delete_round(d, 3) and not delete_round(d, 3)
        assert [r.round_num for r in load_rounds(d / "experiment_log.db")] == [1, 2]

    def test_only_a_person_deletes_history(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from researchforge.cli import app

        d = _project(tmp_path, [(1, 22.3)])
        monkeypatch.chdir(tmp_path)
        assert CliRunner().invoke(app, ["kaggle", "archive", SLUG, "R1"]).exit_code == 0
        assert archived_rounds(d) == {1}
        monkeypatch.setenv("RESEARCHFORGE_AUTO_RUN", "1")
        out = CliRunner().invoke(app, ["kaggle", "delete-round", SLUG, "1", "--yes"])
        assert out.exit_code == 1 and "may not delete" in out.output


class TestRunConfig:
    def test_each_field_changes_in_place_and_is_logged(self, tmp_path: Path) -> None:
        d = _project(tmp_path)
        run_config.set_field(d, "compute", "local", via="dashboard")
        run_config.set_field(d, "kaggle_gpu_daily_cap", "3", via="dashboard")
        run_config.set_field(d, "metric_direction", "maximize", via="dashboard")
        run_config.set_field(d, "time_budget_hours", "12", via="dashboard")
        run_config.set_field(d, "notification_level", "minimal", via="dashboard")
        run_config.set_field(d, "answer:external_data", "not allowed", via="dashboard")
        run_config.set_field(d, "auto_run_rounds_per_launch", "50", via="dashboard")
        view = run_config.config_view(d)
        assert view["compute"]["value"] == "cpu" and view["kaggle_gpu_daily_cap"] == 3
        assert view["metric"]["direction"] == "maximize" and view["time_budget_hours"] == 12
        assert view["notification_level"] == "minimal"
        assert {"key": "external_data", "question": "", "value": "not allowed"} in view["answers"]
        assert view["auto_run"]["rounds_per_launch"] == 20  # clamped
        changes = view["changes"]
        assert changes[0]["field"] == "auto_run_rounds_per_launch" and changes[0]["to"] == 20
        assert {c["field"] for c in changes} >= {
            "compute",
            "metric_direction",
            "answer:external_data",
        }
        assert all(c["via"] == "dashboard" for c in changes)

    def test_refusals_keep_the_old_value(self, tmp_path: Path) -> None:
        d = _project(tmp_path)
        for field, value in (
            ("time_budget_hours", "999"),
            ("metric_direction", "up"),
            ("compute", "a100"),
            ("auto_run_loop", "repo"),
            ("baseline", "1"),
        ):
            with pytest.raises(ValueError):
                run_config.set_field(d, field, value)
        assert run_config.change_log(d) == []

    def test_the_cli_logs_through_the_same_setters(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from researchforge.cli import app

        d = _project(tmp_path)
        monkeypatch.chdir(tmp_path)
        assert CliRunner().invoke(app, ["kaggle", "compute", SLUG, "colab"]).exit_code == 0
        (entry,) = run_config.change_log(d)
        assert (entry["field"], entry["from"], entry["to"], entry["via"]) == (
            "compute",
            "kaggle_gpu",
            "colab_gpu",
            "terminal",
        )


class TestOneEngine:
    def test_a_repo_project_beside_never_becomes_a_second_loop(self, tmp_path: Path) -> None:
        d = _project(tmp_path)
        repo = tmp_path / ".researchforge"
        repo.mkdir()
        with sqlite3.connect(repo / "researchforge.db") as c:
            c.execute("CREATE TABLE experiment_plans (plan_id TEXT, status TEXT, created_at TEXT)")
            c.execute("INSERT INTO experiment_plans VALUES ('plan-004', 'planned', '2026-11-07')")
        (d / "auto_run.json").write_text(json.dumps({"driver": "claude", "loop": "repo"}))
        cfg = ar.AutoRunConfig.load(d)  # an old saved "repo" loop is ignored
        assert not any("approval" in g for g in ar.gates(d, cfg))
        cmd = ar.build_command(d, SLUG, cfg)
        assert "Bash(researchforge kaggle:*)" in cmd and "Bash(researchforge:*)" not in cmd
        prompt = cmd[2]
        assert "prepare-round" in prompt and "researchforge experiment" not in prompt
        assert "Never ask the person to run" in prompt and "skip-idea" in prompt
        with pytest.raises(ValueError):
            run_config.set_field(d, "auto_run_loop", "repo")


def test_pull_kernel_gets_source_outputs_and_lists_datasets(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import subprocess

    from researchforge.cli import app
    from researchforge.kaggle import api

    _project(tmp_path)
    monkeypatch.chdir(tmp_path)

    def fake_pull(ref: str, out: Path) -> Path:
        out.mkdir(parents=True, exist_ok=True)
        (out / "nb.ipynb").write_text("{}")
        (out / "kernel-metadata.json").write_text(json.dumps({"dataset_sources": ["a/data"]}))
        return out

    def fake_run(cmd: list[str], **kw: object) -> SimpleNamespace:
        Path(cmd[cmd.index("-p") + 1], "submission.csv").write_text("x")
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(api, "pull_kernel_source", fake_pull)
    monkeypatch.setattr(subprocess, "run", fake_run)
    out = CliRunner().invoke(app, ["kaggle", "pull-kernel", SLUG, "owner/nb-one"])
    assert out.exit_code == 0, out.output
    dest = tmp_path / "reference_kernels" / "nb-one"
    assert (dest / "nb.ipynb").exists() and (dest / "output" / "submission.csv").exists()
    assert "Datasets attached: a/data" in out.output
    assert "propose-from-kernel attaches all of these" in out.output
