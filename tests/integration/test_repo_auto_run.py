"""Unattended repo auto-run against examples/simple-python, end to end.

Only the AI is replaced (the same scripted planner as test_autorun_example). The
launch goes through the real engine, importer, worktree executor and ranking:
with ``ask`` a plan stops the launch and waits for the person; once approved the
next launch runs it; a standing approval runs rounds like ``autorun --yes``.
"""

import subprocess
from contextlib import closing
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import pytest
from test_autorun_example import ScriptedAI, example_loop  # noqa: F401
from typer.testing import CliRunner

from researchforge.autorun import unattended as ua
from researchforge.storage.db import open_project_db
from researchforge.storage.experiment_repository import list_experiments


class _NoAI:
    name = "scripted"

    def generate(self, system: str, user: str, *, max_tokens: int = 8192) -> str:
        raise AssertionError("the scripted planner replaces every AI call")


@pytest.fixture
def repo_auto(
    example_loop: ScriptedAI,  # noqa: F811
    isolated_project_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> Path:
    monkeypatch.setattr(ua, "driver_provider", lambda root, cfg: _NoAI())
    ua.update_settings(isolated_project_dir, {"enabled": True, "driver": "byok"})
    return isolated_project_dir


def _statuses() -> dict[str, str]:
    with closing(open_project_db()) as conn:
        return {e.experiment_id: e.status.value for e in list_experiments(conn)}


class TestAskEveryPlan:
    def test_a_plan_waits_for_the_person_and_the_next_launch_runs_it(
        self, repo_auto: Path, cli_runner: CliRunner
    ) -> None:
        from researchforge.cli import app

        first = ua.run_launch(repo_auto)
        assert first["rounds_run"] == 0
        assert first["plans_waiting"] == ["plan-001"]
        assert _statuses() == {"exp-001": "planned"}  # nothing ran without the person
        cfg = ua.RepoAutoRunConfig.load(repo_auto)
        assert any("plan-001 waits for your approval" in g for g in ua.gates(repo_auto, cfg))

        approved = cli_runner.invoke(app, ["experiment", "approve", "plan-001", "--yes"])
        assert approved.exit_code == 0, approved.output
        assert ua.plans_approved(repo_auto) == ["plan-001"]
        assert not any("waits" in g for g in ua.gates(repo_auto, cfg))

        second = ua.run_launch(repo_auto)
        assert second["plans_run"] == ["plan-001"]
        assert second["rounds_run"] == 1
        assert _statuses()["exp-001"] != "approved"  # measured, not left waiting
        assert "best f1" in second["best"]

    def test_the_child_records_the_launch_and_counts_its_rounds(self, repo_auto: Path) -> None:
        ua.set_approval(repo_auto, "standing", typed="approve")
        assert ua.run_child(repo_auto) == 0
        state = ua._kaggle.load_state(ua.rf_dir(repo_auto))
        assert state["current"] is None
        assert state["last"]["status"] == "finished"
        assert state["last"]["rounds_run"] == 1
        assert ua.rounds_today(repo_auto) == 1


class TestStandingApproval:
    def test_rounds_run_without_asking_and_compound(
        self,
        repo_auto: Path,
        example_loop: ScriptedAI,  # noqa: F811
    ) -> None:
        ua.update_settings(repo_auto, {"rounds_per_launch": 2})
        ua.set_approval(repo_auto, "standing", typed="approve")
        outcome = ua.run_launch(repo_auto)
        assert outcome["rounds_run"] == 2
        assert outcome["plans_waiting"] == []
        assert example_loop.parents_seen == [None, "exp-001"]  # round 2 builds on round 1
        assert "1.0000" in outcome["best"]

    def test_the_daily_limit_stops_launches(self, repo_auto: Path) -> None:
        ua.update_settings(repo_auto, {"max_rounds_per_day": 1})
        ua.set_approval(repo_auto, "standing", typed="approve")
        assert ua.run_child(repo_auto) == 0
        reasons = ua.gates(repo_auto, ua.RepoAutoRunConfig.load(repo_auto))
        assert any("daily limit of 1" in r for r in reasons)


class TestGatesAndMonitor:
    def test_ready_project_has_clear_gates_until_paused(self, repo_auto: Path) -> None:
        from researchforge.kaggle import control

        cfg = ua.RepoAutoRunConfig.load(repo_auto)
        assert ua.gates(repo_auto, cfg) == []
        control.pause(ua.rf_dir(repo_auto), "testing")
        assert ua.gates(repo_auto, cfg) == ["the loop is paused"]

    def test_a_usage_limit_holds_launches_until_it_resets(self, repo_auto: Path) -> None:
        d = ua.rf_dir(repo_auto)
        state = ua._kaggle.load_state(d)
        until = (datetime.now() + timedelta(hours=2)).isoformat(timespec="minutes")
        state["usage_limit"] = {"driver": "byok", "message": "limit", "until": until}
        ua._kaggle._save_state(d, state)
        reasons = ua.gates(repo_auto, ua.RepoAutoRunConfig.load(repo_auto))
        assert any("usage limit" in r for r in reasons)

    def test_monitor_cycle_starts_a_launch_and_records_each_check(
        self, repo_auto: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        started: list[Any] = []
        monkeypatch.setattr(ua, "readiness", lambda cfg, **kw: ua._kaggle.Readiness(True, "ok"))
        monkeypatch.setattr(
            ua, "launch", lambda root, cfg: started.append(root) or {"pid": 1, "driver": "byok"}
        )
        ua.update_settings(repo_auto, {"papers_hours": 0, "briefing_at": ""})
        state = ua.monitor_cycle(repo_auto)
        assert started == [repo_auto]
        assert state["sensors"]["auto_run"]["status"] == "ok"
        assert state["sensors"]["papers"]["status"] == "skipped"
        assert state["next_cycle_at"] > state["last_cycle_at"]

    def test_status_and_gates_never_write_to_the_project(self, repo_auto: Path) -> None:
        d = ua.rf_dir(repo_auto)
        log = d / "research-log.md"
        log.unlink(missing_ok=True)
        db = d / "researchforge.db"
        before = (db.stat().st_mtime_ns, db.read_bytes())
        ua.status(repo_auto)
        ua.gates(repo_auto, ua.RepoAutoRunConfig.load(repo_auto))
        assert (db.stat().st_mtime_ns, db.read_bytes()) == before  # an idle project stays idle
        assert not log.exists()

    def test_status_carries_what_the_dashboard_shows(self, repo_auto: Path) -> None:
        st = ua.status(repo_auto)
        assert st["enabled"] is True
        assert set(st["drivers"]) == {"claude", "cursor", "byok"}
        assert [s["name"] for s in st["monitor"]["sensors"]] == [n for n, _ in ua.SENSORS]
        assert st["service"]["label"].startswith("com.researchforge.repo.")


def test_cli_status_and_settings(repo_auto: Path, cli_runner: CliRunner) -> None:
    from researchforge.cli import app

    out = cli_runner.invoke(app, ["auto-run", "settings", "--daily-max", "3", "--no-merge"])
    assert out.exit_code == 0, out.output
    cfg = ua.RepoAutoRunConfig.load(repo_auto)
    assert (cfg.max_rounds_per_day, cfg.merge) == (3, False)
    out = cli_runner.invoke(app, ["auto-run", "status"])
    assert "auto-run on" in out.output
    # A standing approval needs a person at a terminal typing it.
    out = cli_runner.invoke(app, ["auto-run", "approval", "standing"], input="approve\n")
    assert out.exit_code == 1
    assert not ua.RepoAutoRunConfig.load(repo_auto).standing


def test_the_checkout_is_untouched_by_a_launch(repo_auto: Path) -> None:
    def porcelain() -> str:
        return subprocess.run(
            ["git", "status", "--porcelain"], cwd=repo_auto / "demo", capture_output=True, text=True
        ).stdout

    before = porcelain()
    ua.set_approval(repo_auto, "standing", typed="approve")
    ua.run_launch(repo_auto)
    assert porcelain() == before  # experiments ran in worktrees, never in the checkout


class TestWhatTheReadmePromises:
    """Each check, alert and sensor the README lists for repo auto-run."""

    def test_a_question_a_running_launch_and_a_reached_target_each_hold_launches(
        self, repo_auto: Path
    ) -> None:
        from researchforge.kaggle import control

        d, cfg = ua.rf_dir(repo_auto), ua.RepoAutoRunConfig.load(repo_auto)
        item = control.ask(d, "Keep tuning the tokenizer?", options=["yes", "no"], holds_loop=True)
        assert any("put the loop on hold" in g for g in ua.gates(repo_auto, cfg))
        control.respond(d, item.id, "yes")
        assert ua.gates(repo_auto, cfg) == []

        with subprocess.Popen(["sleep", "60"]) as proc:
            state = ua._kaggle.load_state(d)
            state["current"] = {"pid": proc.pid, "driver": "byok", "started_at": "2026-01-01T00:00"}
            ua._kaggle._save_state(d, state)
            assert "a launch is already running" in ua.gates(repo_auto, cfg)
            proc.kill()

        state = ua._kaggle.load_state(d)
        state["current"] = None
        state["last"] = {"stop_reason": "objective achieved: f1 0.95 >= 0.9"}
        ua._kaggle._save_state(d, state)
        assert any("the target is reached" in g for g in ua.gates(repo_auto, cfg))

    def test_each_plan_left_waiting_reaches_the_phone(
        self, repo_auto: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from researchforge.notifications import service

        sent: list[Any] = []
        monkeypatch.setattr(service, "send_notification", lambda n: sent.append(n) or True)
        ua.update_settings(repo_auto, {"ntfy_topic": "rf-test-topic"})
        ua.run_launch(repo_auto)
        asks = [n for n in sent if "plan-001 waits for your approval" in n.message]
        assert len(asks) == 1
        assert "researchforge experiment approve plan-001" in asks[0].message

    def test_the_monitor_stops_a_launch_past_its_time_limit(self, repo_auto: Path) -> None:
        d = ua.rf_dir(repo_auto)
        ua.update_settings(repo_auto, {"max_minutes": 60})
        cfg = ua.RepoAutoRunConfig.load(repo_auto)
        with subprocess.Popen(["sleep", "60"]) as proc:
            state = ua._kaggle.load_state(d)
            started = datetime.now() - timedelta(minutes=80)  # inside the 30-minute grace
            state["current"] = {
                "pid": proc.pid,
                "driver": "byok",
                "started_at": started.isoformat(),
            }
            ua._kaggle._save_state(d, state)
            assert ua._watch_launch(repo_auto, cfg)[0] == "ok"
            assert proc.poll() is None

            state["current"]["started_at"] = (started - timedelta(minutes=20)).isoformat()
            ua._kaggle._save_state(d, state)
            status, detail = ua._watch_launch(repo_auto, cfg)
            assert (status, detail[:13]) == ("error", "stopped after")
            assert proc.wait(timeout=10) != 0  # SIGTERM reached it

    def test_the_papers_sensor_searches_arxiv_and_counts_what_it_stored(
        self, repo_auto: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from researchforge.domain.paper import Paper
        from researchforge.storage.paper_repository import upsert_paper
        from researchforge.storage.project_repository import get_project

        calls: list[list[str]] = []

        def search(cmd: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
            calls.append(cmd)  # arXiv itself is the one thing not reached from a test
            with closing(open_project_db()) as conn:
                project = get_project(conn)
                assert project is not None
                upsert_paper(
                    conn,
                    project.id,
                    Paper(
                        paper_id="arxiv:2611.00001",
                        title="Normalizing tokens",
                        authors=["A. Author"],
                        published_at=datetime.now(),
                        abstract="Token normalization helps.",
                        source_url="https://arxiv.org/abs/2611.00001",
                    ),
                )
            return subprocess.CompletedProcess(cmd, 0, "[]", "")

        monkeypatch.setattr(ua.subprocess, "run", search)
        state: dict[str, Any] = {}
        cfg = ua.RepoAutoRunConfig.load(repo_auto)
        assert ua._papers(repo_auto, cfg, state)[1].startswith("1 new paper(s)")
        assert calls[0][3:6] == ["research", "search", "--since"]
        assert ua._papers(repo_auto, cfg, state) == ("skipped", "not due — every 24h")

    def test_the_daily_briefing_goes_out_once_a_day(
        self, repo_auto: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from researchforge.notifications import service

        sent: list[Any] = []
        monkeypatch.setattr(service, "send_notification", lambda n: sent.append(n) or True)
        ua.update_settings(repo_auto, {"ntfy_topic": "rf-test-topic", "briefing_at": "00:00"})
        cfg, state = ua.RepoAutoRunConfig.load(repo_auto), {}
        assert ua._briefing(repo_auto, cfg, state) == ("ok", "briefing sent")
        assert "best f1" in sent[0].message
        assert ua._briefing(repo_auto, cfg, state)[0] == "skipped"
        assert len(sent) == 1


def test_a_project_without_a_contract_is_not_launched(
    isolated_project_dir: Path, cli_runner: CliRunner
) -> None:
    from researchforge.cli import app

    made = cli_runner.invoke(
        app, ["project", "create", "--mode", "improve_repository", "--objective", "Faster tests"]
    )
    assert made.exit_code == 0, made.output
    ua.update_settings(isolated_project_dir, {"enabled": True})
    reasons = ua.gates(isolated_project_dir, ua.RepoAutoRunConfig.load(isolated_project_dir))
    assert any(r.startswith("the loop cannot run yet") for r in reasons)
