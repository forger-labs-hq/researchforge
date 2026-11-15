"""Questions: asked mid-loop without stopping it, answered from the bell or Home."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest
from typer.testing import CliRunner

from researchforge.kaggle import control, dashboard_state
from researchforge.kaggle.dashboard import DashboardService

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
        "CURSOR_AGENT",
        "CURSOR_TRACE_ID",
        "RESEARCHFORGE_AUTO_RUN",
        "RESEARCHFORGE_DRIVER",
    ):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("RESEARCHFORGE_NO_DASHBOARD", "1")
    dashboard_state._LAUNCHD_CACHE.clear()
    return home


def _project(root: Path) -> Path:
    d = root / f".overnight_{SLUG}"
    d.mkdir(parents=True)
    (d / "overnight_setup.json").write_text(json.dumps({"competition_slug": SLUG}))
    pool = [{"idea": "use forum packings", "source": "kernel_watcher"}, {"idea": "tilted bands"}]
    (d / "pending_hypotheses.json").write_text(json.dumps(pool))
    return d


def test_ask_pushes_and_only_its_ideas_wait(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from researchforge.kaggle.hypothesis_pool import rank_untested

    d = _project(tmp_path)
    sent: list[tuple[str, str]] = []
    monkeypatch.setattr(control, "notify", lambda art, cat, msg, **k: sent.append((cat, msg)))
    item = control.ask(
        d,
        "Are forum packings allowed?",
        options=["yes", "no"],
        blocks=["use forum packings"],
        source="auto-run (claude)",
    )
    assert sent and sent[0][0] == "interaction" and item.id in sent[0][1]
    assert [h["idea"] for h in rank_untested(d, "maximize")] == ["tilted bands"]
    assert control.holding_questions(d) == []
    control.respond(d, item.id, "no")
    assert len(rank_untested(d, "maximize")) == 2  # answered: the idea runs again


def test_the_ask_command_names_who_asked(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from researchforge.cli import app

    d = _project(tmp_path)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(control, "notify", lambda *a, **k: True)
    monkeypatch.setenv("RESEARCHFORGE_AUTO_RUN", "1")
    monkeypatch.setenv("RESEARCHFORGE_DRIVER", "claude")
    out = CliRunner().invoke(
        app, ["kaggle", "ask", SLUG, "Which n range first?", "--options", "small,large", "--hold"]
    )
    assert out.exit_code == 0, out.output
    (q,) = control.open_interactions(d)
    assert (q.source, q.holds_loop, q.options) == ("auto-run (claude)", True, ["small", "large"])
    bad = CliRunner().invoke(
        app, ["kaggle", "ask", SLUG, "x?", "--options", "a,b", "--default", "c"]
    )
    assert bad.exit_code == 1


def test_questions_reach_the_dashboard_grouped_and_answerable(tmp_path: Path) -> None:
    d = _project(tmp_path)
    q = control.request_interaction(d, "Allowed?", options=["yes", "no"], source="setup")
    p = next(p for p in dashboard_state.build_state(tmp_path)["projects"])
    assert p["category"] == "kaggle"
    (shown,) = p["questions"]
    assert shown["id"] == q.id and shown["answerable"] and shown["options"] == ["yes", "no"]
    assert shown["command"] == f'researchforge kaggle respond {SLUG} {q.id} "<answer>"'

    result = DashboardService.action_answer(None, p, {"id": q.id, "answer": "no"})  # type: ignore[arg-type]
    assert result == {"answered": q.id, "answer": "no", "applied": []}  # no settings changed
    assert control.open_interactions(d) == []
    with pytest.raises(KeyError):
        DashboardService.action_answer(None, p, {"id": q.id, "answer": "no"})  # type: ignore[arg-type]


def test_only_plans_still_waiting_ask_for_a_typed_approval(tmp_path: Path) -> None:
    from researchforge.kaggle.dashboard_state import _plans_awaiting_approval

    repo = tmp_path / ".researchforge"
    repo.mkdir()
    with sqlite3.connect(repo / "researchforge.db") as c:
        c.execute("CREATE TABLE experiment_plans (plan_id TEXT, status TEXT, created_at TEXT)")
        c.executemany(
            "INSERT INTO experiment_plans VALUES (?,?,?)",
            [
                ("plan-001", "planned", "2026-11-01T10:00:00"),  # superseded by plan-002
                ("plan-002", "completed", "2026-11-01T10:00:05"),
                ("plan-003", "planned", "2026-11-02T09:00:00"),  # newest: really waiting
            ],
        )
    assert _plans_awaiting_approval(repo) == {"plan-003"}


def test_dismissing_closes_without_answering_and_frees_its_ideas(tmp_path: Path) -> None:
    from researchforge.kaggle.hypothesis_pool import rank_untested

    d = _project(tmp_path)
    q = control.request_interaction(d, "Pull the forks?", blocks=["use forum packings"])
    assert len(rank_untested(d, "maximize")) == 1
    p = next(p for p in dashboard_state.build_state(tmp_path)["projects"])
    assert DashboardService.action_answer(None, p, {"id": q.id, "dismiss": True}) == {  # type: ignore[arg-type]
        "dismissed": q.id
    }
    (item,) = control.load_interactions(d)
    assert item.status == "dismissed" and control.open_interactions(d) == []
    assert len(rank_untested(d, "maximize")) == 2
    with pytest.raises(ValueError):
        control.respond(d, q.id, "yes")  # a dismissed question cannot be answered later


def test_a_waiting_plan_is_approved_only_on_the_typed_word(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import subprocess
    from types import SimpleNamespace

    repo = tmp_path / ".researchforge"
    repo.mkdir()
    with sqlite3.connect(repo / "researchforge.db") as c:
        c.execute(
            "CREATE TABLE experiment_plans "
            "(plan_id TEXT, status TEXT, created_at TEXT, record TEXT)"
        )
        c.execute("INSERT INTO experiment_plans VALUES ('plan-9', 'planned', '2026-11-09', '{}')")
    project = {"root": str(tmp_path)}
    with pytest.raises(ValueError, match="approve"):
        DashboardService.action_plan_approve(None, project, {"plan": "plan-9", "typed": "yes"})  # type: ignore[arg-type]
    with pytest.raises(KeyError):
        DashboardService.action_plan_approve(None, project, {"plan": "plan-8", "typed": "approve"})  # type: ignore[arg-type]
    calls: list[list[str]] = []

    def fake_run(cmd: list[str], **kw: object) -> SimpleNamespace:
        calls.append(cmd)
        return SimpleNamespace(returncode=0, stdout="approved", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)
    out = DashboardService.action_plan_approve(
        None, project, {"plan": "plan-9", "typed": " Approve "}
    )  # type: ignore[arg-type]
    assert (out["plan"], out["decision"]) == ("plan-9", "approve") and calls[0][-4:] == [
        "experiment",
        "approve",
        "plan-9",
        "--yes",
    ]


def test_comments_travel_with_answers_and_plans_to_the_next_proposal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import subprocess
    from types import SimpleNamespace

    from researchforge.kaggle import auto_run as ar

    d = _project(tmp_path)
    q = control.request_interaction(d, "Keep LoRA?", options=["yes", "no"])
    control.respond(d, q.id, "no", comment="the adapter is broken, stop trying LoRA")
    repo = tmp_path / ".researchforge"
    repo.mkdir()
    with sqlite3.connect(repo / "researchforge.db") as c:
        c.execute(
            "CREATE TABLE experiment_plans "
            "(plan_id TEXT, status TEXT, created_at TEXT, record TEXT)"
        )
        c.execute("INSERT INTO experiment_plans VALUES ('plan-9', 'planned', '2026-11-09', '{}')")
    calls: list[list[str]] = []
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda cmd, **kw: calls.append(cmd) or SimpleNamespace(returncode=0, stdout="", stderr=""),
    )
    project = {"root": str(tmp_path)}  # a repo project on its own (no competition)
    out = DashboardService.action_plan_approve(  # type: ignore[arg-type]
        None, project, {"plan": "plan-9", "decision": "reject", "comment": "threshold too low"}
    )
    assert out["decision"] == "reject" and calls[0][-3:] == ["cancel", "plan-9", "--yes"]
    notes = control.person_notes(d)
    assert any("threshold too low" in n for n in notes) and any(
        "stop trying LoRA" in n for n in notes
    )
    prompt = ar.build_command(d, SLUG, ar.AutoRunConfig(driver="claude"))[2]
    assert "follow them" in prompt and "stop trying LoRA" in prompt


def test_a_competition_has_no_repo_plans_to_approve(tmp_path: Path) -> None:
    d = _project(tmp_path)
    repo = tmp_path / ".researchforge"
    repo.mkdir()
    with sqlite3.connect(repo / "researchforge.db") as c:
        c.execute(
            "CREATE TABLE experiment_plans "
            "(plan_id TEXT, status TEXT, created_at TEXT, record TEXT)"
        )
        c.execute("INSERT INTO experiment_plans VALUES ('plan-9', 'planned', '2026-11-09', '{}')")
    project = {"root": str(tmp_path), "overnight": {"artifacts_dir": str(d)}}
    with pytest.raises(ValueError, match="only on its Kaggle loop"):
        DashboardService.action_plan_approve(  # type: ignore[arg-type]
            None, project, {"plan": "plan-9", "decision": "approve", "typed": "approve"}
        )
    (p,) = dashboard_state.build_state(tmp_path)["projects"]
    assert p["kinds"] == ["kaggle"] and not any(q.get("approvable") for q in p["questions"])
