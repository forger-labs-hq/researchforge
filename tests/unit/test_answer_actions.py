"""Answers that change Run configs, and the submissions-left count on submission offers.

An auto-run cannot change settings, so it used to ask "switch compute to kaggle?" and
the person's click only recorded the words; the next launch found compute unchanged
and asked again. An answer can now carry the change, applied when the person picks it.
"""

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from researchforge.kaggle import control, run_config, submission

SLUG = "demo-comp"
SWITCH = "Switch to Kaggle GPU and raise the budget to 40"


def _overnight(root: Path) -> Path:
    d = root / f".overnight_{SLUG}"
    d.mkdir(parents=True)
    setup = {
        "competition_slug": SLUG,
        "artifacts_dir": str(d),
        "data_dir": str(root / "data"),
        "selected_gpu": "colab_gpu",
        "max_experiments": 20,
    }
    (d / "overnight_setup.json").write_text(json.dumps(setup))
    return d


def _ask(d: Path) -> control.Interaction:
    return control.ask(
        d,
        "The budget is used up and compute is still Colab. How should rounds continue?",
        options=[SWITCH, "Stop the loop"],
        actions={SWITCH: {"compute": "kaggle", "max_experiments": 40}},
        source="auto-run (claude)",
    )


def _setup(d: Path) -> dict[str, Any]:
    return dict(json.loads((d / "overnight_setup.json").read_text()))


class TestAnswersThatChangeSettings:
    def test_picking_the_answer_changes_run_configs(self, tmp_path: Path) -> None:
        d = _overnight(tmp_path)
        item = control.respond(d, _ask(d).id, SWITCH)
        setup = _setup(d)
        assert (setup["selected_gpu"], setup["max_experiments"]) == ("kaggle_gpu", 40)
        assert item.applied == ["compute: colab_gpu → kaggle_gpu", "max_experiments: 20 → 40"]
        log = run_config.change_log(d)
        assert {c["field"] for c in log} == {"compute", "max_experiments"}
        assert all(c["via"] == f"answer to {item.id}" for c in log)

    def test_only_real_changes_are_reported(self, tmp_path: Path) -> None:
        d = _overnight(tmp_path)
        q = _ask(d)
        run_config.set_field(d, "compute", "kaggle", via="dashboard")  # done by hand first
        assert control.respond(d, q.id, SWITCH).applied == ["max_experiments: 20 → 40"]

    def test_the_other_answer_changes_nothing(self, tmp_path: Path) -> None:
        d = _overnight(tmp_path)
        item = control.respond(d, _ask(d).id, "Stop the loop")
        assert item.applied == []
        assert _setup(d)["selected_gpu"] == "colab_gpu"

    def test_a_refused_value_keeps_the_question_open(self, tmp_path: Path) -> None:
        d = _overnight(tmp_path)
        q = control.ask(
            d,
            "Raise the budget?",
            options=["Raise to 5000", "no"],
            actions={"Raise to 5000": {"max_experiments": 5000}},
        )
        with pytest.raises(ValueError, match="could not set max_experiments to 5000"):
            control.respond(d, q.id, "Raise to 5000")
        assert [i.id for i in control.open_interactions(d)] == [q.id]
        assert _setup(d)["max_experiments"] == 20

    def test_an_unattended_run_cannot_pick_it(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        d = _overnight(tmp_path)
        q = _ask(d)
        monkeypatch.setenv("RESEARCHFORGE_AUTO_RUN", "1")
        with pytest.raises(ValueError, match="only the person can pick it"):
            control.respond(d, q.id, SWITCH)
        assert _setup(d)["selected_gpu"] == "colab_gpu"
        assert control.respond(d, q.id, "Stop the loop").applied == []  # plain answers still work

    def test_only_known_answers_and_fields(self, tmp_path: Path) -> None:
        d = _overnight(tmp_path)
        with pytest.raises(ValueError, match="not an answer"):
            control.ask(d, "Q?", options=["a", "b"], actions={"c": {"compute": "kaggle"}})
        with pytest.raises(ValueError, match="cannot change 'ntfy_topic'"):
            control.ask(d, "Q?", options=["a", "b"], actions={"a": {"ntfy_topic": "x"}})


def test_actions_are_written_as_answer_arrow_changes() -> None:
    assert control.parse_action("Switch => compute:kaggle, max_experiments:40") == (
        "Switch",
        {"compute": "kaggle", "max_experiments": "40"},
    )
    with pytest.raises(ValueError, match="<answer> => field:value"):
        control.parse_action("compute:kaggle")
    with pytest.raises(ValueError, match="field:value"):
        control.parse_action("Switch => compute")


def test_the_cli_asks_with_an_action_and_respond_reports_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from researchforge.cli import app

    d = _overnight(tmp_path)
    monkeypatch.setenv("RESEARCHFORGE_OVERNIGHT_DIR", str(d))
    runner = CliRunner()
    asked = runner.invoke(
        app,
        [
            "kaggle",
            "ask",
            SLUG,
            "How should rounds continue?",
            "--options",
            f"{SWITCH}|Stop",
            "--action",
            f"{SWITCH} => compute:kaggle, max_experiments:40",
        ],
    )
    assert asked.exit_code == 0, asked.output
    assert "changes Run configs: compute → kaggle, max_experiments → 40" in asked.output
    qid = control.open_interactions(d)[0].id
    answered = runner.invoke(app, ["kaggle", "respond", SLUG, qid, SWITCH])
    assert answered.exit_code == 0, answered.output
    assert "Run configs: compute: colab_gpu → kaggle_gpu" in answered.output


def test_the_dashboard_answer_returns_what_changed(tmp_path: Path) -> None:
    from researchforge.kaggle.dashboard import DashboardService

    d = _overnight(tmp_path)
    q = _ask(d)
    project = {"root": str(tmp_path), "overnight": {"artifacts_dir": str(d)}, "slug": SLUG}
    out = DashboardService(tmp_path).action_answer(project, {"id": q.id, "answer": SWITCH})
    assert out["applied"] == ["compute: colab_gpu → kaggle_gpu", "max_experiments: 20 → 40"]


def test_the_dashboard_shows_each_answers_changes(tmp_path: Path) -> None:
    from researchforge.kaggle.dashboard_state import _questions

    d = _overnight(tmp_path)
    _ask(d)
    (q,) = _questions(d, None, [], SLUG)
    assert q["actions"] == {SWITCH: {"compute": "kaggle", "max_experiments": 40}}


class TestSubmissionsLeftToday:
    def _snapshot(self, d: Path, dates: list[str], max_daily: int | None = 5) -> None:
        rows = [{"ref": str(i), "date": day} for i, day in enumerate(dates)]
        snap: dict[str, Any] = {
            "fetched_at": "2026-11-15T12:00:00",
            "error": "",
            "submissions": rows,
        }
        if max_daily:
            snap["max_daily"] = max_daily
        (d / "submissions.json").write_text(json.dumps(snap))

    def test_the_daily_limit_minus_todays_submissions(self, tmp_path: Path) -> None:
        d = _overnight(tmp_path)
        today = datetime.now(UTC).strftime("%Y-%m-%d")
        yesterday = (datetime.now(UTC) - timedelta(days=1)).strftime("%Y-%m-%d")
        self._snapshot(d, [f"{today} 07:08:13.9", f"{today} 07:07:58.2", f"{yesterday} 23:00:00"])
        left = submission.daily_allowance(d)
        assert left is not None
        assert (left["remaining"], left["max"], left["after_round"]) == (3, 5, None)

    def test_unknown_without_the_limit(self, tmp_path: Path) -> None:
        d = _overnight(tmp_path)
        self._snapshot(d, [], max_daily=None)
        assert submission.daily_allowance(d) is None

    def test_kaggles_own_reply_wins_and_keeps_the_limit(self, tmp_path: Path) -> None:
        d = _overnight(tmp_path)
        self._snapshot(d, [], max_daily=5)
        now = datetime.now(UTC).astimezone().isoformat(timespec="seconds")
        line = {
            "round": 22,
            "at": now,
            "reply": "Successfully submitted. 1 submission remaining today",
        }
        (d / submission.LOG_FILE).write_text(json.dumps(line) + "\n")
        left = submission.daily_allowance(d)
        assert left is not None
        assert (left["remaining"], left["max"], left["after_round"]) == (1, 5, 22)

    def test_a_used_up_day_refuses_before_calling_kaggle(self, tmp_path: Path) -> None:
        d = _overnight(tmp_path)
        today = datetime.now(UTC).strftime("%Y-%m-%d")
        self._snapshot(d, [f"{today} 0{i}:00:00" for i in range(5)])
        assert "all 5 of today's submissions are used" in submission._limit_reached(d)


def test_kaggles_competition_facts_are_read_safely() -> None:
    from researchforge.kaggle.competition_reader import _facts

    assert _facts(True, 5) == {"kernels_only": True, "max_daily": 5}
    assert _facts(None, "0") == {}
    assert _facts(False, "not a number") == {"kernels_only": False}
