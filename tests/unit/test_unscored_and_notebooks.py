"""Unscored public-notebook rounds reach a submission question; the notebook board ranks by LB.

Enveda's loop ran 0.48–0.50 public notebooks (no local score: hidden test set) and marked
them failed, so they were never offered; an agent's "Submit R21?" was a plain question,
so the person's "yes" submitted nothing; the watcher ignored re-run notebooks and
compared claims with a baseline no project had.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from researchforge.kaggle import control, submission
from researchforge.kaggle.benchmark_runner import UNSCORED_REASON

SLUG = "comp"


@pytest.fixture(autouse=True)
def _env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("RESEARCHFORGE_AUTO_RUN", raising=False)
    monkeypatch.setattr(control, "notify", lambda *a, **k: True)


def _project(root: Path, direction: str = "maximize") -> Path:
    d = root / f".overnight_{SLUG}"
    d.mkdir(parents=True)
    (d / "overnight_setup.json").write_text(
        json.dumps({"competition_slug": SLUG, "selected_gpu": "kaggle_gpu"})
    )
    (d / "competition_metadata.json").write_text(json.dumps({"metric_direction": direction}))
    return d


def _log(d: Path, rows: list[tuple[int, float | None, int, int, str]]) -> None:
    """rows: (round, metric, full_passed, full_improved, error_message)."""
    from researchforge.kaggle.experiment_loop import (
        BenchmarkResult,
        ExperimentRound,
        OvernightExperimentLoop,
    )

    loop = OvernightExperimentLoop.__new__(OvernightExperimentLoop)
    loop.db_path = d / "experiment_log.db"
    loop._init_database()
    for num, metric, passed, improved, error in rows:
        r = ExperimentRound(round_num=num, patch_description=f"idea {num}", patch_code="")
        r.full_result = BenchmarkResult(
            passed=bool(passed),
            metric_value=metric if metric is not None else 0.0,
            metric_improved=bool(improved),
            duration_seconds=1.0,
            error_message=error or None,
        )
        r.started_at = r.completed_at = datetime.now()
        loop._persist_round(r)


def _file(d: Path, name: str) -> str:
    f = d / name
    f.write_text("id,y\n")
    return str(f)


# ---------------------------------------------------------------------------
# Unscored rounds
# ---------------------------------------------------------------------------


def test_a_notebook_with_no_score_keeps_its_run_and_is_marked_unscored(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from researchforge.kaggle import api
    from researchforge.kaggle.benchmark_runner import BenchmarkRunner

    out = tmp_path / ".overnight_c" / "kernel_outputs" / "nb"
    out.mkdir(parents=True)
    (out / "nb.log").write_text("all done\n")
    (out / "submission.csv").write_text("id,y\n")
    monkeypatch.setattr(api, "fetch_kernel_output", lambda ref, o, **kw: list(out.iterdir()))
    runner = BenchmarkRunner.__new__(BenchmarkRunner)
    runner.setup = SimpleNamespace(
        artifacts_dir=tmp_path / ".overnight_c", baseline_result=None, contract=None
    )
    runner.contract = None
    res = runner.collect_kernel("me/nb", 3, 10.0)
    assert not res.passed and res.error_message == UNSCORED_REASON
    assert res.outputs == {
        "kernel_ref": "me/nb",
        "kernel_version": 3,
        "submission_file": str(out / "submission.csv"),
    }


def test_unscored_rounds_with_a_submission_are_offered_once(tmp_path: Path) -> None:
    d = _project(tmp_path)
    _log(
        d,
        [
            (1, 0.90, 1, 0, ""),
            (2, None, 0, 0, UNSCORED_REASON),
            (3, None, 0, 0, "Kernel completed but no metric found in output files"),  # older
            (4, None, 0, 0, "Exit code 1: boom"),  # a crash is not offered
            (5, None, 0, 0, UNSCORED_REASON),  # no kept output: nothing to submit
        ],
    )
    submission.record_round_outputs(d, 2, {"submission_file": _file(d, "a.csv")})
    submission.record_round_outputs(d, 3, {"kernel_ref": "me/nb", "kernel_version": 1})
    submission.record_round_outputs(d, 4, {"submission_file": _file(d, "b.csv")})
    assert [n for n, _ in submission.unscored_rounds(d)] == [2, 3]
    assert submission.offer_missed(d) == 2
    asked = {q.answer_key: q for q in control.open_interactions(d)}
    assert set(asked) == {"round:2", "round:3"}
    assert all(q.kind == "submission_approval" for q in asked.values())
    assert "printed no score" in asked["round:2"].question
    assert submission.offer_missed(d) is None  # asked once


def test_conservative_policy_offers_nothing_unscored(tmp_path: Path) -> None:
    d = _project(tmp_path)
    _log(d, [(1, None, 0, 0, UNSCORED_REASON)])
    submission.record_round_outputs(d, 1, {"submission_file": _file(d, "a.csv")})
    ctl = control.LoopControl.load(d)
    ctl.submission_policy = "conservative"
    ctl.save(d)
    assert submission.offer_missed(d) is None and not control.open_interactions(d)


# ---------------------------------------------------------------------------
# An agent's "Submit R21?" is a real submission question
# ---------------------------------------------------------------------------


def test_asking_to_submit_a_round_makes_its_submission_question(tmp_path: Path) -> None:
    d = _project(tmp_path)
    _log(d, [(21, None, 0, 0, UNSCORED_REASON)])
    submission.record_round_outputs(d, 21, {"submission_file": _file(d, "a.csv")})
    item = control.ask(
        d, "R21 ran the 0.488 notebook. Submit R21's output to Kaggle?", options=["yes", "no"]
    )
    assert item.kind == "submission_approval" and item.answer_key == "round:21"
    assert submission.round_of(item.answer_key, item.question) == 21


def test_asking_to_submit_a_round_with_nothing_to_submit_is_refused(tmp_path: Path) -> None:
    d = _project(tmp_path)
    with pytest.raises(ValueError, match="R9 left nothing Kaggle takes"):
        control.ask(d, "Submit R9 to Kaggle?", options=["yes", "no"])
    assert not control.open_interactions(d)


def test_other_questions_stay_plain(tmp_path: Path) -> None:
    d = _project(tmp_path)
    item = control.ask(
        d, "Should rounds submit R21-style notebooks?", options=["yes", "run them unscored"]
    )
    assert item.kind == "other"
    assert control.submission_round("Which compute for R3?", ["yes", "no"]) is None


def test_offer_submission_command(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from typer.testing import CliRunner

    from researchforge.kaggle import control_cli
    from researchforge.kaggle.cli import kaggle_app

    d = _project(tmp_path)
    _log(d, [(22, None, 0, 0, UNSCORED_REASON)])
    monkeypatch.setattr(control_cli, "_artifacts", lambda c: d)
    run = CliRunner().invoke(kaggle_app, ["offer-submission", SLUG, "22", "--note", "claims 0.500"])
    assert run.exit_code == 1 and "nothing Kaggle takes" in run.output
    submission.record_round_outputs(d, 22, {"kernel_ref": "me/nb", "kernel_version": 1})
    run = CliRunner().invoke(kaggle_app, ["offer-submission", SLUG, "22", "--note", "claims 0.500"])
    assert run.exit_code == 0, run.output
    (q,) = control.open_interactions(d)
    assert q.answer_key == "round:22" and q.question == (
        "Submit R22 'idea 22' (claims 0.500) to Kaggle?"
    )


# ---------------------------------------------------------------------------
# Dashboard: improvements without a baseline, unscored rounds
# ---------------------------------------------------------------------------


def test_dashboard_counts_improvements_over_earlier_rounds(tmp_path: Path) -> None:
    from researchforge.kaggle.dashboard_state import _overnight_experiment

    row = {
        "round_num": 3,
        "full_passed": 1,
        "full_metric": 0.92,
        "full_improved": 1,
        "completed_at": "2026-11-13T10:00:00",
    }
    assert _overnight_experiment(row, None, "maximize", 0.91)["status"] == "improved"
    assert _overnight_experiment(row, None, "maximize", 0.95)["status"] == "complete"
    low = {**row, "full_metric": 22.1}
    assert _overnight_experiment(low, None, "minimize", 22.2)["status"] == "improved"
    unscored = {"round_num": 4, "full_passed": 0, "rejected_reason": UNSCORED_REASON}
    assert _overnight_experiment(unscored, None, "maximize", 0.9)["status"] == "unscored"


# ---------------------------------------------------------------------------
# Notebook board and the watcher
# ---------------------------------------------------------------------------


def _k(ref: str, title: str, run: str = "2026-11-13 10:00:00", votes: int = 1) -> Any:
    from researchforge.kaggle.api import PublicKernel

    return PublicKernel(ref=ref, title=title, author="a", last_run_time=run, total_votes=votes)


def _scores(d: Path, rows: list[tuple[str, str]]) -> None:
    (d / "submissions.json").write_text(
        json.dumps(
            {
                "fetched_at": "x",
                "error": "",
                "submissions": [{"description": desc, "publicscore": s} for desc, s in rows],
            }
        )
    )


def test_board_ranks_by_claimed_lb_and_links_our_rounds(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from researchforge.kaggle import kernel_board, kernel_claims

    d = _project(tmp_path)
    _log(d, [(21, None, 0, 0, UNSCORED_REASON)])
    with sqlite3.connect(d / "experiment_log.db") as c:
        c.execute("UPDATE experiments SET source_credit='x/b' WHERE round_num=21")
    _scores(d, [("RF R21: x/b verbatim", "0.47")])
    fetched: list[str] = []
    monkeypatch.setattr(
        kernel_claims,
        "notebook_claim",
        lambda ref: (
            fetched.append(ref)
            or {"claimed_score": 0.51, "claim_source": "notebook", "claim_snippet": "LB 0.51"}
        ),
    )
    board = kernel_board.update_board(
        d,
        [_k("x/c", "plain eda")],
        [_k("x/a", "casmi 0.488"), _k("x/b", "LB 0.500 fork")],
        direction="maximize",
        our_best_lb=0.21,
    )
    rows = board["rows"]
    assert [r["ref"] for r in rows] == ["x/c", "x/b", "x/a"]  # 0.51 (notebook), 0.500, 0.488
    assert [r["kaggle_rank"] for r in rows] == [None, 2, 1]
    assert rows[1]["gap"] == pytest.approx(0.29)
    assert rows[1]["tried"] == [{"round": 21, "local": None, "lb": 0.47, "note": UNSCORED_REASON}]
    assert fetched == ["x/c"]  # the titles carry the other claims
    # Unchanged notebooks are not downloaded again; a re-run one is.
    kernel_board.update_board(
        d,
        [_k("x/c", "plain eda", run="2026-11-13 12:00:00")],
        [_k("x/a", "casmi 0.488"), _k("x/b", "LB 0.500 fork")],
        direction="maximize",
        our_best_lb=0.21,
        updated_refs={"x/c"},
    )
    assert fetched == ["x/c", "x/c"]
    ahead = kernel_board.ahead_of_us(d)
    assert [a["ref"] for a in ahead] == ["x/c", "x/b", "x/a"] and ahead[0]["updated_at"]


def test_watcher_sees_rerun_notebooks_and_compares_with_our_best_lb(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from researchforge.kaggle import api, kernel_claims, kernel_watcher

    d = _project(tmp_path)
    _scores(d, [("RF R5", "0.21")])
    monkeypatch.setattr(kernel_claims, "notebook_claim", lambda ref: None)
    sent: list[dict[str, Any]] = []
    monkeypatch.setattr(
        kernel_watcher, "_notify_better_kernel", lambda slug, k, d=None: sent.append(k) or True
    )
    listing = {"dateCreated": [], "scoreDescending": [_k("x/a", "casmi 0.300")], "dateRun": []}
    monkeypatch.setattr(
        api, "list_competition_kernels", lambda slug, sort_by="", page_size=0: listing[sort_by]
    )
    kernel_watcher.WatcherConfig(competition_slug=SLUG, interval_hours=1.0).save(d)

    first = kernel_watcher.run_watch_cycle(d, None, None, check_submissions=False)
    assert first.new_kernels_found == 1 and first.better_kernel["delta"] == pytest.approx(0.09)
    quiet = kernel_watcher.run_watch_cycle(d, None, None, check_submissions=False)
    assert quiet.new_kernels_found == 0 and quiet.updated_kernels_found == 0

    # The author re-runs it with a better score in the title.
    listing["dateRun"] = [_k("x/a", "casmi 0.540", run="2026-11-13 15:00:00")]
    listing["scoreDescending"] = listing["dateRun"]
    again = kernel_watcher.run_watch_cycle(d, None, None, check_submissions=False)
    assert again.updated_kernels_found == 1
    assert again.better_kernel["lb_score"] == 0.54 and again.better_kernel["updated"]
    assert len(sent) == 2
    row = kernel_watcher_board_row(d, "x/a")
    assert row["claimed_score"] == 0.54 and row["updated_at"]


def kernel_watcher_board_row(d: Path, ref: str) -> dict[str, Any]:
    from researchforge.kaggle.kernel_board import load_board

    return next(r for r in load_board(d)["rows"] if r["ref"] == ref)


# ---------------------------------------------------------------------------
# Submitting: the daily limit, a round sent twice, two approvals at once
# ---------------------------------------------------------------------------


def _sent(d: Path, round_num: int, reply: str, at: datetime) -> None:
    with (d / submission.LOG_FILE).open("a") as fh:
        fh.write(
            json.dumps({"at": at.isoformat(timespec="seconds"), "round": round_num, "reply": reply})
            + "\n"
        )


def _no_kaggle(monkeypatch: pytest.MonkeyPatch) -> list[Any]:
    calls: list[Any] = []
    monkeypatch.setattr(submission.subprocess, "run", lambda *a, **k: calls.append(a))
    return calls


def test_a_used_up_daily_limit_is_refused_before_calling_kaggle(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    d = _project(tmp_path)
    calls = _no_kaggle(monkeypatch)
    _sent(d, 22, "Successfully submitted\n0 submissions remaining today.", datetime.now())
    left = submission.daily_allowance(d)
    assert left is not None and left["remaining"] == 0 and left["after_round"] == 22
    with pytest.raises(ValueError, match="daily submission limit is used up: after R22"):
        submission.submit_round(d, SLUG, 21, "x")
    assert calls == []


def test_yesterdays_allowance_does_not_count(tmp_path: Path) -> None:
    from datetime import timedelta

    d = _project(tmp_path)
    _sent(d, 22, "0 submissions remaining today.", datetime.now() - timedelta(days=2))
    assert submission.daily_allowance(d) is None


def test_a_round_is_never_submitted_twice(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    d = _project(tmp_path)
    calls = _no_kaggle(monkeypatch)
    _sent(d, 26, "3 submissions remaining today.", datetime.now())
    with pytest.raises(ValueError, match="R26 was already submitted"):
        submission.submit_round(d, SLUG, 26, "x")
    (d / "submitting.lock").write_text("27")
    with pytest.raises(ValueError, match="another submission"):
        submission.submit_round(d, SLUG, 27, "x")
    assert calls == []


def test_a_bare_kaggle_400_names_its_usual_causes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    d = _project(tmp_path)
    submission.record_round_outputs(d, 8, {"submission_file": _file(d, "a.csv")})
    monkeypatch.setattr(submission, "is_code_competition", lambda d, slug: False)
    reply = (
        "Warning: Looks like you're using an outdated `kaggle` version (installed: 2.2.4)\n"
        "400 Client Error: Bad Request for url: https://api.kaggle.com/v1/competitions.Create"
    )
    monkeypatch.setattr(
        submission.subprocess,
        "run",
        lambda *a, **k: SimpleNamespace(returncode=1, stdout=reply, stderr=""),
    )
    monkeypatch.setattr(submission, "_limit_reached", lambda d: "")
    with pytest.raises(ValueError) as err:
        submission.submit_round(d, SLUG, 8, "x")
    said = str(err.value)
    assert "outdated" not in said and "400 Bad Request" in said and "submission limit" in said
    assert not (d / "submitting.lock").exists()


def test_the_dashboard_carries_the_notebook_board(tmp_path: Path) -> None:
    from researchforge.kaggle import dashboard_state, kernel_board

    d = _project(tmp_path)
    kernel_board.update_board(
        d, [], [_k("x/a", "casmi 0.488")], direction="maximize", our_best_lb=0.21
    )
    state = dashboard_state.build_state(tmp_path)
    (p,) = [p for p in state["projects"] if p.get("overnight")]
    assert p["kernel_board"]["rows"][0]["ref"] == "x/a" and p["kernel_board"]["total"] == 1
