"""Approving a submission question submits that round to Kaggle — only on the person's yes."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from typer.testing import CliRunner

from researchforge.kaggle import control, submission

SLUG = "comp"


@pytest.fixture(autouse=True)
def _env(monkeypatch: pytest.MonkeyPatch) -> None:
    from researchforge.kaggle import competition_reader

    monkeypatch.delenv("RESEARCHFORGE_AUTO_RUN", raising=False)
    monkeypatch.setattr(control, "notify", lambda *a, **k: True)
    monkeypatch.setattr(competition_reader, "_fetch_competition_api", lambda slug: None)
    # No Kaggle Python API in tests: submissions go through the (mocked) CLI.
    monkeypatch.setattr(competition_reader, "run_on_kaggle_python", lambda *a, **k: "")
    from researchforge.kaggle import cli as _cli

    listed: list[str] = []
    monkeypatch.setattr(_cli, "_submission_check", lambda slug, d: listed.append(slug))


def _project(root: Path, code: bool = False) -> Path:
    d = root / f".overnight_{SLUG}"
    d.mkdir()
    (d / "overnight_setup.json").write_text(json.dumps({"competition_slug": SLUG}))
    (d / "competition_metadata.json").write_text(json.dumps({"is_code_competition": code}))
    return d


def _fake_kaggle(monkeypatch: pytest.MonkeyPatch, reply: str, code: int = 0) -> list[list[str]]:
    calls: list[list[str]] = []

    def run(cmd: list[str], **kw: object) -> SimpleNamespace:
        calls.append(cmd)
        return SimpleNamespace(returncode=code, stdout=reply, stderr="")

    monkeypatch.setattr(submission.subprocess, "run", run)
    return calls


def _offer(d: Path, round_num: int = 7) -> str:
    """A question for the round, whatever it left (these tests are about the answer)."""
    ctl = control.LoopControl.load(d)
    ctl.submission_policy = "aggressive"
    ctl.save(d)
    item = control.offer_submission(d, round_num, "tilted bands", 22.2, "kaggle_gpu")
    assert item is not None and item.answer_key == f"round:{round_num}"
    return item.id


def test_yes_submits_the_rounds_file_and_records_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    d = _project(tmp_path)
    sub = tmp_path / "submission.csv"
    sub.write_text("id,y\n1,0\n")
    submission.record_round_outputs(d, 7, {"submission_file": str(sub)})
    calls = _fake_kaggle(monkeypatch, "Successfully submitted to comp")
    qid = _offer(d)
    line = submission.answer(d, SLUG, qid, "yes")
    assert line.startswith("R7 submitted to Kaggle")
    assert calls[0][:6] == ["kaggle", "competitions", "submit", SLUG, "-f", str(sub)]
    assert calls[0][-1] == "RF R7: tilted bands"
    (item,) = control.load_interactions(d)
    assert (item.status, item.answer) == ("answered", "yes")
    (logged,) = (d / submission.LOG_FILE).read_text().splitlines()
    assert json.loads(logged)["round"] == 7


def test_a_refused_submission_leaves_the_question_open(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    d = _project(tmp_path)
    sub = tmp_path / "submission.csv"
    sub.write_text("x")
    submission.record_round_outputs(d, 7, {"submission_file": str(sub)})
    _fake_kaggle(monkeypatch, "400 - Submission limit exceeded for today")  # exit code 0
    qid = _offer(d)
    with pytest.raises(ValueError, match="refused"):
        submission.answer(d, SLUG, qid, "yes")
    assert [q.id for q in control.open_interactions(d)] == [qid]
    assert not (d / submission.LOG_FILE).exists()


def test_nothing_to_submit_says_so_and_no_submits_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    d = _project(tmp_path)
    calls = _fake_kaggle(monkeypatch, "Successfully submitted")
    qid = _offer(d)
    with pytest.raises(ValueError, match="left no submission file"):
        submission.answer(d, SLUG, qid, "yes")
    assert "nothing submitted" in submission.answer(d, SLUG, qid, "no")
    assert calls == []


def test_a_code_competition_submits_the_rounds_notebook(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    d = _project(tmp_path, code=True)
    calls = _fake_kaggle(monkeypatch, "Successfully submitted")
    qid = _offer(d, 3)
    with pytest.raises(ValueError, match="notebook submissions"):
        submission.answer(d, SLUG, qid, "yes")
    submission.record_round_outputs(
        d,
        3,
        {"kernel_ref": "me/rf-run", "kernel_version": 2, "submission_file": "/x/submission.zip"},
    )
    submission.answer(d, SLUG, qid, "yes")
    assert calls[0][4:10] == ["-k", "me/rf-run", "-v", "2", "-f", "submission.zip"]


def test_an_unattended_run_can_never_approve(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from researchforge.kaggle import auto_run as ar

    d = _project(tmp_path)
    calls = _fake_kaggle(monkeypatch, "Successfully submitted")
    qid = _offer(d)
    monkeypatch.setenv("RESEARCHFORGE_AUTO_RUN", "1")
    with pytest.raises(ValueError, match="only the person"):
        submission.answer(d, SLUG, qid, "yes")
    assert calls == []
    cmd = ar.build_command(d, SLUG, ar.AutoRunConfig(driver="claude"))
    assert "Bash(researchforge kaggle respond:*)" in cmd[cmd.index("--disallowedTools") :]


def test_an_older_question_finds_its_round_in_the_text() -> None:
    assert submission.round_of("", "Submit R12 'x' (metric 0.9, measured on cpu) to Kaggle?") == 12
    assert submission.round_of("round:4", "anything") == 4


def test_the_cli_and_the_dashboard_submit_on_yes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from researchforge.cli import app
    from researchforge.kaggle.dashboard import DashboardService

    d = _project(tmp_path)
    sub = tmp_path / "submission.csv"
    sub.write_text("x")
    submission.record_round_outputs(d, 7, {"submission_file": str(sub)})
    submission.record_round_outputs(d, 8, {"submission_file": str(sub)})
    calls = _fake_kaggle(monkeypatch, "Successfully submitted")
    monkeypatch.chdir(tmp_path)
    out = CliRunner().invoke(app, ["kaggle", "respond", SLUG, _offer(d, 7), "yes"])
    assert out.exit_code == 0 and "R7 submitted" in out.output, out.output
    project = {"slug": SLUG, "overnight": {"artifacts_dir": str(d)}}
    got = DashboardService.action_answer(None, project, {"id": _offer(d, 8), "answer": "yes"})  # type: ignore[arg-type]
    assert "R8 submitted" in got["output"] and len(calls) == 2


def test_a_local_round_keeps_the_file_it_wrote(tmp_path: Path) -> None:
    import time

    from researchforge.kaggle.benchmark_runner import BenchmarkRunner

    d = _project(tmp_path)
    work = tmp_path / "work"
    work.mkdir()
    r = BenchmarkRunner.__new__(BenchmarkRunner)
    r.working_dir = work
    r.setup = SimpleNamespace(artifacts_dir=d)  # type: ignore[assignment]
    since = time.time() - 1
    assert r._keep_submission_file(since) == ""
    (work / "submission.csv").write_text("id,y\n")
    kept = Path(r._keep_submission_file(since))
    assert kept.read_text() == "id,y\n" and kept.parent.parent == d / "round_outputs"


def test_kaggles_answer_wins_over_stale_metadata(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from researchforge.kaggle import competition_reader

    d = _project(tmp_path, code=False)
    monkeypatch.setattr(competition_reader, "_fetch_competition_api", lambda slug: True)
    assert submission.is_code_competition(d, SLUG) is True
    assert json.loads((d / "competition_metadata.json").read_text())["is_code_competition"]


# ── asked automatically ─────────────────────────────────────────────────────


def _log(d: Path, rows: list[tuple[int, float | None, int, int]], minimize: bool = True) -> None:
    """rows: (round, metric, full_passed, full_improved)."""
    from datetime import datetime

    from researchforge.kaggle.experiment_loop import (
        BenchmarkResult,
        ExperimentRound,
        OvernightExperimentLoop,
    )

    meta = {
        "is_code_competition": False,
        "metric_direction": "minimize" if minimize else "maximize",
    }
    (d / "competition_metadata.json").write_text(json.dumps(meta))
    loop = OvernightExperimentLoop.__new__(OvernightExperimentLoop)
    loop.db_path = d / "experiment_log.db"
    loop._init_database()
    for num, metric, passed, improved in rows:
        r = ExperimentRound(round_num=num, patch_description=f"idea {num}", patch_code="")
        r.full_result = BenchmarkResult(
            passed=bool(passed),
            metric_value=metric if metric is not None else 0.0,
            metric_improved=bool(improved),
            duration_seconds=1.0,
            error_message=None if passed else "Kernel ended with status: error",
        )
        r.started_at = r.completed_at = datetime.now()
        loop._persist_round(r)


def test_the_best_round_is_asked_about_once_when_its_offer_was_missed(
    tmp_path: Path,
) -> None:
    d = _project(tmp_path)
    _log(d, [(1, 22.25, 1, 1), (2, 22.21, 1, 1), (3, 22.21, 1, 0), (4, None, 0, 0)])
    assert submission.offer_missed(d) is None  # R2 left nothing to submit
    sub = tmp_path / "best.csv"
    sub.write_text("id,y\n")
    submission.record_round_outputs(d, 2, {"submission_file": str(sub)})
    assert submission.offer_missed(d) == 2
    (q,) = control.open_interactions(d)
    assert q.answer_key == "round:2" and "R2 'idea 2'" in q.question
    assert submission.offer_missed(d) is None  # asked once, even after "no"
    submission.answer(d, SLUG, q.id, "no")
    assert submission.offer_missed(d) is None


def test_a_failed_round_records_why(tmp_path: Path) -> None:
    import sqlite3

    d = _project(tmp_path)
    _log(d, [(1, None, 0, 0)])
    with sqlite3.connect(d / "experiment_log.db") as conn:
        (reason,) = conn.execute("SELECT rejected_reason FROM experiments").fetchone()
    assert reason == "Kernel ended with status: error"


def test_a_kaggle_round_that_outlives_its_wait_is_collected_and_offered(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import sqlite3

    from researchforge.kaggle import api, late_rounds
    from researchforge.kaggle.benchmark_runner import BenchmarkRunner

    d = _project(tmp_path)
    (d / "overnight_setup.json").write_text(  # no contract: the metric is maximized
        json.dumps(
            {
                "competition_slug": SLUG,
                "selected_gpu": "kaggle_gpu",
                "artifacts_dir": str(d),
                "data_dir": str(d / "data"),
            }
        )
    )
    _log(d, [(1, 0.80, 1, 1), (2, None, 0, 0)], minimize=False)
    submission.record_round_outputs(
        d, 2, {"kernel_ref": "me/rf-r2", "kernel_version": 1, "pending": True}
    )
    assert not submission.submittable(d, 2)
    status = {"me/rf-r2": "running"}
    monkeypatch.setattr(api, "get_kernel_run_status", lambda ref: (status[ref], ""))
    assert late_rounds.collect(d) == []  # still running: left alone

    def fetch(
        ref: str,
        out: Path,
        force: bool = True,
        file_pattern: str | None = None,
        timeout: float = 600,
    ) -> list[Path]:
        out.mkdir(parents=True, exist_ok=True)
        (out / "inputs.json").write_text('{"score": 9.99}')  # a data file, not the run's score
        log = [  # Kaggle's log format
            {"stream_name": "stdout", "time": 1.0, "data": "Score: 0.85\n"},
            {"stream_name": "stdout", "time": 2.0, "data": "SUBMISSION_FILE: preds.csv\n"},
        ]
        (out / "rf-r2.log").write_text(json.dumps(log))
        if file_pattern and "preds" in file_pattern:
            (out / "preds.csv").write_text("id,y\n")
        return list(out.iterdir())

    monkeypatch.setattr(api, "fetch_kernel_output", fetch)
    status["me/rf-r2"] = "complete"
    (line,) = late_rounds.collect(d)
    assert "R2 'idea 2' finished on Kaggle: 0.8500" in line and "improved" in line
    with sqlite3.connect(d / "experiment_log.db") as conn:
        row = conn.execute(
            "SELECT full_passed, full_metric, full_improved, rejected_reason "
            "FROM experiments WHERE round_num = 2"
        ).fetchone()
    assert row == (1, 0.85, 1, None)
    out = submission.round_outputs(d, 2)
    assert "pending" not in out and Path(str(out["submission_file"])).name == "preds.csv"
    (q,) = control.open_interactions(d)
    assert q.answer_key == "round:2"
    assert late_rounds.pending_rounds(d) == {}
    assert BenchmarkRunner  # imported for the runner the collector builds


def test_a_round_that_names_its_submission_file_keeps_it(tmp_path: Path) -> None:
    import time

    from researchforge.kaggle.benchmark_runner import BenchmarkRunner, declared_submission_file

    d = _project(tmp_path)
    work = tmp_path / "work"
    work.mkdir()
    r = BenchmarkRunner.__new__(BenchmarkRunner)
    r.working_dir = tmp_path / "data"
    r.setup = SimpleNamespace(artifacts_dir=d)  # type: ignore[assignment]
    since = time.time() - 1
    (work / "best.csv").write_text("id,n\n")
    named = declared_submission_file(f"LB 22.2\nSUBMISSION_FILE: {work / 'best.csv'}\ndone\n")
    kept = Path(r._keep_submission_file(since, named))
    assert kept.name == "submission.csv" and kept.read_text() == "id,n\n"


def test_a_percent_continuation_line_is_not_a_notebook_command() -> None:
    from researchforge.kaggle.kernel_port import cell_to_script

    src = 'msg = ("%s of %s"\n       % (a, b))\n%matplotlib inline'
    out = cell_to_script(src)
    assert "       % (a, b))" in out and "pass  # notebook magic" in out


# A notebook round that wrote no submission file (R49 ran out of time) has nothing
# Kaggle takes: approving it only got a 400 Bad Request.


def _notebook_output(
    monkeypatch: pytest.MonkeyPatch, files: list[str] | None, log: bool = True
) -> list[str]:
    """Kaggle's output for the notebook: its log plus ``files``; None when it cannot be read.

    ``log=False`` is a listing cut short (Kaggle rate-limits its pages)."""
    from researchforge.kaggle import api

    asked: list[str] = []

    def fetch(ref: str, out: Path, **kw: object) -> list[Path]:
        asked.append(ref)
        if files is None:
            return []
        out.mkdir(parents=True, exist_ok=True)
        for name in [f"{ref.split('/')[-1]}.log"] * log + files:
            (out / name).write_text("x")
        return list(out.iterdir())

    monkeypatch.setattr(api, "fetch_kernel_output", fetch)
    return asked


def test_a_notebook_that_wrote_no_submission_is_not_submittable(tmp_path: Path) -> None:
    d = _project(tmp_path, code=True)
    submission.record_round_outputs(d, 49, {"kernel_ref": "me/nb", "kernel_version": 1})
    assert not submission.submittable(d, 49)
    assert control.offer_submission(d, 49, "public demo", None, "kaggle_gpu") is None
    with pytest.raises(ValueError, match="nothing Kaggle takes"):
        control.ask(d, "Submit R49 to Kaggle?", options=["yes", "no"])
    submission.record_round_outputs(
        d, 49, {"kernel_ref": "me/nb", "kernel_version": 1, "submission_file": "/gone/s.csv"}
    )
    assert submission.submittable(d, 49)  # Kaggle takes it from the notebook's output


def test_approving_one_refuses_before_calling_kaggle(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    d = _project(tmp_path, code=True)
    submission.record_round_outputs(d, 49, {"kernel_ref": "me/nb", "kernel_version": 1})
    calls = _fake_kaggle(monkeypatch, "Successfully submitted")
    _notebook_output(monkeypatch, ["baseline_submission.csv"])
    qid = _offer(d, 49)
    with pytest.raises(ValueError, match="wrote no submission file"):
        submission.answer(d, SLUG, qid, "yes")
    assert calls == []


def test_a_submission_it_never_kept_is_found_in_its_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    d = _project(tmp_path, code=True)
    submission.record_round_outputs(d, 8, {"kernel_ref": "me/nb", "kernel_version": 3})
    calls = _fake_kaggle(monkeypatch, "Successfully submitted")
    _notebook_output(monkeypatch, ["submission.csv"])
    qid = _offer(d, 8)
    submission.answer(d, SLUG, qid, "yes")
    assert calls[0][4:10] == ["-k", "me/nb", "-v", "3", "-f", "submission.csv"]
    assert submission.submittable(d, 8)


def test_stale_offers_are_withdrawn_only_when_the_output_shows_none(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    d = _project(tmp_path, code=True)
    for n in (47, 49, 53):
        submission.record_round_outputs(d, n, {"kernel_ref": f"me/nb{n}", "kernel_version": 1})
        _offer(d, n)
    good = tmp_path / "submission.csv"
    good.write_text("x")
    submission.record_round_outputs(
        d, 53, {"kernel_ref": "me/nb53", "kernel_version": 1, "submission_file": str(good)}
    )
    asked = _notebook_output(monkeypatch, None)  # Kaggle cannot be read: keep them all
    assert submission.withdraw_unsubmittable(d) == []
    assert asked == ["me/nb47", "me/nb49"]  # R53 needs no look
    _notebook_output(monkeypatch, ["baseline_submission.csv"], log=False)
    assert submission.withdraw_unsubmittable(d) == []  # a cut-short listing proves nothing
    _notebook_output(monkeypatch, ["baseline_submission.csv"])
    assert submission.withdraw_unsubmittable(d) == [47, 49]
    assert [q.answer_key for q in control.open_interactions(d)] == ["round:53"]
    closed = [q for q in control.load_interactions(d) if q.status == "dismissed"]
    assert all("wrote no submission file" in q.answer for q in closed)
    assert submission.withdraw_unsubmittable(d) == []


def test_submissions_without_a_count_still_use_one_up(tmp_path: Path) -> None:
    """Four quick approvals: Kaggle gave a count after the first two, none after the rest."""
    from datetime import datetime

    d = _project(tmp_path)
    (d / "submissions.json").write_text(json.dumps({"max_daily": 5, "submissions": []}))
    now = datetime.now().isoformat(timespec="seconds")
    replies = ["4 submissions remaining today.", "3 submissions remaining today.", "", ""]
    with (d / submission.LOG_FILE).open("w") as fh:
        for n, reply in zip((58, 57, 56, 55), replies, strict=True):
            fh.write(json.dumps({"at": now, "round": n, "reply": reply}) + "\n")
    left = submission.daily_allowance(d)
    assert left is not None and (left["remaining"], left["after_round"]) == (1, 55)
    with (d / submission.LOG_FILE).open("a") as fh:
        fh.write(json.dumps({"at": now, "round": 53, "reply": ""}) + "\n")
    assert "none are left after R53" in submission._limit_reached(d)


def test_a_refusal_is_recorded_with_its_round(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    d = _project(tmp_path)
    sub = tmp_path / "submission.csv"
    sub.write_text("x")
    submission.record_round_outputs(d, 7, {"submission_file": str(sub)})
    _fake_kaggle(monkeypatch, "400 Client Error: Bad Request for url: x", code=1)
    qid = _offer(d)
    with pytest.raises(ValueError, match="Kaggle refused R7 "):
        submission.answer(d, SLUG, qid, "yes")
    (row,) = (d / submission.REFUSED_FILE).read_text().splitlines()
    assert json.loads(row)["round"] == 7 and "400" in json.loads(row)["reply"]


def test_a_steered_round_is_offered_even_when_it_did_not_beat_the_best(
    tmp_path: Path,
) -> None:
    from researchforge.kaggle import steering

    d = _project(tmp_path)
    # maximize: R2 is best (0.6); R3 followed the person's direction and scored 0.4;
    # R4 also followed it but failed.
    _log(d, [(1, 0.4, 1, 1), (2, 0.6, 1, 1), (3, 0.4, 1, 0), (4, None, 0, 0)], minimize=False)
    rl = steering.add(d, "try RL")
    steering.record_round(d, 3, {"follows": [rl.id]})
    steering.record_round(d, 4, {"follows": [rl.id]})
    for n in (2, 3, 4):
        f = tmp_path / f"r{n}.csv"
        f.write_text("id,y\n")
        submission.record_round_outputs(d, n, {"submission_file": str(f)})
    submission.offer_missed(d)
    offers = {
        submission.round_of(q.answer_key, q.question): q.question
        for q in control.open_interactions(d)
    }
    assert set(offers) == {2, 3}  # the best, and the steered round; never the failed one
    assert "followed your direction; did not beat the best: 0.4000 vs 0.6000 (R2)" in offers[3]
    for q in control.open_interactions(d):
        submission.answer(d, SLUG, q.id, "no")
    assert submission.offer_missed(d) is None  # asked once, even after "no"


def test_submit_now_sends_a_round_nobody_offered(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    d = _project(tmp_path)
    _log(d, [(1, 0.4, 1, 1), (2, 0.3, 1, 0)], minimize=False)
    calls = _fake_kaggle(monkeypatch, "Successfully submitted to comp")
    with pytest.raises(ValueError, match="nothing Kaggle takes"):
        submission.submit_now(d, SLUG, 2)
    f = tmp_path / "r2.csv"
    f.write_text("id,y\n")
    submission.record_round_outputs(d, 2, {"submission_file": str(f)})
    assert submission.submit_now(d, SLUG, 2).startswith("R2 submitted to Kaggle")
    assert calls[0][-1] == "RF R2: idea 2" and submission.submitted_rounds(d) == {2}
    (item,) = control.load_interactions(d)
    assert (item.source, item.answer) == ("you", "yes")


def test_the_submission_message_uses_the_rounds_title_not_the_offers_text(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    d = _project(tmp_path)
    _log(d, [(1, 0.4, 1, 1)], minimize=False)
    f = tmp_path / "r1.csv"
    f.write_text("id,y\n")
    submission.record_round_outputs(d, 1, {"submission_file": str(f)})
    calls = _fake_kaggle(monkeypatch, "Successfully submitted to comp")
    item = control.ask_submission(d, 1, "Submit R1 'idea 1' (it's 0.4; can't tell) to Kaggle?")
    submission.answer(d, SLUG, item.id, "yes")
    assert calls[0][-1] == "RF R1: idea 1"


def test_a_pending_round_whose_notebook_is_gone_is_closed_once_it_is_old(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import sqlite3

    from researchforge.kaggle import api, late_rounds

    d = _project(tmp_path)
    (d / "overnight_setup.json").write_text(
        json.dumps({"competition_slug": SLUG, "artifacts_dir": str(d), "data_dir": str(d)})
    )
    _log(d, [(1, 0.8, 1, 1), (2, None, 0, 0)], minimize=False)
    submission.record_round_outputs(d, 2, {"kernel_ref": "me/rf-r2", "pending": True})
    monkeypatch.setattr(api, "get_kernel_run_status", lambda ref: ("missing", "404"))
    assert late_rounds.collect(d) == []  # just pushed, Kaggle may not list it yet
    with sqlite3.connect(d / "experiment_log.db") as conn:
        conn.execute("UPDATE experiments SET started_at = '2026-01-01T00:00:00'")
    (line,) = late_rounds.collect(d)
    assert "no longer has the notebook me/rf-r2" in line
    assert late_rounds.pending_rounds(d) == {}
    assert control.open_interactions(d) == []  # nothing to offer


def test_kaggle_scores_are_matched_to_their_rounds(tmp_path: Path) -> None:
    d = _project(tmp_path)
    _log(d, [(14, 0.4, 1, 1), (24, 0.4, 1, 0), (25, 0.3, 1, 0), (37, 0.4, 1, 0)], minimize=False)

    def sub(desc: str, date: str, status: str, public: str = "", **kw: str) -> dict[str, str]:
        return {"description": desc, "date": date, "status": status, "publicscore": public, **kw}

    snap = {
        "submissions": [
            sub("RF R37: GRPO", "2026-11-19", "SubmissionStatus.PENDING"),
            sub("RF R25: v4 config", "2026-11-17", "SubmissionStatus.COMPLETE", "0.13"),
            sub("RF R24: bundle", "2026-11-16", "SubmissionStatus.COMPLETE"),
            sub(
                "RF R26: bundle",
                "2026-11-18",
                "SubmissionStatus.COMPLETE",
                errordescription="Your submission notebook exceeded the allowed runtime.",
            ),
            sub("CONTROL: resubmit of round_14", "2026-11-01", "SubmissionStatus.COMPLETE", "0.17"),
            sub("RF R14 again", "2026-11-20", "SubmissionStatus.PENDING"),
            sub("RF R9: x", "2026-11-02", "SubmissionStatus.ERROR", errorDescription="timeout"),
            sub("hand-made", "2026-11-03", "SubmissionStatus.COMPLETE", "0.9"),
        ]
    }
    (d / "submissions.json").write_text(json.dumps(snap))
    lb = submission.leaderboard_by_round(d)
    assert set(lb) == {9, 14, 24, 25, 26, 37}  # "hand-made" names no round
    # Kaggle lists a notebook that ran out of time as complete; its error text says so.
    assert (lb[26]["status"], lb[26]["error"]) == (
        "error",
        "Your submission notebook exceeded the allowed runtime.",
    )
    assert (lb[25]["public"], lb[25]["status"]) == (0.13, "scored")
    assert lb[24]["status"] == "no_score" and lb[37]["status"] == "pending"
    assert (lb[9]["status"], lb[9]["error"]) == ("error", "timeout")
    # R14's newer submission is still scoring: the score it already has stays.
    assert (lb[14]["public"], lb[14]["submissions"], lb[14]["latest"]) == (0.17, 2, "pending")
    table = submission.leaderboard_table(d)
    assert [r["round"] for r in table] == [37, 26, 25, 24, 14, 9]
    assert table[2] == {
        "round": 25,
        "title": "idea 25",
        "local": 0.3,
        "lb": 0.13,
        "lb_status": "scored",
    }

    from researchforge.kaggle.patch_proposer import PatchProposer

    proposer = PatchProposer.__new__(PatchProposer)
    proposer.setup = SimpleNamespace(artifacts_dir=d)
    text = proposer._leaderboard_section()
    assert "R25: local 0.3 -> LB 0.13 — idea 25" in text and "R24: local 0.4 -> LB no score" in text
    assert "R26: local — -> LB error (Your submission notebook exceeded" in text


def test_the_submission_listing_carries_kaggles_reason_for_a_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import subprocess

    from researchforge.kaggle import kernel_watcher

    csv_out = "ref,fileName,date,description,status,publicScore\n7,s.zip,2026,RF R26: x,COMPLETE,\n"
    reply = SimpleNamespace(returncode=0, stdout=csv_out, stderr="")
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: reply)
    errors = {"7": "exceeded the allowed runtime"}
    monkeypatch.setattr(kernel_watcher, "submission_errors", lambda slug: errors)
    rows, err = kernel_watcher.fetch_submissions(SLUG)
    assert err == "" and rows[0]["errordescription"] == "exceeded the allowed runtime"


TIMEOUT = "Your submission notebook exceeded the allowed runtime."


def _failed_on_kaggle(d: Path, *rounds: int, error: str = TIMEOUT) -> None:
    from datetime import datetime

    today = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    subs = [
        {
            "description": f"RF R{n}: x",
            "date": today,
            "status": "SubmissionStatus.COMPLETE",
            "publicscore": "",
            "errordescription": error,
        }
        for n in rounds
    ]
    (d / "submissions.json").write_text(json.dumps({"submissions": subs}))


def test_a_failed_submission_queues_one_fix_that_inherits_the_steer(tmp_path: Path) -> None:
    from researchforge.kaggle import fixes, steering

    d = _project(tmp_path)
    _log(d, [(3, 0.4, 1, 1), (5, 0.4, 1, 0)], minimize=False)
    rl = steering.add(d, "try RL")
    steering.record_round(d, 5, {"follows": [rl.id]})
    _failed_on_kaggle(d, 3, 5)
    (line,) = fixes.queue_fixes(d)
    assert line == "R5 timed out on Kaggle: fix round 1 queued first in Up next"
    head = steering.up_next(d)[0]
    assert head["kind"] == "fix" and head["fixes_round"] == 5
    assert (
        "R3 failed the same way" in head["text"]
        and "It continues your direction: try RL" in head["text"]
    )
    assert "FIX_CHECK: passed" in head["text"]
    assert fixes.queue_fixes(d) == []  # once per failure


def test_a_fix_is_offered_first_only_once_it_shows_the_error_is_gone(tmp_path: Path) -> None:
    from researchforge.kaggle import fixes, steering
    from researchforge.kaggle.benchmark_runner import declared_fix_check

    d = _project(tmp_path)
    _log(d, [(5, 0.6, 1, 1), (6, 0.5, 1, 0)], minimize=False)
    _failed_on_kaggle(d, 5)
    fixes.queue_fixes(d)
    fix = steering.up_next(d)[0]
    steering.record_round(d, 6, {"follows": [fix["id"]]})
    f = tmp_path / "r6.csv"
    f.write_text("id,y\n")
    submission.record_round_outputs(d, 6, {"submission_file": str(f)})
    assert fixes.unproven(d, 6)
    assert not any(q.answer_key == "round:6" for q in control.open_interactions(d))
    check = declared_fix_check("...\nFIX_CHECK: passed 41 min on 2x data, guard at 7h\n")
    assert check == {"passed": True, "evidence": "41 min on 2x data, guard at 7h"}
    submission.record_round_outputs(d, 6, {"submission_file": str(f), "fix_check": check})
    submission.offer_missed(d)
    (q,) = [q for q in control.open_interactions(d) if q.answer_key == "round:6"]
    assert "fixes R5 (timed out on Kaggle); check: 41 min on 2x data" in q.question
    assert fixes.queue_fixes(d) == []  # it passed: no second try


def test_a_fix_that_does_not_fix_it_is_tried_once_more_then_asks(tmp_path: Path) -> None:
    from researchforge.kaggle import fixes, steering

    d = _project(tmp_path)
    _log(d, [(5, 0.6, 1, 1), (6, 0.5, 1, 0), (7, 0.5, 1, 0)], minimize=False)
    _failed_on_kaggle(d, 5)
    fixes.queue_fixes(d)
    steering.record_round(d, 6, {"follows": [steering.up_next(d)[0]["id"]]})
    submission.record_round_outputs(
        d, 6, {"fix_check": {"passed": False, "evidence": "still 9h on 2x data"}}
    )
    (line,) = fixes.queue_fixes(d)
    assert "fix round 2 queued" in line
    second = steering.up_next(d)[0]
    assert "R6 did not fix it: still 9h on 2x data" in second["text"]
    steering.record_round(d, 7, {"follows": [second["id"]]})  # prints no FIX_CHECK at all
    (line,) = fixes.queue_fixes(d)
    assert line.endswith("asked you") and steering.up_next(d) == []
    (q,) = [q for q in control.open_interactions(d) if q.answer_key == "fix:5"]
    assert fixes.queue_fixes(d) == []  # asked once
    control.respond(d, q.id, "try another fix")
    (line,) = fixes.answered(d)
    assert "fix round 1 queued" in line and steering.up_next(d)[0]["fixes_round"] == 5
    assert fixes.answered(d) == []


def test_old_failures_are_left_alone(tmp_path: Path) -> None:
    from researchforge.kaggle import fixes

    d = _project(tmp_path)
    _log(d, [(5, 0.6, 1, 1)], minimize=False)
    snap = {
        "submissions": [
            {
                "description": "RF R5",
                "date": "2025-01-01 00:00:00",
                "status": "COMPLETE",
                "errordescription": TIMEOUT,
            }
        ]
    }
    (d / "submissions.json").write_text(json.dumps(snap))
    assert fixes.queue_fixes(d) == []


def test_the_monitor_restarts_itself_when_researchforge_is_updated(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import os

    from researchforge.kaggle import cli

    ran: list[list[str]] = []
    monkeypatch.setattr(os, "execv", lambda exe, argv: ran.append(argv))
    monkeypatch.setattr(cli, "_code_stamp", lambda: 100.0)
    cli._reload_if_updated(100.0)
    assert ran == []
    cli._reload_if_updated(50.0)
    assert len(ran) == 1  # the same command line, run again on the new code


def test_a_refused_notebook_submission_says_kaggles_reason(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from researchforge.kaggle import competition_reader

    d = _project(tmp_path, code=True)
    monkeypatch.setattr(submission, "is_code_competition", lambda art, slug: True)
    (tmp_path / "x.csv").write_text("id,y\n")
    submission.record_round_outputs(
        d,
        7,
        {"kernel_ref": "me/rf-r7", "kernel_version": 1, "submission_file": str(tmp_path / "x.csv")},
    )
    sent: list[tuple[str, ...]] = []
    body = json.dumps({"error": "400 Client Error", "body": "Notebook uses an input not allowed"})

    def api(snippet: str, *args: str, timeout: int = 60) -> str:
        sent.append(args)
        return "RF_ERR " + body

    monkeypatch.setattr(competition_reader, "run_on_kaggle_python", api)
    calls = _fake_kaggle(monkeypatch, "should not be called")
    with pytest.raises(ValueError, match="Notebook uses an input not allowed"):
        submission._submit(d, SLUG, 7, "idea")
    assert len(sent) == 1 and sent[0][:3] == (SLUG, "me/rf-r7", "1") and calls == []
    assert "Notebook uses an input not allowed" in (d / submission.REFUSED_FILE).read_text()


def test_an_offer_is_withdrawn_when_its_collection_found_no_file_and_kaggle_wont_answer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import sqlite3

    from researchforge.kaggle.benchmark_runner import NO_SUBMISSION_REASON

    d = _project(tmp_path, code=True)
    _log(d, [(52, None, 0, 0), (53, None, 0, 0)], minimize=False)
    with sqlite3.connect(d / "experiment_log.db") as conn:
        conn.execute(
            "UPDATE experiments SET rejected_reason=? WHERE round_num=52", (NO_SUBMISSION_REASON,)
        )
    for n in (52, 53):
        submission.record_round_outputs(d, n, {"kernel_ref": f"me/nb{n}", "kernel_version": 1})
        _offer(d, n)
    _notebook_output(monkeypatch, None)  # Kaggle rate-limits the listing (429)
    assert submission.withdraw_unsubmittable(d) == [52]  # R53's record says nothing either way


def test_a_refusal_while_kaggle_still_scores_the_last_one_says_so(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from datetime import UTC, datetime, timedelta

    d = _project(tmp_path)
    sent = (datetime.now(UTC) - timedelta(hours=21)).strftime("%Y-%m-%d %H:%M:%S")
    pending = {"description": "RF R65: bundle", "date": sent, "status": "SubmissionStatus.PENDING"}
    (d / "submissions.json").write_text(json.dumps({"submissions": [pending]}))
    assert submission.still_scoring(d) == "R65, submitted 21 h ago"
    f = tmp_path / "s.zip"
    f.write_text("x")
    submission.record_round_outputs(d, 70, {"submission_file": str(f)})
    _fake_kaggle(monkeypatch, "400 Client Error: Bad Request for url: …/CreateSubmission", code=1)
    with pytest.raises(ValueError, match="still scoring R65, submitted 21 h ago.*one submission"):
        submission._submit(d, SLUG, 70, "retry")
