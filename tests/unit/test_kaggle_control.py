"""Human control of a running overnight loop: pause/resume, questions, compute, settings."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from researchforge.cli import app
from researchforge.kaggle import control
from researchforge.kaggle.compute_policy import ComputePolicy, DataRule
from researchforge.kaggle.human_questions import Question, UserAnswers, ask_mid_loop_questions

SLUG = "demo-comp"


@pytest.fixture
def overnight(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: home))
    d = tmp_path / f".overnight_{SLUG}"
    d.mkdir()
    (d / "overnight_setup.json").write_text(
        json.dumps(
            {
                "competition_slug": SLUG,
                "data_dir": str(tmp_path / "data"),
                "artifacts_dir": str(d),
                "selected_gpu": "kaggle_gpu",
                "baseline_result": None,
                "user_answers": {"answers": {"external_data": "no"}, "asked_at": ""},
            }
        )
    )
    monkeypatch.setenv("RESEARCHFORGE_OVERNIGHT_DIR", str(d))
    monkeypatch.delenv("NTFY_TOPIC", raising=False)
    return d


def _run(*args: str) -> tuple[int, str]:
    result = CliRunner().invoke(app, ["kaggle", *args])
    return result.exit_code, result.output


# ── pause / resume ──────────────────────────────────────────────────────────


def test_pause_stops_prepare_round_and_execute_round(overnight: Path) -> None:
    code, out = _run("pause", SLUG, "--reason", "checking a leak")
    assert code == 0 and "Paused" in out
    assert control.LoopControl.load(overnight).paused

    code, out = _run("prepare-round", SLUG)
    assert code == 0
    payload = json.loads(out)
    assert payload["paused"] is True
    assert payload["pause_reason"] == "checking a leak"
    assert not (overnight / "pending_round_context.json").exists()

    (overnight / "proposal.json").write_text(json.dumps({"title": "x", "code_changes": ""}))
    code, _ = _run("execute-round", SLUG)
    assert code == 1
    assert (overnight / "proposal.json").exists()  # nothing consumed
    assert not (overnight / "execute_round.lock").exists()

    code, out = _run("resume", SLUG)
    assert code == 0 and "Resumed" in out
    assert not control.LoopControl.load(overnight).paused


def test_run_parallel_refuses_while_paused(overnight: Path) -> None:
    control.pause(overnight)
    code, _ = _run("run-parallel", SLUG)
    assert code == 1


# ── non-blocking questions ──────────────────────────────────────────────────


def test_request_is_non_blocking_and_deduplicated(overnight: Path) -> None:
    a = control.request_interaction(
        overnight, "Is external data allowed?", kind="domain_question", blocks=["use COCONUT"]
    )
    b = control.request_interaction(overnight, "Is external data allowed?")
    assert a.id == b.id
    assert control.blocked_ideas(overnight) == {"use COCONUT"}


def test_respond_unblocks_and_feeds_answers_to_proposals(overnight: Path) -> None:
    q = control.request_interaction(
        overnight,
        "Is external data allowed?",
        kind="domain_question",
        options=["yes", "no"],
        blocks=["use COCONUT"],
        answer_key="external_data",
    )
    code, _ = _run("respond", SLUG, q.id, "maybe")
    assert code == 1  # not one of the options

    code, out = _run("respond", SLUG, q.id, "yes")
    assert code == 0 and "answered" in out
    assert control.blocked_ideas(overnight) == set()
    assert UserAnswers.load(overnight).get("external_data") == "yes"
    setup = json.loads((overnight / "overnight_setup.json").read_text())
    assert setup["user_answers"]["answers"]["external_data"] == "yes"

    code, _ = _run("respond", SLUG, q.id, "no")
    assert code == 1  # already answered


def test_mid_loop_questions_become_interactions(overnight: Path) -> None:
    ask_mid_loop_questions(
        [Question(id="leak", text="Is patient age leakage?", choices=["yes", "no"])],
        UserAnswers(),
        overnight,
    )
    open_qs = control.open_interactions(overnight)
    assert [q.answer_key for q in open_qs] == ["leak"]
    assert open_qs[0].options == ["yes", "no"]


def test_status_json_lists_questions_and_settings(overnight: Path) -> None:
    control.pause(overnight, "lunch")
    control.request_interaction(overnight, "Switch to local?", kind="compute_question")
    code, out = _run("status", SLUG, "--json")
    assert code == 0
    st = json.loads(out)
    assert st["paused"] is True and st["pause_reason"] == "lunch"
    assert st["compute"] == "kaggle_gpu"
    assert [q["question"] for q in st["open_questions"]] == ["Switch to local?"]
    assert st["settings"]["notification_level"] == "balanced"


# ── compute, ideas, settings ────────────────────────────────────────────────


def test_compute_switch_respects_data_policy(overnight: Path) -> None:
    ComputePolicy(
        target="kaggle_gpu",
        slug=SLUG,
        data=DataRule(
            location="kaggle_servers",
            experiment_code_path="/kaggle/input/",
            can_download_locally=False,
        ),
    ).save(overnight)
    code, out = _run("compute", SLUG, "local")
    assert code == 1 and "download" in out
    code, _ = _run("compute", SLUG, "colab")
    assert code == 0
    setup = json.loads((overnight / "overnight_setup.json").read_text())
    assert setup["selected_gpu"] == "colab_gpu"
    code, _ = _run("compute", SLUG, "quantum")
    assert code == 1


def test_add_hypothesis_goes_first_once(overnight: Path) -> None:
    pool_path = overnight / "pending_hypotheses.json"
    pool_path.write_text(json.dumps([{"idea": "old", "tested": False}]))
    assert _run("add-hypothesis", SLUG, "try focal loss")[0] == 0
    assert _run("add-hypothesis", SLUG, "try focal loss")[0] == 0
    pool = json.loads((overnight / "pending_hypotheses.json").read_text())
    assert [h["idea"] for h in pool] == ["try focal loss", "old"]
    assert pool[0]["source"] == "user"


def test_settings_validate_and_drive_notification_filter(overnight: Path) -> None:
    assert _run("settings", SLUG, "--notify-level", "loud")[0] == 1
    assert control.should_notify(overnight, "ideas") is False  # balanced default
    assert control.should_notify(overnight, "improvement") is True

    assert _run("settings", SLUG, "--notify-level", "minimal")[0] == 0
    assert control.should_notify(overnight, "improvement") is False
    assert control.should_notify(overnight, "candidate") is True

    assert _run("settings", SLUG, "--notify-level", "verbose")[0] == 0
    assert control.should_notify(overnight, "result") is True


def test_kaggle_gpu_daily_cap_blocks_execute_round(overnight: Path) -> None:
    assert _run("settings", SLUG, "--kaggle-gpu-daily-cap", "2")[0] == 0
    control.record_compute_use(overnight, "kaggle_gpu")
    assert not control.kaggle_gpu_cap_reached(overnight, "kaggle_gpu")
    control.record_compute_use(overnight, "kaggle_gpu")
    assert control.kaggle_gpu_cap_reached(overnight, "kaggle_gpu")
    assert not control.kaggle_gpu_cap_reached(overnight, "ssh_server")  # cap is Kaggle-only

    (overnight / "proposal.json").write_text(json.dumps({"title": "x", "code_changes": ""}))
    code, out = _run("execute-round", SLUG)
    assert code == 1 and "cap" in out


def test_unknown_values_in_control_file_fall_back(overnight: Path) -> None:
    (overnight / control.CONTROL_FILE).write_text(
        json.dumps({"notification_level": "loud", "submission_policy": "yolo", "extra": 1})
    )
    ctl = control.LoopControl.load(overnight)
    assert ctl.notification_level == "balanced"
    assert ctl.submission_policy == "balanced"


# ── submission policy ───────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("policy", "kept", "offered"),
    [
        ("aggressive", False, True),
        ("balanced", False, False),
        ("balanced", True, True),  # wherever it ran: a local round's kept file counts
        ("conservative", True, False),
    ],
)
def test_submission_offer_follows_policy(
    overnight: Path, policy: str, kept: bool, offered: bool
) -> None:
    from researchforge.kaggle import submission

    assert _run("settings", SLUG, "--submission-policy", policy)[0] == 0
    if kept:
        (overnight / "submission.csv").write_text("id,y\n")
        submission.record_round_outputs(
            overnight, 7, {"submission_file": str(overnight / "submission.csv")}
        )
    item = control.offer_submission(overnight, 7, "focal loss", 0.91, "cpu")
    assert (item is not None) is offered
    if item and not kept:
        assert item.kind == "submission_approval" and item.options == ["yes", "no"]
        code, out = _run("respond", SLUG, item.id, "yes")  # the round left nothing to submit
        assert code == 1 and "nothing was submitted" in out
        assert [q.id for q in control.open_interactions(overnight)] == [item.id]


# ── kernel watcher relevance ────────────────────────────────────────────────


def test_watcher_ideas_are_tagged_by_relation_to_the_experiment_path(overnight: Path) -> None:
    import sqlite3

    from researchforge.kaggle.kernel_watcher import idea_relevance

    with sqlite3.connect(overnight / "experiment_log.db") as c:
        c.execute(
            "CREATE TABLE experiments (round_num INTEGER PRIMARY KEY, patch_description TEXT, "
            "full_improved INTEGER)"
        )
        c.execute("INSERT INTO experiments VALUES (1, 'LightGBM with target encoding', 1)")
        c.execute("INSERT INTO experiments VALUES (2, 'Transformer encoder on raw spectra', 0)")
    assert idea_relevance(overnight, "Stronger target encoding folds")["relevance"] == (
        "builds_on_best"
    )
    assert idea_relevance(overnight, "transformer encoder spectra")["relevance"] == (
        "repeats_failure"
    )
    assert idea_relevance(overnight, "Pseudo-labelling unlabeled molecules")["relevance"] == (
        "new_direction"
    )
