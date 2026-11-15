"""Colab rounds that survive a dropped connection, the Colab-only rule, round time
limits, and phone alerts that reach the person's ntfy topic."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import pytest

from researchforge.kaggle import colab, control, round_target, run_config
from researchforge.kaggle.dashboard_state import _overnight_experiment
from researchforge.notifications import service

SLUG = "demo-comp"


def _overnight(root: Path, gpu: str = "colab_gpu", **extra: Any) -> Path:
    d = root / f".overnight_{SLUG}"
    d.mkdir(parents=True)
    setup = {
        "competition_slug": SLUG,
        "artifacts_dir": str(d),
        "data_dir": str(root / "data"),
        "selected_gpu": gpu,
        **extra,
    }
    (d / "overnight_setup.json").write_text(json.dumps(setup))
    return d


class FakeColab:
    """Google's Colab CLI, scripted: what each poll answers, and what was asked."""

    def __init__(self, polls: list[dict[str, Any] | None], log: str = "", sub: str = "") -> None:
        self.polls, self.log, self.sub = list(polls), log, sub
        self.calls: list[list[str]] = []
        self.env: list[str] = []

    def __call__(self, args: list[str], timeout: float) -> tuple[int | None, str, str]:
        self.calls.append(args)
        cmd = args[0]
        if cmd in ("new", "upload", "stop"):
            return 0, "", ""
        if cmd == "download":
            remote, local = args[-2], Path(args[-1])
            local.write_text(self.log if remote.endswith("out.log") else "PK-zip-bytes")
            return 0, "", ""
        assert cmd == "exec"
        code = Path(args[args.index("-f") + 1]).read_text()
        if "RF_STARTED" in code:
            self.env = [args[i + 1] for i, a in enumerate(args) if a == "--env"]
            return 0, "RF_STARTED 42\n", ""
        if "RF_POLL" in code:
            answer = self.polls.pop(0) if len(self.polls) > 1 else self.polls[0]
            if answer is None:  # the connection dropped on this check
                return 1, "", "RuntimeError: Connection was lost."
            return 0, "RF_POLL " + json.dumps(answer) + "\n", ""
        if "RF_SUB" in code:
            return 0, "RF_SUB " + json.dumps(self.sub) + "\n", ""
        if "RF_STOPPED" in code:
            return 0, "RF_STOPPED\n", ""
        raise AssertionError(code)

    def did(self, cmd: str) -> bool:
        return any(c[0] == cmd for c in self.calls)

    def stopped_round(self) -> bool:
        return any(
            c[0] == "exec" and "--timeout" in c and c[c.index("--timeout") + 1] == "60"
            for c in self.calls
        )


@pytest.fixture
def fake(monkeypatch: pytest.MonkeyPatch):
    def install(*args: Any, **kwargs: Any) -> FakeColab:
        f = FakeColab(*args, **kwargs)
        monkeypatch.setattr(colab, "_cli", f)
        monkeypatch.setattr(colab, "kaggle_env", lambda: {"KAGGLE_API_TOKEN": "test-token"})
        return f

    return install


def _run(tmp_path: Path, timeout: float = 3600.0) -> dict[str, Any]:
    return colab.run_round(
        "print('hi')",
        SLUG,
        colab.ColabConfig(gpu="L4"),
        tmp_path / "w",
        timeout,
        sleep=lambda _s: None,
    )


# ── the detached Colab runner ───────────────────────────────────────────────


class TestColabRound:
    def test_a_dropped_connection_costs_one_check_not_the_round(self, tmp_path, fake):
        f = fake(
            [None, {"done": False, "rc": None}, None, {"done": True, "rc": 0}],
            log="CV Score: 0.4100\n",
            sub="/content/rf_data/submission.zip",
        )
        out = _run(tmp_path)
        assert out["ok"] and out["exit_code"] == 0
        assert "CV Score: 0.4100" in out["stdout_text"]
        assert Path(out["submission_file"]).read_text() == "PK-zip-bytes"
        assert f.calls[-1][0] == "stop"
        assert f.env == ["KAGGLE_API_TOKEN=test-token"]
        assert [c[0] for c in f.calls[:2]] == ["new", "upload"]
        assert f.calls[0][-2:] == ["--gpu", "L4"]

    def test_a_round_past_the_time_limit_is_stopped_and_the_vm_released(self, tmp_path, fake):
        f = fake([{"done": False, "rc": None}], log="still going\n")
        out = _run(tmp_path, timeout=0.05)
        assert not out["ok"]
        assert "round time limit" in out["stderr_text"]
        assert f.stopped_round() and f.calls[-1][0] == "stop"
        assert out["submission_file"] == ""

    def test_lost_contact_gives_up_after_many_checks_and_releases_the_vm(self, tmp_path, fake):
        f = fake([None])
        out = _run(tmp_path)
        assert not out["ok"]
        assert "lost contact" in out["stderr_text"]
        polls = [c for c in f.calls if c[0] == "exec" and "--env" not in c]
        assert len(polls) >= colab.MAX_FAILED_POLLS
        assert f.calls[-1][0] == "stop"

    def test_a_failed_script_reports_its_output_and_fetches_no_submission(self, tmp_path, fake):
        f = fake([{"done": True, "rc": 1}], log="Traceback: boom\n", sub="/content/x.csv")
        out = _run(tmp_path)
        assert not out["ok"] and out["exit_code"] == 1
        assert "boom" in out["stderr_text"]
        assert out["submission_file"] == ""
        assert not any(c[0] == "download" and c[-2] == "/content/x.csv" for c in f.calls)

    def test_a_vm_colab_will_not_give_is_reported_without_a_session_to_stop(
        self, tmp_path, monkeypatch
    ):
        calls: list[list[str]] = []

        def cli(args, timeout):
            calls.append(args)
            return 1, "", "[colab] Allocation refused (precondition failed)."

        monkeypatch.setattr(colab, "_cli", cli)
        monkeypatch.setattr(colab, "kaggle_env", lambda: {"KAGGLE_API_TOKEN": "t"})
        out = _run(tmp_path)
        assert not out["ok"] and "Allocation refused" in out["stderr_text"]
        assert [c[0] for c in calls] == ["new"]

    def test_the_vm_snippets_are_valid_python(self):
        for code in (colab._START, colab._POLL, colab._STOP, colab._FIND_SUBMISSION):
            compile(code, "snippet", "exec")


class TestColabRoundKeepsItsSubmission:
    def test_the_benchmark_runner_keeps_the_file_so_the_round_can_be_offered(
        self, tmp_path, fake, monkeypatch
    ):
        from researchforge.kaggle.benchmark_runner import BenchmarkRunner
        from researchforge.kaggle.gpu_detector import GPUBackend

        fake([{"done": True, "rc": 0}], log="CV Score: 0.4100\n", sub="/content/rf_data/s.zip")
        monkeypatch.setattr(colab, "ready", lambda: (True, "signed in", ""))
        d = _overnight(tmp_path)
        runner = BenchmarkRunner.__new__(BenchmarkRunner)
        runner.gpu = GPUBackend.COLAB_GPU
        runner.setup = type(
            "S",
            (),
            {
                "artifacts_dir": d,
                "competition_slug": SLUG,
                "baseline_result": None,
                "contract": None,
                "metric_extraction_patterns": [],
            },
        )()
        monkeypatch.setattr(runner, "_write_patched_script", lambda code, tmp: _script(tmp))
        monkeypatch.setattr(runner, "_parse_metric", lambda r: 0.41)
        res = runner._run_colab_full("print(1)", 3600)
        assert res.passed and res.metric_value == 0.41
        kept = Path(res.outputs["submission_file"])
        assert kept.is_file() and kept.is_relative_to(d / "round_outputs")


def _script(tmp: Path) -> Path:
    p = tmp / "experiment.py"
    p.write_text("print(1)")
    return p


# ── where rounds run, and the Colab-only rule ──────────────────────────────


class TestRoundTarget:
    PUSH = "kaggle('kernels', 'push', '-p', str(work))"

    def test_a_colab_round_that_pushes_a_kaggle_notebook_is_refused(self, tmp_path):
        d = _overnight(tmp_path)
        for code in (self.PUSH, "!kaggle kernels push -p .", "api.kernels_push(folder)"):
            assert "Colab only" in round_target.refusal(d, code)
        assert round_target.refusal(d, "kaggle kernels output owner/nb -p out") == ""

    def test_other_compute_may_push_notebooks(self, tmp_path):
        d = _overnight(tmp_path, gpu="kaggle_gpu")
        assert round_target.refusal(d, self.PUSH) == ""

    def test_the_proposer_is_told_the_vm_the_limit_and_the_rule(self, tmp_path):
        d = _overnight(tmp_path, round_time_limit_minutes=120)
        colab.ColabConfig(plan="pro_plus", gpu="A100").save(d)
        b = round_target.brief(d)
        assert b["target"] == "colab_gpu" and b["time_limit_minutes"] == 120
        assert "A100" in b["runs_on"] and "Colab Pro+" in b["runs_on"]
        assert b["rules"][0].startswith("Colab only")
        assert "120 minutes" in b["note"]

    def test_execute_round_refuses_before_a_round_is_recorded(self, tmp_path, monkeypatch):
        from typer.testing import CliRunner

        from researchforge.cli import app

        d = _overnight(tmp_path)
        (d / "proposal.json").write_text(json.dumps({"title": "t", "code_changes": self.PUSH}))
        monkeypatch.setenv("RESEARCHFORGE_OVERNIGHT_DIR", str(d))
        monkeypatch.chdir(tmp_path)
        result = CliRunner().invoke(app, ["kaggle", "execute-round", SLUG])
        assert result.exit_code == 1, result.output
        assert "Colab only" in result.output or "Colab only" in (result.stderr or "")
        assert not (d / "experiment_log.db").exists()


# ── round time limit ────────────────────────────────────────────────────────


class TestRoundTimeLimit:
    def test_default_is_the_old_thirty_minutes(self, tmp_path):
        assert run_config.round_time_limit_seconds(_overnight(tmp_path)) == 1800

    def test_set_from_run_configs_and_logged(self, tmp_path):
        d = _overnight(tmp_path)
        assert run_config.set_field(d, "round_time_limit_minutes", "150", via="dashboard") == (
            30,
            150,
        )
        assert run_config.round_time_limit_seconds(d) == 150 * 60
        assert run_config.config_view(d)["round_time_limit_minutes"] == 150
        assert run_config.change_log(d)[0]["field"] == "round_time_limit_minutes"

    @pytest.mark.parametrize("bad", ["0", "2000", "soon"])
    def test_out_of_range_is_refused(self, tmp_path, bad):
        with pytest.raises(ValueError):
            run_config.set_field(_overnight(tmp_path), "round_time_limit_minutes", bad)


# ── phone alerts ────────────────────────────────────────────────────────────


class TestAlerts:
    @pytest.fixture
    def sent(self, monkeypatch):
        got: list[tuple[str, str]] = []

        def send(notif):
            got.append((os.environ.get("NTFY_TOPIC", ""), notif.message))
            return True

        monkeypatch.setattr(service, "send_notification", send)
        return got

    def test_questions_reach_the_machine_wide_topic(self, tmp_path, sent):
        d = _overnight(tmp_path)
        service.save_notify_config(topic="rf-phone-1234")
        assert control.notify_topic(d) == ("rf-phone-1234", "machine")
        assert control.notify(d, "interaction", "Submit R14?")
        assert sent == [("rf-phone-1234", "Submit R14?")]

    def test_a_project_topic_wins_over_the_machine_topic(self, tmp_path, sent):
        d = _overnight(tmp_path, ntfy_topic="rf-own")
        service.save_notify_config(topic="rf-phone-1234")
        assert control.notify_topic(d) == ("rf-own", "project")
        control.notify(d, "candidate", "R3 is better")
        assert sent[0][0] == "rf-own"

    def test_a_topic_given_as_a_setup_answer_is_where_alerts_go(self, tmp_path, sent):
        d = _overnight(tmp_path, user_answers={"answers": {"ntfy_topic": "rf-said-at-setup"}})
        service.save_notify_config(topic="rf-phone-1234")
        assert control.notify_topic(d) == ("rf-said-at-setup", "project")
        control.notify(d, "interaction", "Which GPU?")
        assert sent[0][0] == "rf-said-at-setup"

    def test_no_topic_means_nothing_is_sent_not_a_public_default(self, tmp_path, sent):
        d = _overnight(tmp_path)
        assert control.notify_topic(d) == ("", "")
        assert not control.notify(d, "interaction", "hello")
        assert sent == []
        assert service._ntfy_topic() == ""

    def test_machine_topic_set_and_cleared_from_run_configs(self, tmp_path):
        d = _overnight(tmp_path)
        run_config.set_field(d, "ntfy_machine_topic", "rf-new-topic", via="dashboard")
        assert service._load_notify_config()["ntfy_topic"] == "rf-new-topic"
        view = run_config.config_view(d)["notify"]
        assert view["topic"] == "rf-new-topic" and view["where"] == "machine"
        assert view["subscribe_url"].endswith("/rf-new-topic")
        assert "a round waiting for your approval to submit" in view["level_sends"]["minimal"]
        run_config.set_field(d, "ntfy_machine_topic", "", via="dashboard")
        assert "ntfy_topic" not in service._load_notify_config()

    def test_test_alert_goes_to_the_resolved_topic_whatever_the_level(self, tmp_path, sent):
        d = _overnight(tmp_path)
        with pytest.raises(ValueError, match="no ntfy topic"):
            run_config.send_test_alert(d)
        service.save_notify_config(topic="rf-phone-1234")
        control.LoopControl(notification_level="minimal").save(d)
        assert run_config.send_test_alert(d)["topic"] == "rf-phone-1234"
        assert sent[0][0] == "rf-phone-1234"


# ── small fixes ─────────────────────────────────────────────────────────────


class TestSplitOptions:
    def test_commas_inside_brackets_stay_in_the_answer(self):
        assert control.split_options(
            "Switch to local (runs run_benchmark.py here, which pushes Kaggle kernels),"
            "Keep colab,Pause the loop"
        ) == [
            "Switch to local (runs run_benchmark.py here, which pushes Kaggle kernels)",
            "Keep colab",
            "Pause the loop",
        ]

    def test_pipes_keep_every_comma_and_apostrophes_are_not_quotes(self):
        assert control.split_options("a, b | c") == ["a, b", "c"]
        assert control.split_options("it's fine,no") == ["it's fine", "no"]
        assert control.split_options("yes,no") == ["yes", "no"]


def test_a_failed_run_shows_no_score_instead_of_zero():
    row = {
        "round_num": 18,
        "full_passed": 0,
        "full_metric": 0.0,
        "screening_passed": 1,
        "screening_metric": 0.3,
        "rejected_reason": "Colab exit 1: Connection was lost.",
    }
    assert _overnight_experiment(row, 0.2, "maximize")["metric"] is None
    ok = {**row, "full_passed": 1, "full_metric": 0.41, "rejected_reason": None}
    assert _overnight_experiment(ok, 0.2, "maximize")["metric"] == 0.41
