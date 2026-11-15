"""Lower-is-better competitions: direction is measured, shared, and correctable."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from types import SimpleNamespace

import pytest
from typer.testing import CliRunner

from researchforge.kaggle import api, competition_reader, dashboard_state

_SQUARES_BOARD = """\
Warning: Looks like you're using an outdated `kaggle` version
Next Page Token = CfDJ8Fw
teamId,teamName,submissionDate,score
1,"Cats, and boxes",2026-11-01 20:51:25,22.0544214515
2,Chris,2026-11-04 17:28:04,22.0871069170
3,Junie,2026-11-05 06:56:45,22.1835980754
"""


@pytest.fixture(autouse=True)
def _isolated_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: home))
    monkeypatch.delenv("RESEARCHFORGE_OVERNIGHT_DIR", raising=False)
    monkeypatch.delenv("RESEARCHFORGE_SCAN_ROOTS", raising=False)
    monkeypatch.setenv("RESEARCHFORGE_NO_DASHBOARD", "1")
    dashboard_state._LAUNCHD_CACHE.clear()


class TestMeasuredDirection:
    def test_leaderboard_order_gives_the_direction(self) -> None:
        assert api._direction_from_leaderboard_csv(_SQUARES_BOARD) == "minimize"
        auc = "teamId,teamName,submissionDate,score\n1,a,x,0.97\n2,b,x,0.95\n3,c,x,0.91\n"
        assert api._direction_from_leaderboard_csv(auc) == "maximize"
        solved = "teamId,teamName,submissionDate,score\n1,a,x,1.0\n2,b,x,1.0\n"
        assert api._direction_from_leaderboard_csv(solved) is None
        assert api._direction_from_leaderboard_csv("403 - Forbidden") is None

    def test_leaderboard_overrides_a_guess_and_says_so(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(competition_reader, "_fetch_kaggle_info", lambda slug: {})
        monkeypatch.setattr(competition_reader, "_fetch_public_page", lambda slug: "")
        monkeypatch.setattr(competition_reader, "_detect_code_competition", lambda s, t: False)
        monkeypatch.setattr(competition_reader, "_estimate_data_size", lambda slug: 0.0)

        monkeypatch.setattr(api, "leaderboard_direction", lambda slug: None)
        guessed = competition_reader.read_competition_metadata("squares", tmp_path / "a")
        assert (guessed.metric_direction, guessed.metric_direction_source) == (
            "maximize",
            "guess",
        )

        monkeypatch.setattr(api, "leaderboard_direction", lambda slug: "minimize")
        measured = competition_reader.read_competition_metadata("squares", tmp_path / "b")
        assert (measured.metric_direction, measured.metric_direction_source) == (
            "minimize",
            "leaderboard",
        )


class TestParsing:
    @pytest.mark.parametrize(
        ("stdout", "value"),
        [
            ("SCORE: 22.2501795", 22.2501795),
            ("score = -0.25", -0.25),
            ("Score: 22", 22.0),
            ("score: 1.5e-03", 0.0015),
        ],
    )
    def test_generic_patterns_read_signed_and_whole_numbers(
        self, stdout: str, value: float
    ) -> None:
        from researchforge.kaggle.benchmark_runner import BenchmarkRunner

        runner = BenchmarkRunner.__new__(BenchmarkRunner)
        runner.setup = SimpleNamespace(contract=None, competition_metadata=None)  # type: ignore[assignment]
        assert runner._parse_metric({"stdout_text": stdout}) == pytest.approx(value)

    @pytest.mark.parametrize(
        ("stdout", "value"),
        [
            ("loss 4.856317039 after 12.3456 s", None),  # a stray long decimal is no score
            ("training done\n0.8875\n", 0.8875),  # a line that is only a number
        ],
    )
    def test_only_a_bare_number_line_counts_without_a_label(
        self, stdout: str, value: float | None
    ) -> None:
        from researchforge.kaggle.benchmark_runner import BenchmarkRunner

        runner = BenchmarkRunner.__new__(BenchmarkRunner)
        runner.setup = SimpleNamespace(contract=None, competition_metadata=None)  # type: ignore[assignment]
        assert runner._parse_metric({"stdout_text": stdout}) == value

    def test_saved_unsigned_patterns_keep_the_minus_sign(self) -> None:
        from researchforge.kaggle.benchmark_runner import BenchmarkRunner

        runner = BenchmarkRunner.__new__(BenchmarkRunner)
        meta = SimpleNamespace(metric_extraction_patterns=[r"(?i)score[:\s=]+([0-9]+\.[0-9]+)"])
        runner.setup = SimpleNamespace(contract=None, competition_metadata=meta)  # type: ignore[assignment]
        assert runner._parse_metric({"stdout_text": "score: -0.75"}) == -0.75


def _project_dir(root: Path, direction: str = "maximize") -> Path:
    d = root / ".overnight_squares"
    d.mkdir(parents=True)
    contract = {"objective": {"primary_metric": {"name": "score", "direction": direction}}}
    (d / "overnight_setup.json").write_text(
        json.dumps({"competition_slug": "squares", "contract": contract})
    )
    (d / "competition_metadata.json").write_text(
        json.dumps(
            {"slug": "squares", "metric_direction": direction, "metric_direction_source": "guess"}
        )
    )
    return d


class TestCorrecting:
    def test_settings_flips_the_loop_and_the_dashboard_together(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from researchforge.cli import app
        from researchforge.kaggle.experiment_loop import OvernightExperimentLoop

        d = _project_dir(tmp_path)
        loop = OvernightExperimentLoop.__new__(OvernightExperimentLoop)
        loop.db_path = d / "experiment_log.db"
        loop._init_database()
        with sqlite3.connect(loop.db_path) as c:
            c.execute(
                "INSERT INTO experiments (round_num, patch_description, patch_code, "
                "full_passed, full_metric, full_improved) VALUES (1, 'a', '', 1, 6.68, 1)"
            )
        monkeypatch.chdir(tmp_path)

        p = next(p for p in dashboard_state.build_state(tmp_path)["projects"])
        assert p["overnight"]["competition"]["direction_source"] == "guess"

        out = CliRunner().invoke(
            app, ["kaggle", "settings", "squares", "--metric-direction", "minimize"]
        )
        assert out.exit_code == 0, out.output
        assert "maximize → minimize" in out.output and "1 finished round(s)" in out.output
        setup = json.loads((d / "overnight_setup.json").read_text())
        assert setup["contract"]["objective"]["primary_metric"]["direction"] == "minimize"
        meta = json.loads((d / "competition_metadata.json").read_text())
        assert meta["metric_direction_source"] == "user"

        p = next(p for p in dashboard_state.build_state(tmp_path)["projects"])
        assert p["overnight"]["competition"]["direction"] == "minimize"
        assert p["experiments"][0]["metric_direction"] == "minimize"

        bad = CliRunner().invoke(
            app, ["kaggle", "settings", "squares", "--metric-direction", "down"]
        )
        assert bad.exit_code == 1

    def test_dashboard_follows_the_contract_the_loop_judges_by(self, tmp_path: Path) -> None:
        d = _project_dir(tmp_path, "minimize")
        (d / "competition_metadata.json").write_text(json.dumps({"metric_direction": "maximize"}))
        p = next(p for p in dashboard_state.build_state(tmp_path)["projects"])
        assert p["overnight"]["competition"]["direction"] == "minimize"


def test_watcher_alerts_on_a_lower_score_when_lower_is_better(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from researchforge.kaggle import kernel_watcher
    from researchforge.kaggle.api import PublicKernel

    d = _project_dir(tmp_path, "minimize")
    kernel_watcher.WatcherConfig(competition_slug="squares", interval_hours=6).save(d)
    kernels = [
        PublicKernel(
            ref="a/worse", title="Packing | LB 22.40", author="a", last_run_time="", total_votes=0
        ),
        PublicKernel(
            ref="b/better", title="Packing | LB 22.10", author="b", last_run_time="", total_votes=0
        ),
    ]
    monkeypatch.setattr(api, "list_competition_kernels", lambda *a, **k: kernels)
    result = kernel_watcher.run_watch_cycle(d, None, baseline_score=22.25, dry_run=True)
    assert result.better_kernel is not None
    assert result.better_kernel["ref"] == "b/better"
    assert result.better_kernel["delta"] == pytest.approx(0.15)


def test_without_a_baseline_a_round_must_beat_the_best_so_far(tmp_path: Path) -> None:
    import sqlite3

    from researchforge.domain.contract import MetricDirection
    from researchforge.kaggle.benchmark_runner import BenchmarkRunner
    from researchforge.kaggle.experiment_loop import OvernightExperimentLoop

    loop = OvernightExperimentLoop.__new__(OvernightExperimentLoop)
    loop.db_path = tmp_path / "experiment_log.db"
    loop._init_database()

    def runner(direction: MetricDirection, cv: float | None) -> BenchmarkRunner:
        r = BenchmarkRunner.__new__(BenchmarkRunner)
        metric = SimpleNamespace(direction=direction)
        r.setup = SimpleNamespace(  # type: ignore[assignment]
            contract=SimpleNamespace(objective=SimpleNamespace(primary_metric=metric)),
            baseline_result=SimpleNamespace(cv_score=cv),
            artifacts_dir=tmp_path,
        )
        return r

    # No baseline and no earlier score: the first round is only the reference.
    assert not runner(MetricDirection.MINIMIZE, None)._metric_improved(22.25, 0.0)
    with sqlite3.connect(loop.db_path) as c:
        c.execute(
            "INSERT INTO experiments (round_num, patch_description, patch_code, full_passed, "
            "full_metric) VALUES (1, 'a', '', 1, 22.25), (2, 'b', '', 0, 0.0)"
        )
    assert runner(MetricDirection.MINIMIZE, None)._metric_improved(22.24, 0.0)
    assert not runner(MetricDirection.MINIMIZE, None)._metric_improved(22.26, 0.0)
    assert not runner(MetricDirection.MAXIMIZE, None)._metric_improved(
        0.1, 0.0
    )  # failed 0.0 ignored
    # With a baseline, it is the reference.
    assert runner(MetricDirection.MINIMIZE, 22.30)._metric_improved(22.25, 22.30)
    assert not runner(MetricDirection.MINIMIZE, 22.20)._metric_improved(22.25, 22.20)
    assert not runner(MetricDirection.MAXIMIZE, 0.9)._metric_improved(0.8, 0.9)
