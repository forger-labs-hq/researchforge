"""Repeated baseline measurement (`--n-runs`) and dropping one (`baseline reset`)."""

import json
from contextlib import closing
from pathlib import Path

from typer.testing import CliRunner

from researchforge.cli import app
from researchforge.execution.baseline import averaged_metrics, summarize_repeats
from researchforge.execution.metrics import MetricResult
from researchforge.storage.baseline_repository import (
    get_latest_baseline,
    get_latest_successful_baseline,
    list_baseline_runs,
)
from researchforge.storage.db import open_project_db


def _metrics(value: float, **secondary: float) -> MetricResult:
    return MetricResult.model_validate(
        {
            "schema_version": 1,
            "primary_metric": {"name": "f1", "value": value},
            "secondary_metrics": secondary,
        }
    )


class TestSummarizeRepeats:
    def test_a_single_measurement_gets_no_summary(self) -> None:
        assert summarize_repeats(1, [0.80], failed=0) is None

    def test_no_measurements_get_no_summary(self) -> None:
        assert summarize_repeats(3, [], failed=3) is None

    def test_the_mean_is_over_the_values_given(self) -> None:
        repeats = summarize_repeats(2, [0.80, 0.90], failed=0)
        assert repeats is not None
        assert abs(repeats.mean - 0.85) < 1e-12
        assert repeats.values == [0.80, 0.90]
        assert repeats.requested == 2

    def test_the_spread_is_the_sample_stdev(self) -> None:
        repeats = summarize_repeats(2, [0.84, 0.86], failed=0)
        assert repeats is not None
        # sqrt(((0.01)^2 + (0.01)^2)/1)
        assert repeats.stdev is not None and abs(repeats.stdev - 0.0141421356) < 1e-6

    def test_one_successful_repeat_has_no_spread(self) -> None:
        repeats = summarize_repeats(3, [0.80], failed=2)
        assert repeats is not None
        assert repeats.stdev is None
        assert repeats.coefficient_of_variation is None
        assert repeats.failed == 2

    def test_the_coefficient_of_variation_is_relative_to_the_mean(self) -> None:
        repeats = summarize_repeats(2, [0.84, 0.86], failed=0)
        assert repeats is not None
        assert repeats.coefficient_of_variation is not None
        assert abs(repeats.coefficient_of_variation - (repeats.stdev or 0) / 0.85) < 1e-12

    def test_failures_are_counted_alongside_the_mean(self) -> None:
        repeats = summarize_repeats(5, [0.80, 0.82, 0.84], failed=2)
        assert repeats is not None
        assert repeats.requested == 5
        assert repeats.failed == 2
        assert len(repeats.values) == 3


class TestAveragedMetrics:
    def test_the_primary_metric_is_averaged(self) -> None:
        result = averaged_metrics([_metrics(0.80), _metrics(0.90)])
        assert result.primary_metric.name == "f1"
        assert abs(result.primary_metric.value - 0.85) < 1e-12

    def test_a_secondary_metric_in_every_repeat_is_averaged(self) -> None:
        result = averaged_metrics(
            [_metrics(0.80, p95_latency_ms=100.0), _metrics(0.90, p95_latency_ms=200.0)]
        )
        assert result.secondary_metrics["p95_latency_ms"] == 150.0

    def test_a_secondary_metric_missing_from_one_repeat_is_dropped(self) -> None:
        result = averaged_metrics(
            [_metrics(0.80, p95_latency_ms=100.0, memory_mb=50.0), _metrics(0.90, memory_mb=70.0)]
        )
        assert "p95_latency_ms" not in result.secondary_metrics
        assert result.secondary_metrics["memory_mb"] == 60.0

    def test_one_repeat_averages_to_itself(self) -> None:
        result = averaged_metrics([_metrics(0.83, p95_latency_ms=120.0)])
        assert result.primary_metric.value == 0.83
        assert result.secondary_metrics["p95_latency_ms"] == 120.0


class TestBaselineRunWithRepeats:
    def test_a_single_run_records_no_repeats(
        self, cli_runner: CliRunner, contracted_project: Path
    ) -> None:
        result = cli_runner.invoke(app, ["baseline", "run"])
        assert result.exit_code == 0, result.output
        with closing(open_project_db()) as conn:
            latest = get_latest_baseline(conn)
        assert latest is not None
        assert latest.repeats is None

    def test_three_runs_are_one_baseline_holding_three_values(
        self, cli_runner: CliRunner, contracted_project: Path
    ) -> None:
        result = cli_runner.invoke(app, ["baseline", "run", "--n-runs", "3"])
        assert result.exit_code == 0, result.output

        with closing(open_project_db()) as conn:
            stored = list_baseline_runs(conn)
        assert len(stored) == 1, "repeats are one measurement, not three baselines"
        assert stored[0].repeats is not None
        assert stored[0].repeats.requested == 3
        assert len(stored[0].repeats.values) == 3

    def test_the_frozen_metric_is_the_mean_of_the_repeats(
        self, cli_runner: CliRunner, contracted_project: Path
    ) -> None:
        assert cli_runner.invoke(app, ["baseline", "run", "--n-runs", "3"]).exit_code == 0
        with closing(open_project_db()) as conn:
            latest = get_latest_successful_baseline(conn)
        assert latest is not None
        assert latest.metrics is not None
        assert latest.repeats is not None
        assert abs(latest.metrics.primary_metric.value - latest.repeats.mean) < 1e-12

    def test_each_repeat_keeps_its_own_logs(
        self, cli_runner: CliRunner, contracted_project: Path
    ) -> None:
        assert cli_runner.invoke(app, ["baseline", "run", "--n-runs", "2"]).exit_code == 0
        with closing(open_project_db()) as conn:
            latest = get_latest_baseline(conn)
        assert latest is not None
        run_dir = Path(latest.stdout_path).parent
        assert (run_dir / "stdout.log").is_file()
        assert (run_dir / "repeat-2" / "stdout.log").is_file()

    def test_the_output_reports_the_spread(
        self, cli_runner: CliRunner, contracted_project: Path
    ) -> None:
        result = cli_runner.invoke(app, ["baseline", "run", "--n-runs", "2"])
        assert result.exit_code == 0, result.output
        assert "Repeats:" in result.output
        assert "(mean)" in result.output

    def test_zero_runs_is_refused(
        self, cli_runner: CliRunner, contracted_project: Path
    ) -> None:
        result = cli_runner.invoke(app, ["baseline", "run", "--n-runs", "0"])
        assert result.exit_code != 0


class TestBaselineReset:
    def test_it_refuses_without_confirm(
        self, cli_runner: CliRunner, baselined_project: Path
    ) -> None:
        result = cli_runner.invoke(app, ["baseline", "reset"])
        assert result.exit_code == 1
        assert "--confirm" in result.output
        with closing(open_project_db()) as conn:
            assert get_latest_baseline(conn) is not None

    def test_it_drops_the_baseline_when_nothing_depends_on_it(
        self, cli_runner: CliRunner, baselined_project: Path
    ) -> None:
        result = cli_runner.invoke(app, ["baseline", "reset", "--confirm"])
        assert result.exit_code == 0, result.output
        with closing(open_project_db()) as conn:
            assert get_latest_baseline(conn) is None

    def test_resetting_with_no_baseline_says_so(
        self, cli_runner: CliRunner, contracted_project: Path
    ) -> None:
        result = cli_runner.invoke(app, ["baseline", "reset", "--confirm"])
        assert result.exit_code == 1
        assert "No baseline to reset" in result.output

    def test_experiments_are_blocked_again_after_a_reset(
        self, cli_runner: CliRunner, baselined_project: Path
    ) -> None:
        assert cli_runner.invoke(app, ["baseline", "reset", "--confirm"]).exit_code == 0
        result = cli_runner.invoke(app, ["baseline", "show"])
        assert result.exit_code == 1

    def test_it_refuses_when_experiments_were_measured_against_it(
        self, cli_runner: CliRunner, validated_project: Path
    ) -> None:
        result = cli_runner.invoke(app, ["baseline", "reset", "--confirm"])
        assert result.exit_code == 1
        assert "measured against this baseline" in result.output
        with closing(open_project_db()) as conn:
            assert get_latest_baseline(conn) is not None

    def test_force_drops_it_anyway(
        self, cli_runner: CliRunner, validated_project: Path
    ) -> None:
        result = cli_runner.invoke(app, ["baseline", "reset", "--confirm", "--force"])
        assert result.exit_code == 0, result.output
        with closing(open_project_db()) as conn:
            assert get_latest_baseline(conn) is None

    def test_forcing_leaves_the_experiment_records_alone(
        self, cli_runner: CliRunner, validated_project: Path
    ) -> None:
        from researchforge.storage.experiment_repository import list_experiments

        with closing(open_project_db()) as conn:
            before = [e.experiment_id for e in list_experiments(conn)]
        assert cli_runner.invoke(
            app, ["baseline", "reset", "--confirm", "--force"]
        ).exit_code == 0
        with closing(open_project_db()) as conn:
            after = [e.experiment_id for e in list_experiments(conn)]
        assert before == after

    def test_json_output_names_the_dependents(
        self, cli_runner: CliRunner, validated_project: Path
    ) -> None:
        result = cli_runner.invoke(
            app, ["baseline", "reset", "--confirm", "--force", "--json"]
        )
        assert result.exit_code == 0, result.output
        payload = json.loads(result.output)
        assert payload["removed"] >= 1
        assert payload["dependent_experiments"]
