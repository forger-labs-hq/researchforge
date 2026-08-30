"""Autorun loop logic: metric comparison, compound parenting, loop state."""

from datetime import UTC, datetime
from pathlib import Path

import pytest
import yaml

from researchforge.autorun.engine import (
    AutorunConfig,
    force_plan_parent,
    is_better,
    reaches_target,
    target_progress_pct,
)
from researchforge.autorun.state import (
    AutorunStateError,
    RoundRecord,
    load_state,
    new_state,
    resumable,
    save_state,
)
from researchforge.config.paths import autorun_state_path
from researchforge.domain.contract import MetricDirection

MAXIMIZE = MetricDirection.MAXIMIZE
MINIMIZE = MetricDirection.MINIMIZE


class TestMetricComparison:
    def test_higher_is_better_when_maximizing(self) -> None:
        assert is_better(0.76, 0.74, MAXIMIZE) is True

    def test_lower_is_not_better_when_maximizing(self) -> None:
        assert is_better(0.72, 0.74, MAXIMIZE) is False

    def test_lower_is_better_when_minimizing(self) -> None:
        assert is_better(11.0, 14.0, MINIMIZE) is True

    def test_higher_is_not_better_when_minimizing(self) -> None:
        assert is_better(16.0, 14.0, MINIMIZE) is False

    def test_equal_is_never_better(self) -> None:
        assert is_better(0.74, 0.74, MAXIMIZE) is False
        assert is_better(0.74, 0.74, MINIMIZE) is False


class TestTarget:
    def test_maximize_target_reached_on_exact_value(self) -> None:
        assert reaches_target(0.80, 0.80, MAXIMIZE) is True

    def test_maximize_target_not_reached_below(self) -> None:
        assert reaches_target(0.79, 0.80, MAXIMIZE) is False

    def test_minimize_target_reached_when_lower(self) -> None:
        assert reaches_target(9.5, 10.0, MINIMIZE) is True

    def test_minimize_target_not_reached_when_higher(self) -> None:
        assert reaches_target(10.5, 10.0, MINIMIZE) is False

    def test_progress_is_zero_at_baseline(self) -> None:
        assert target_progress_pct(0.70, 0.70, 0.80, MAXIMIZE) == 0.0

    def test_progress_is_halfway_at_midpoint(self) -> None:
        assert target_progress_pct(0.75, 0.70, 0.80, MAXIMIZE) == pytest.approx(50.0)

    def test_progress_caps_at_hundred_when_target_passed(self) -> None:
        assert target_progress_pct(0.90, 0.70, 0.80, MAXIMIZE) == 100.0

    def test_progress_never_negative_when_metric_regresses(self) -> None:
        assert target_progress_pct(0.60, 0.70, 0.80, MAXIMIZE) == 0.0

    def test_progress_counts_downward_for_minimize(self) -> None:
        assert target_progress_pct(12.0, 14.0, 10.0, MINIMIZE) == pytest.approx(50.0)


PLAN_WITHOUT_PARENT = """\
hypothesis_id: hyp-001
approach_summary: Two independent variants.
experiments:
  - key: conf
    title: Lower confidence threshold
    change_summary: CONF=0.001
    env_overrides:
      CONF_THRESHOLD: "0.001"
  - key: epochs
    title: More epochs
    change_summary: EPOCHS=10
    env_overrides:
      EPOCHS: "10"
"""

PLAN_WITH_PARENT = """\
hypothesis_id: hyp-001
approach_summary: One variant already chained.
experiments:
  - key: conf
    title: Lower confidence threshold
    change_summary: CONF=0.001
    parent: exp-004
    env_overrides:
      CONF_THRESHOLD: "0.001"
"""


class TestCompoundParenting:
    def test_parent_is_set_on_every_entry_that_lacks_one(self) -> None:
        result = yaml.safe_load(force_plan_parent(PLAN_WITHOUT_PARENT, "exp-007"))
        assert result["experiments"][0]["parent"] == "exp-007"
        assert result["experiments"][1]["parent"] == "exp-007"

    def test_existing_parent_is_never_overwritten(self) -> None:
        result = yaml.safe_load(force_plan_parent(PLAN_WITH_PARENT, "exp-007"))
        assert result["experiments"][0]["parent"] == "exp-004"

    def test_other_fields_survive_the_rewrite(self) -> None:
        result = yaml.safe_load(force_plan_parent(PLAN_WITHOUT_PARENT, "exp-007"))
        assert result["hypothesis_id"] == "hyp-001"
        assert result["experiments"][0]["env_overrides"] == {"CONF_THRESHOLD": "0.001"}

    def test_malformed_yaml_is_returned_untouched_for_the_importer_to_reject(self) -> None:
        malformed = "experiments: [unclosed"
        assert force_plan_parent(malformed, "exp-007") == malformed

    def test_document_without_experiments_list_is_returned_untouched(self) -> None:
        document = "hypothesis_id: hyp-001\n"
        assert force_plan_parent(document, "exp-007") == document


class TestAutorunConfigSettings:
    def test_settings_snapshot_is_json_encoded_strings(self) -> None:
        settings = AutorunConfig(stall=2, global_stall=3, max_hours=8.0).as_settings()
        assert settings["stall"] == "2"
        assert settings["max_hours"] == "8.0"
        assert settings["target_value"] == "null"


def _record(round_num: int) -> RoundRecord:
    return RoundRecord(
        round_num=round_num,
        hypotheses_planned=["hyp-001"],
        plan_ids=["plan-001"],
        experiments_run=2,
        promising=["exp-001"],
        rejected=["exp-002"],
        failed=[],
        best_experiment_id="exp-001",
        best_metric_value=0.76,
        improved=True,
        duration_seconds=42.0,
        completed_at=datetime.now(UTC),
    )


class TestAutorunState:
    def test_missing_state_file_loads_as_none(self, isolated_project_dir: Path) -> None:
        assert load_state() is None

    def test_saved_state_round_trips(self, isolated_project_dir: Path) -> None:
        state = new_state({"stall": "2"})
        state = state.model_copy(
            update={"rounds": [_record(1)], "rounds_completed": 1, "global_stall_count": 0}
        )
        save_state(state)

        loaded = load_state()
        assert loaded is not None
        assert loaded.rounds_completed == 1
        assert loaded.rounds[0].best_experiment_id == "exp-001"
        assert loaded.settings == {"stall": "2"}

    def test_save_writes_into_the_project_directory(self, isolated_project_dir: Path) -> None:
        save_state(new_state({}))
        assert autorun_state_path().is_file()
        assert autorun_state_path().name == "autorun.json"

    def test_corrupt_state_is_refused_rather_than_guessed(
        self, isolated_project_dir: Path
    ) -> None:
        path = autorun_state_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{not json", encoding="utf-8")
        with pytest.raises(AutorunStateError):
            load_state()

    def test_unknown_fields_are_refused(self, isolated_project_dir: Path) -> None:
        path = autorun_state_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            '{"started_at": "2026-01-01T00:00:00Z", "updated_at": '
            '"2026-01-01T00:00:00Z", "surprise": 1}',
            encoding="utf-8",
        )
        with pytest.raises(AutorunStateError):
            load_state()

    def test_running_state_is_resumable(self) -> None:
        assert resumable(new_state({})) is True

    def test_stopped_state_is_not_resumable(self) -> None:
        assert resumable(new_state({}).model_copy(update={"status": "stopped"})) is False

    def test_absent_state_is_not_resumable(self) -> None:
        assert resumable(None) is False
