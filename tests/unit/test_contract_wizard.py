from datetime import UTC, datetime

import pytest

from researchforge.contract.wizard import (
    FULL_COMMAND_PLACEHOLDER,
    build_draft_spec,
    guess_commands,
    guess_primary_metric,
)
from researchforge.domain.contract import ContractExecutionMode, MetricDirection, NetworkMode
from researchforge.domain.project import Project, ProjectMode
from researchforge.domain.repo_scan import CompatibilityStatus, GitInfo, PythonInfo, RepoScan


def _scan(**overrides: object) -> RepoScan:
    defaults: dict[str, object] = {
        "scan_id": "s1",
        "repo_path": "/tmp/repo",
        "git": GitInfo(is_repo=True, branch="develop"),
        "python": PythonInfo(has_pyproject=True),
        "suggested_editable_paths": ["src/"],
        "suggested_protected_paths": ["tests/", ".researchforge/"],
        "compatibility": CompatibilityStatus.READY,
        "scanned_at": datetime.now(UTC),
    }
    defaults.update(overrides)
    return RepoScan(**defaults)  # type: ignore[arg-type]


def _project(objective: str) -> Project:
    now = datetime.now(UTC)
    return Project(
        id="p1",
        name="demo",
        mode=ProjectMode.IMPROVE_REPOSITORY,
        objective=objective,
        created_at=now,
        updated_at=now,
    )


class TestMetricGuess:
    @pytest.mark.parametrize(
        ("objective", "name", "direction"),
        [
            ("Improve classification F1", "f1", MetricDirection.MAXIMIZE),
            ("Reduce P95 latency below 250ms", "latency_ms", MetricDirection.MINIMIZE),
            ("Cut inference cost by 20%", "average_cost_usd", MetricDirection.MINIMIZE),
            ("Improve accuracy on the benchmark", "accuracy", MetricDirection.MAXIMIZE),
            (
                "Improve F1 without increasing latency",  # maximize target wins
                "f1",
                MetricDirection.MAXIMIZE,
            ),
            ("Make the thing better somehow", "primary_metric", MetricDirection.MAXIMIZE),
        ],
    )
    def test_guesses(self, objective: str, name: str, direction: MetricDirection) -> None:
        metric = guess_primary_metric(objective)
        assert metric.name == name
        assert metric.direction is direction


class TestCommandGuess:
    def test_pyproject_setup(self) -> None:
        setup, _screening, _full = guess_commands(_scan())
        assert setup == "python -m pip install -e ."

    def test_requirements_setup(self) -> None:
        setup, _screening, _full = guess_commands(
            _scan(python=PythonInfo(requirements_files=["requirements.txt"]))
        )
        assert setup is not None
        assert "requirements.txt" in setup

    def test_benchmark_script_becomes_full_command(self) -> None:
        _setup, _screening, full = guess_commands(
            _scan(benchmark_candidates=["scripts/benchmark.py"])
        )
        assert full == "python scripts/benchmark.py"

    def test_placeholder_when_undetectable(self) -> None:
        _setup, _screening, full = guess_commands(_scan(benchmark_candidates=["benchmarks/"]))
        assert full == FULL_COMMAND_PLACEHOLDER


class TestDraftSpec:
    def test_draft_defaults(self) -> None:
        spec = build_draft_spec(_project("Improve accuracy"), _scan())

        assert spec.version == 1
        assert spec.project.mode is ProjectMode.IMPROVE_REPOSITORY
        assert spec.repository.baseline_ref == "develop"
        assert spec.execution.mode is ContractExecutionMode.AUTO
        assert spec.execution.trusted_repository is True  # default True for local venv runs
        assert spec.execution.timeout_minutes == 20
        assert spec.network.mode is NetworkMode.NONE
        assert spec.permissions.editable_paths == ["src/"]
        assert "tests/" in spec.permissions.protected_paths

    def test_requires_defined_project(self) -> None:
        now = datetime.now(UTC)
        undefined = Project(id="x", name="x", created_at=now, updated_at=now)
        with pytest.raises(ValueError):
            build_draft_spec(undefined, _scan())


class TestTargetValue:
    def test_no_target_is_never_invented_from_the_objective(self) -> None:
        spec = build_draft_spec(_project("Improve accuracy to 0.95"), _scan())
        assert spec.objective.primary_metric.target_value is None

    def test_explicit_target_is_carried_into_the_spec(self) -> None:
        spec = build_draft_spec(_project("Improve accuracy"), _scan(), target_value=0.95)
        assert spec.objective.primary_metric.target_value == 0.95

    def test_unset_target_renders_as_a_commented_hint(self) -> None:
        from researchforge.contract.render import render_contract_yaml

        yaml_text = render_contract_yaml(build_draft_spec(_project("Improve accuracy"), _scan()))
        assert "# target_value:" in yaml_text

    def test_set_target_renders_as_a_live_field(self) -> None:
        from researchforge.contract.render import render_contract_yaml

        yaml_text = render_contract_yaml(
            build_draft_spec(_project("Improve accuracy"), _scan(), target_value=0.9)
        )
        assert "    target_value: 0.9" in yaml_text
