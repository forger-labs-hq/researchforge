"""Deterministic YAML rendering of a contract draft, with guidance comments."""

from __future__ import annotations

import json

from researchforge.domain.contract import ContractSpec


def _q(value: str) -> str:
    """YAML-safe scalar via JSON quoting (JSON strings are valid YAML)."""
    return json.dumps(value)


def _str_list(items: list[str], indent: str = "  ") -> list[str]:
    return [f"{indent}- {_q(item)}" for item in items]


def render_contract_yaml(spec: ContractSpec) -> str:
    e = spec.execution
    p = spec.permissions
    lines: list[str] = [
        "# ResearchForge experiment contract",
        "# Edit as needed, then run:",
        "#   researchforge contract validate",
        "#   researchforge contract approve",
        "",
        "version: 1",
        "",
        "project:",
        f"  name: {_q(spec.project.name)}",
        f"  mode: {spec.project.mode.value}",
        "",
        "objective:",
        f"  description: {_q(spec.objective.description)}",
        "  primary_metric:",
        f"    name: {_q(spec.objective.primary_metric.name)}",
        f"    direction: {spec.objective.primary_metric.direction.value}  # maximize | minimize",
    ]

    target = spec.objective.primary_metric.target_value
    if target is None:
        lines.append(
            "    # target_value: 0.85   # optional — `autorun` stops once the metric reaches it"
        )
    else:
        lines.append(
            f"    target_value: {target}  # `autorun` stops once the metric reaches this"
        )

    if spec.objective.hard_constraints:
        lines.append("  hard_constraints:")
        for constraint in spec.objective.hard_constraints:
            lines += [
                f"    - name: {_q(constraint.name)}",
                f"      operator: {_q(constraint.operator.value)}",
                f"      value: {constraint.value}",
            ]
    else:
        lines += [
            "  hard_constraints: []",
            "  # hard_constraints:",
            "  #   - name: p95_latency_ms",
            '  #     operator: "<="',
            "  #     value: 250",
        ]

    if spec.objective.secondary_metrics:
        lines.append("  secondary_metrics:")
        lines += _str_list(spec.objective.secondary_metrics, "    ")
    else:
        lines.append("  secondary_metrics: []")

    lines += [
        "",
        "repository:",
        f"  baseline_ref: {_q(spec.repository.baseline_ref)}",
        "",
        "execution:",
        f"  mode: {e.mode.value}  # auto | docker | venv",
        "  # ⚡ Docker strongly recommended for public/cloned repos (e.g. from GitHub).",
        "  #    Set mode: docker and run: researchforge generate dockerfile",
        "  #    Docker avoids ALL setup_command fragility and isolates your machine.",
        "  # venv mode: faster for your own projects, but does NOT isolate code.",
        f"  trusted_repository: {str(e.trusted_repository).lower()}",
    ]
    if e.setup_command is not None:
        lines.append(f"  setup_command: {_q(e.setup_command)}")
    else:
        lines.append("  setup_command: null")
    if e.screening_command is not None:
        lines.append(f"  screening_command: {_q(e.screening_command)}")
    else:
        lines += [
            "  # screening_command: optional faster subset used by Phase 1C screening",
        ]
    lines += [
        f"  full_command: {_q(e.full_command)}",
        f"  result_file: {_q(e.result_file)}",
        f"  timeout_minutes: {e.timeout_minutes}",
        f"  cpu_limit: {e.cpu_limit:g}",
        f"  memory_mb: {e.memory_mb}",
        f"  max_experiments: {e.max_experiments}",
        "  # stall: 3  # stop after N consecutive non-improvements (comment out to run all)",
        "",
        "permissions:",
    ]
    lines.append("  editable_paths:")
    lines += _str_list(p.editable_paths, "    ")
    if p.protected_paths:
        lines.append("  protected_paths:")
        lines += _str_list(p.protected_paths, "    ")
    else:
        lines.append("  protected_paths: []")

    lines += [
        "",
        "network:",
        f"  mode: {spec.network.mode.value}  # none | enabled (enabling requires approval)",
        "",
        "secrets:",
    ]
    if spec.secrets.forward_environment_variables:
        lines.append("  forward_environment_variables:")
        lines += _str_list(spec.secrets.forward_environment_variables, "    ")
    else:
        lines += [
            "  forward_environment_variables: []",
            "  # forward_environment_variables:",
            "  #   - ANTHROPIC_API_KEY",
        ]

    lines += [
        "",
        "validation:",
        f"  repeat_finalists: {spec.validation.repeat_finalists}",
        f"  require_existing_tests: {str(spec.validation.require_existing_tests).lower()}",
        "",
        "shipping:",
        f"  allow_branch_creation: {str(spec.shipping.allow_branch_creation).lower()}",
        f"  allow_draft_pr: {str(spec.shipping.allow_draft_pr).lower()}",
        "",
    ]
    return "\n".join(lines)
