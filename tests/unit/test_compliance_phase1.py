"""Unit tests for Phase 1 compliance engine."""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any

from researchforge.compliance.engine import run_compliance
from researchforge.compliance.gate_impls.human_approval import HumanApprovalGate
from researchforge.compliance.gate_impls.protected_assets import ProtectedAssetGate
from researchforge.compliance.gate_impls.provider_policy import ProviderPolicyGate
from researchforge.compliance.gates import GateContext
from researchforge.compliance.policy import LoadedPolicy, PolicySpec
from researchforge.compliance.ship_eligibility import can_ship
from researchforge.storage.compliance_repository import (
    get_approval,
    get_compliance_runs,
    insert_approval,
    insert_compliance_run,
    insert_waiver,
    list_waivers,
)
from researchforge.storage.db import ensure_schema

# ── helpers ──────────────────────────────────────────────────────────────────


def _memory_db() -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    ensure_schema(conn)
    return conn


def _default_policy(**overrides: Any) -> LoadedPolicy:
    spec = PolicySpec(**overrides)
    return LoadedPolicy(spec=spec, path="/fake/policy.yaml", raw_hash="abc123", present=True)


def _ctx(
    changed_files: list[str] | None = None,
    policy: LoadedPolicy | None = None,
    experiment_id: str = "exp-001",
) -> GateContext:
    return GateContext(
        project_id="proj-001",
        experiment_id=experiment_id,
        worktree="/tmp/worktree",
        repo_root="/tmp/repo",
        changed_files=changed_files or [],
        policy=policy or _default_policy(),
    )


# ── policy loading ────────────────────────────────────────────────────────────


def test_policy_defaults_when_no_file(tmp_path: Path) -> None:
    from researchforge.compliance.policy import load_policy

    policy = load_policy(tmp_path)
    assert not policy.present
    assert policy.raw_hash == ""
    assert policy.spec.version == 1


def test_policy_loads_yaml(tmp_path: Path) -> None:
    from researchforge.compliance.policy import load_policy

    rf_dir = tmp_path / ".researchforge"
    rf_dir.mkdir()
    (rf_dir / "policy.yaml").write_text("version: 1\nshipping:\n  human_approval_required: true\n")
    policy = load_policy(tmp_path)
    assert policy.present
    assert policy.spec.shipping.human_approval_required is True


# ── protected asset gate ──────────────────────────────────────────────────────


def test_protected_asset_gate_pass_no_changes() -> None:
    gate = ProtectedAssetGate()
    result = gate.evaluate(_ctx(changed_files=[]))
    assert result.status == "pass"


def test_protected_asset_gate_pass_safe_files() -> None:
    gate = ProtectedAssetGate()
    result = gate.evaluate(_ctx(changed_files=["src/model.py", "README.md"]))
    assert result.status == "pass"


def test_protected_asset_gate_blocks_researchforge_yaml() -> None:
    gate = ProtectedAssetGate()
    result = gate.evaluate(_ctx(changed_files=["researchforge.yaml"]))
    assert result.status == "fail"
    assert result.severity == "block"
    assert "researchforge.yaml" in result.message


def test_protected_asset_gate_blocks_policy_yaml() -> None:
    gate = ProtectedAssetGate()
    result = gate.evaluate(_ctx(changed_files=[".researchforge/policy.yaml"]))
    assert result.status == "fail"
    assert result.severity == "block"


def test_protected_asset_gate_blocks_policy_files() -> None:
    policy = _default_policy(protected_files=["benchmarks/evaluate.py"])
    gate = ProtectedAssetGate()
    result = gate.evaluate(_ctx(changed_files=["benchmarks/evaluate.py"], policy=policy))
    assert result.status == "fail"
    assert "benchmarks/evaluate.py" in result.message


def test_protected_asset_gate_allows_other_files() -> None:
    policy = _default_policy(protected_files=["benchmarks/evaluate.py"])
    gate = ProtectedAssetGate()
    result = gate.evaluate(_ctx(changed_files=["src/train.py"], policy=policy))
    assert result.status == "pass"


# ── provider policy gate ──────────────────────────────────────────────────────


def test_provider_gate_skips_when_no_policy() -> None:
    gate = ProviderPolicyGate(conn=None)
    policy = _default_policy()  # empty allowed/denied = no config
    result = gate.evaluate(_ctx(policy=policy))
    assert result.status == "skip"


def test_provider_gate_pass_when_provider_in_allowed(tmp_path: Path) -> None:
    from researchforge.compliance.policy import ModelsPolicy

    conn = _memory_db()
    # Insert a fake ai_call for the project
    conn.execute(
        "INSERT INTO ai_calls (call_id, project_id, purpose, provider, model, "
        "input_tokens, output_tokens, duration_seconds, created_at, estimated) "
        "VALUES ('c1', 'proj-001', 'plan', 'anthropic', 'claude-3', 100, 50, 1.0, '2026-01-01', 0)"
    )
    conn.commit()

    policy = _default_policy(
        models=ModelsPolicy(allowed_providers=["anthropic"], denied_providers=[])
    )
    gate = ProviderPolicyGate(conn=conn)
    ctx = _ctx(policy=policy)
    ctx.project_id = "proj-001"
    result = gate.evaluate(ctx)
    assert result.status == "pass"


def test_provider_gate_fail_when_provider_denied() -> None:
    from researchforge.compliance.policy import ModelsPolicy

    conn = _memory_db()
    conn.execute(
        "INSERT INTO ai_calls (call_id, project_id, purpose, provider, model, "
        "input_tokens, output_tokens, duration_seconds, created_at, estimated) "
        "VALUES ('c2', 'proj-002', 'plan', 'openai', 'gpt-4o', 100, 50, 1.0, '2026-01-01', 0)"
    )
    conn.commit()

    policy = _default_policy(models=ModelsPolicy(allowed_providers=[], denied_providers=["openai"]))
    gate = ProviderPolicyGate(conn=conn)
    ctx = GateContext(
        project_id="proj-002",
        experiment_id="exp-001",
        worktree="/tmp/wt",
        repo_root="/tmp/repo",
        changed_files=[],
        policy=policy,
    )
    result = gate.evaluate(ctx)
    assert result.status == "fail"
    assert result.severity == "block"
    assert "openai" in result.message


# ── human approval gate ───────────────────────────────────────────────────────


def test_approval_gate_skips_when_not_required() -> None:
    gate = HumanApprovalGate(conn=None)
    policy = _default_policy()  # human_approval_required defaults to False
    result = gate.evaluate(_ctx(policy=policy))
    assert result.status == "skip"


def test_approval_gate_fails_when_required_and_absent() -> None:
    conn = _memory_db()
    from researchforge.compliance.policy import ShippingPolicy

    policy = _default_policy(shipping=ShippingPolicy(human_approval_required=True))
    gate = HumanApprovalGate(conn=conn)
    result = gate.evaluate(_ctx(policy=policy))
    assert result.status == "fail"
    assert result.severity == "block"


def test_approval_gate_passes_when_approval_recorded() -> None:
    conn = _memory_db()
    insert_approval(conn, "proj-001", "exp-001", actor="alice", reason="looks good")

    from researchforge.compliance.policy import ShippingPolicy

    policy = _default_policy(shipping=ShippingPolicy(human_approval_required=True))
    gate = HumanApprovalGate(conn=conn)
    result = gate.evaluate(_ctx(policy=policy))
    assert result.status == "pass"
    assert "alice" in result.message


# ── compliance engine ─────────────────────────────────────────────────────────


def test_run_compliance_returns_compliance_run() -> None:
    cr = run_compliance(
        project_id="proj-001",
        experiment_id="exp-001",
        checkpoint="post_experiment",
        changed_files=[],
        worktree="/tmp/wt",
        repo_root="/tmp/repo",
        policy=_default_policy(),
    )
    assert cr.experiment_id == "exp-001"
    assert len(cr.results) > 0


def test_run_compliance_fails_on_protected_file() -> None:
    cr = run_compliance(
        project_id="proj-001",
        experiment_id="exp-001",
        checkpoint="post_experiment",
        changed_files=["researchforge.yaml"],
        worktree="/tmp/wt",
        repo_root="/tmp/repo",
        policy=_default_policy(),
    )
    assert not cr.passed
    blocking = cr.blocking_failures
    assert any("ASSET-001" in f.gate_id for f in blocking)


# ── ship eligibility ──────────────────────────────────────────────────────────


def test_can_ship_eligible() -> None:
    cr = run_compliance(
        project_id="p",
        experiment_id="exp-001",
        checkpoint="post_experiment",
        changed_files=[],
        worktree="/tmp",
        repo_root="/tmp",
        policy=_default_policy(),
    )
    decision = can_ship("exp-001", True, [cr], _default_policy())
    assert decision.eligible
    assert decision.status == "ship_eligible"


def test_can_ship_blocked_by_compliance_failure() -> None:
    cr = run_compliance(
        project_id="p",
        experiment_id="exp-001",
        checkpoint="post_experiment",
        changed_files=["researchforge.yaml"],
        worktree="/tmp",
        repo_root="/tmp",
        policy=_default_policy(),
    )
    decision = can_ship("exp-001", True, [cr], _default_policy())
    assert not decision.eligible
    assert decision.status == "not_eligible"
    assert any("ASSET-001" in r for r in decision.blocking_reasons)


def test_can_ship_not_eligible_when_not_validated() -> None:
    decision = can_ship("exp-001", False, [], _default_policy())
    assert not decision.eligible
    assert "technically validated" in decision.blocking_reasons[0]


def test_can_ship_ready_for_approval() -> None:
    from researchforge.compliance.policy import ShippingPolicy

    policy = _default_policy(shipping=ShippingPolicy(human_approval_required=True))
    decision = can_ship("exp-001", True, [], policy, conn=_memory_db())
    assert not decision.eligible
    assert decision.status == "ready_for_approval"
    assert decision.required_actions


def test_can_ship_eligible_after_approval() -> None:
    from researchforge.compliance.policy import ShippingPolicy

    conn = _memory_db()
    insert_approval(conn, "proj-001", "exp-001", actor="alice")
    policy = _default_policy(shipping=ShippingPolicy(human_approval_required=True))
    decision = can_ship("exp-001", True, [], policy, conn=conn)
    assert decision.eligible


# ── storage ───────────────────────────────────────────────────────────────────


def test_insert_and_retrieve_compliance_run() -> None:
    conn = _memory_db()
    cr = run_compliance(
        project_id="proj-001",
        experiment_id="exp-005",
        checkpoint="post_experiment",
        changed_files=[],
        worktree="/tmp",
        repo_root="/tmp",
        policy=_default_policy(),
    )
    insert_compliance_run(conn, "proj-001", cr)
    retrieved = get_compliance_runs(conn, "exp-005")
    assert len(retrieved) == 1
    assert retrieved[0].compliance_run_id == cr.compliance_run_id


def test_approval_round_trip() -> None:
    conn = _memory_db()
    approval_id = insert_approval(conn, "proj-1", "exp-002", "bob", "looks good")
    approval = get_approval(conn, "exp-002")
    assert approval is not None
    assert approval["actor"] == "bob"
    assert approval["approval_id"] == approval_id


def test_waiver_round_trip() -> None:
    conn = _memory_db()
    waiver_id = insert_waiver(
        conn,
        "proj-1",
        "exp-003",
        gate_id="DEP-004",
        reason="Vendor patch unavailable",
        actor="alice",
        expires_at="2027-01-01",
    )
    waivers = list_waivers(conn, "exp-003")
    assert len(waivers) == 1
    assert waivers[0]["gate_id"] == "DEP-004"
    assert waivers[0]["waiver_id"] == waiver_id


# ── experiment status ─────────────────────────────────────────────────────────


def test_new_statuses_exist() -> None:
    from researchforge.domain.experiment import ExperimentStatus

    assert ExperimentStatus.REJECTED_COMPLIANCE == "rejected_compliance"
    assert ExperimentStatus.REJECTED_SECURITY == "rejected_security"
    assert ExperimentStatus.READY_FOR_APPROVAL == "ready_for_approval"
    assert ExperimentStatus.SHIP_ELIGIBLE == "ship_eligible"


def test_validated_can_transition_to_ready_for_approval() -> None:
    from researchforge.domain.experiment import ExperimentStatus, advance

    result = advance(ExperimentStatus.VALIDATED, ExperimentStatus.READY_FOR_APPROVAL)
    assert result == ExperimentStatus.READY_FOR_APPROVAL


def test_validated_can_transition_to_rejected_compliance() -> None:
    from researchforge.domain.experiment import ExperimentStatus, advance

    result = advance(ExperimentStatus.VALIDATED, ExperimentStatus.REJECTED_COMPLIANCE)
    assert result == ExperimentStatus.REJECTED_COMPLIANCE
