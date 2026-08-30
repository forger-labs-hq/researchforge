"""Contract lifecycle: generate draft, approve, drift detection."""

from __future__ import annotations

import hashlib
import sqlite3
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from researchforge.contract.render import render_contract_yaml
from researchforge.contract.validate import ContractValidation, validate_contract_file
from researchforge.contract.wizard import build_draft_spec
from researchforge.domain.contract import ExperimentContract
from researchforge.domain.project import ProjectStatus
from researchforge.project.service import touch_project_status
from researchforge.storage.contract_repository import (
    get_active_contract,
    insert_contract,
    next_contract_version,
)
from researchforge.storage.project_repository import get_project
from researchforge.storage.scan_repository import get_latest_scan


class ContractError(Exception):
    """A contract operation could not proceed; message is user-facing."""


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def resolve_git_ref(repo_root: Path, ref: str) -> str:
    """Resolve a ref to a commit sha, or raise ContractError."""
    try:
        result = subprocess.run(
            [
                "git",
                "-C",
                str(repo_root),
                "rev-parse",
                "--verify",
                "--end-of-options",
                f"{ref}^{{commit}}",
            ],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise ContractError(f"Could not run git to resolve ref {ref!r}: {exc}") from exc
    if result.returncode != 0:
        raise ContractError(
            f"repository.baseline_ref {ref!r} does not resolve to a commit: {result.stderr.strip()}"
        )
    return result.stdout.strip()


def generate_contract(
    conn: sqlite3.Connection,
    *,
    output: Path,
    force: bool,
    target_value: float | None = None,
) -> Path:
    project = get_project(conn)
    if project is None or project.mode is None or project.objective is None:
        raise ContractError("Define the project first: `researchforge project create`.")
    scan = get_latest_scan(conn)
    if scan is None:
        raise ContractError("Scan the repository first: `researchforge repo scan`.")
    if output.exists() and not force:
        raise ContractError(f"{output} already exists — use --force to overwrite.")

    repo_root = Path(project.repository.path) if project.repository.path else output.parent
    spec = build_draft_spec(project, scan, repo_root=repo_root, target_value=target_value)
    output.write_text(render_contract_yaml(spec), encoding="utf-8")
    return output


def validate_for_cli(conn: sqlite3.Connection, path: Path) -> ContractValidation:
    return validate_contract_file(path, project=get_project(conn), scan=get_latest_scan(conn))


def approve_contract(
    conn: sqlite3.Connection, *, path: Path, repo_root: Path
) -> tuple[ExperimentContract, bool]:
    """Approve the contract file; returns (contract, created).

    `created` is False when the file is byte-identical to the already
    approved version (no-op).
    """
    project = get_project(conn)
    if project is None:
        raise ContractError("No project found. Run `researchforge project create` first.")

    validation = validate_for_cli(conn, path)
    if not validation.ok or validation.spec is None:
        raise ContractError(
            "Contract is invalid — fix these first:\n  " + "\n  ".join(validation.errors)
        )

    digest = file_sha256(path)
    active = get_active_contract(conn)
    if active is not None and active.source_sha256 == digest:
        return active, False

    baseline_commit = resolve_git_ref(repo_root, validation.spec.repository.baseline_ref)
    now = datetime.now(UTC)
    contract = ExperimentContract(
        contract_id=uuid4().hex,
        contract_version=next_contract_version(conn, project.id),
        spec=validation.spec,
        source_sha256=digest,
        baseline_commit=baseline_commit,
        approved_at=now,
        created_at=now,
    )
    insert_contract(conn, project.id, contract)
    touch_project_status(conn, ProjectStatus.CONTRACTED)
    return contract, True


def check_contract_drift(conn: sqlite3.Connection, path: Path) -> bool:
    """True when the on-disk yaml differs from the approved snapshot."""
    active = get_active_contract(conn)
    if active is None or not path.is_file():
        return False
    return file_sha256(path) != active.source_sha256
