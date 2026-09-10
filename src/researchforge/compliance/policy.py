"""Load and validate .researchforge/policy.yaml.

The policy file is optional — when absent every gate runs in INFO-only mode
and nothing is blocked.  The schema is intentionally human-readable and easy
to diff in Git.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Annotated, Literal

import yaml
from pydantic import BaseModel, Field

from researchforge.config.paths import researchforge_dir

POLICY_FILENAME = "policy.yaml"

# ── sub-sections ────────────────────────────────────────────────────────────


class DataPolicy(BaseModel):
    pii_allowed: bool = True
    secrets_allowed: bool = False
    allowed_data_paths: list[str] = Field(default_factory=list)
    denied_data_paths: list[str] = Field(default_factory=list)


class ModelsPolicy(BaseModel):
    allowed_providers: list[str] = Field(default_factory=list)
    denied_providers: list[str] = Field(default_factory=list)
    allowed_models: list[str] = Field(default=["*"])


class NetworkPolicy(BaseModel):
    default: Literal["deny", "allow"] = "allow"
    allowed_domains: list[str] = Field(default_factory=list)


class ShippingPolicy(BaseModel):
    human_approval_required: bool = False
    require_clean_worktree: bool = True
    require_all_blocking_gates: bool = True


class VulnerabilityThresholds(BaseModel):
    block_severity: list[str] = Field(
        default_factory=lambda: ["critical"],
        description="Severity levels that block shipping.",
    )


class SecurityPolicy(BaseModel):
    secret_scan: Literal["required", "optional", "disabled"] = "optional"
    dependency_scan: Literal["required", "optional", "disabled"] = "optional"
    sbom: Literal["required", "optional", "disabled"] = "optional"
    provenance: Literal["required", "optional", "disabled"] = "optional"
    vulnerabilities: VulnerabilityThresholds = Field(default_factory=VulnerabilityThresholds)


class GateTiming(BaseModel):
    run: Literal["post_experiment", "pre_validation", "pre_ship"] = "post_experiment"


class GatesPolicy(BaseModel):
    secret_scan: GateTiming = Field(default_factory=lambda: GateTiming(run="post_experiment"))
    sbom: GateTiming = Field(default_factory=lambda: GateTiming(run="pre_ship"))
    dependency_scan: GateTiming = Field(default_factory=lambda: GateTiming(run="pre_ship"))


class CommandsPolicy(BaseModel):
    allow: list[str] = Field(default_factory=list)
    deny: list[str] = Field(default_factory=list)


# ── root model ───────────────────────────────────────────────────────────────


class PolicySpec(BaseModel):
    """Validated representation of policy.yaml."""

    version: Annotated[int, Field(ge=1)] = 1
    data: DataPolicy = Field(default_factory=DataPolicy)
    models: ModelsPolicy = Field(default_factory=ModelsPolicy)
    network: NetworkPolicy = Field(default_factory=NetworkPolicy)
    shipping: ShippingPolicy = Field(default_factory=ShippingPolicy)
    security: SecurityPolicy = Field(default_factory=SecurityPolicy)
    gates: GatesPolicy = Field(default_factory=GatesPolicy)
    commands: CommandsPolicy = Field(default_factory=CommandsPolicy)
    protected_files: list[str] = Field(default_factory=list)


class LoadedPolicy(BaseModel):
    """A validated policy together with its provenance."""

    spec: PolicySpec
    path: str  # absolute path that was read
    raw_hash: str  # SHA-256 of the raw file bytes
    present: bool = True  # False → default/community mode


_DEFAULT_POLICY = PolicySpec()


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def load_policy(base: Path | None = None) -> LoadedPolicy:
    """Load policy.yaml from the .researchforge directory.

    When the file does not exist a permissive default is returned with
    ``present=False`` — no gates are blocked in community mode.
    """
    path = researchforge_dir(base) / POLICY_FILENAME
    if not path.exists():
        return LoadedPolicy(
            spec=_DEFAULT_POLICY,
            path=str(path),
            raw_hash="",
            present=False,
        )
    raw = path.read_bytes()
    data = yaml.safe_load(raw)
    spec = PolicySpec.model_validate(data or {})
    return LoadedPolicy(spec=spec, path=str(path), raw_hash=_sha256(raw))


def policy_path(base: Path | None = None) -> Path:
    return researchforge_dir(base) / POLICY_FILENAME
