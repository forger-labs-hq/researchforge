"""Install packaged Cursor rules into a repository's `.cursor/rules/`.

Rules are UX, not a security boundary — every gate they describe is enforced
by the Python engine regardless of what a rule (or the AI) says.
Installation is manifest-based so ResearchForge never overwrites or removes
Cursor configuration it does not own: a sha256 per installed file is recorded
in `.researchforge/cursor-rules-manifest.json`, and any file that no longer
matches its recorded hash is treated as user-owned.
"""

from __future__ import annotations

import hashlib
from enum import StrEnum
from importlib import resources
from pathlib import Path

from pydantic import BaseModel

from researchforge.config.paths import researchforge_dir

RULES_PACKAGE = "researchforge.cursor.rules"
MANIFEST_FILENAME = "cursor-rules-manifest.json"
CURSOR_RULES_DIR = Path(".cursor") / "rules"
GATEWAY_RULE_NAME = "researchforge"

# Written by `researchforge init --cursor`; always-on project-level hook that
# surfaces all the workflow rules to Cursor before a project is even started.
GATEWAY_RULE_CONTENT = """\
---
description: >-
  ResearchForge is active in this project. Entry point for any research or
  experimentation task — consult this rule whenever the user mentions research,
  papers, experiments, or improving a repository.
globs:
alwaysApply: true
---

# ResearchForge project

This is an initialized ResearchForge project. The full workflow is driven by
the `researchforge` CLI — your role is to run commands, read their `--json`
output, and guide the user through each stage.

Always check where things stand before suggesting a next step:

```bash
researchforge status --json
```

## Available rules

Reference these with `@rule-name` for detailed guidance on each stage:

| Rule | When to use |
|---|---|
| `@researchforge-start` | beginning or resuming any journey |
| `@researchforge-doctor` | checking dependencies / setup failures |
| `@researchforge-papers` | finding and reviewing literature |
| `@researchforge-landscape` | synthesizing papers into research directions |
| `@researchforge-hypotheses` | generating testable hypotheses |
| `@researchforge-baseline` | drafting the contract and running the baseline |
| `@researchforge-plan` | designing experiment variants and patches |
| `@researchforge-run` | executing the experiment funnel |
| `@researchforge-results` | reading and explaining results |
| `@researchforge-validate` | validating finalists with repeated runs |
| `@researchforge-ship` | shipping the winning result as a branch or PR |
| `@researchforge-paper` | building the research publication package |

## Rules

- The Python engine is the boundary: never work around a validation error, a
  protected path, or an approval gate — fix the artifact or ask the user.
- Approvals belong to the user: never pass `--yes` or type a confirmation
  unless the user explicitly approved that step in this conversation.
- Ground every summary in stored data: quote only numbers returned by
  `--json` output or files under `.researchforge/` — never invent metrics.
"""


class RuleAction(StrEnum):
    INSTALLED = "installed"
    UPDATED = "updated"
    UNCHANGED = "unchanged"
    SKIPPED_MODIFIED = "skipped_modified"
    REMOVED = "removed"
    LEFT_MODIFIED = "left_modified"
    MISSING = "missing"
    MODIFIED = "modified"


class RuleReport(BaseModel):
    rule: str
    action: RuleAction
    path: str


class InstallReport(BaseModel):
    rules_dir: str
    results: list[RuleReport]

    @property
    def conflicts(self) -> list[RuleReport]:
        return [r for r in self.results if r.action is RuleAction.SKIPPED_MODIFIED]


class RulesManifest(BaseModel):
    """sha256 of each installed .mdc file, keyed by rule name (no extension)."""

    version: int = 1
    hashes: dict[str, str] = {}


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def manifest_path(base: Path | None = None, user: bool = False) -> Path:
    if user:
        return Path.home() / ".cursor" / f"researchforge-{MANIFEST_FILENAME}"
    return researchforge_dir(base) / MANIFEST_FILENAME


def _rules_root(base: Path | None, user: bool) -> Path:
    if user:
        return Path.home() / ".cursor" / "rules"
    return (base if base is not None else Path.cwd()) / CURSOR_RULES_DIR


def load_manifest(base: Path | None = None, user: bool = False) -> RulesManifest:
    path = manifest_path(base, user)
    if not path.is_file():
        return RulesManifest()
    return RulesManifest.model_validate_json(path.read_text(encoding="utf-8"))


def save_manifest(manifest: RulesManifest, base: Path | None = None, user: bool = False) -> None:
    path = manifest_path(base, user)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(manifest.model_dump_json(indent=2) + "\n", encoding="utf-8")


def list_packaged_rules() -> dict[str, str]:
    """Rule name (no extension) -> .mdc content, from the wheel's packaged assets."""
    rules: dict[str, str] = {}
    root = resources.files(RULES_PACKAGE)
    for entry in root.iterdir():
        name = getattr(entry, "name", "")
        if name.endswith(".mdc"):
            rule_name = name[:-4]  # strip .mdc
            rules[rule_name] = entry.read_text(encoding="utf-8")
    return dict(sorted(rules.items()))


def install_rules(
    base: Path | None = None, force: bool = False, user: bool = False
) -> InstallReport:
    """Copy packaged rules into `.cursor/rules/`, never clobbering user edits.

    `user=True` installs into `~/.cursor/rules/` (available in every project
    on this machine) instead of the repository.

    Per rule: missing -> write; unchanged since our last install (manifest
    hash matches) -> update in place; anything else -> skip unless `force`.
    """
    rules_root = _rules_root(base, user)
    manifest = load_manifest(base, user)
    results: list[RuleReport] = []

    for name, content in list_packaged_rules().items():
        target = rules_root / f"{name}.mdc"
        packaged_hash = _sha256(content.encode("utf-8"))
        if target.is_file():
            current_hash = _sha256(target.read_bytes())
            if current_hash == packaged_hash:
                action = RuleAction.UNCHANGED
            elif current_hash == manifest.hashes.get(name) or force:
                action = RuleAction.UPDATED
            else:
                action = RuleAction.SKIPPED_MODIFIED
        else:
            action = RuleAction.INSTALLED

        if action in (RuleAction.INSTALLED, RuleAction.UPDATED):
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content, encoding="utf-8")
        if action is not RuleAction.SKIPPED_MODIFIED:
            manifest.hashes[name] = packaged_hash
        results.append(RuleReport(rule=name, action=action, path=str(target)))

    save_manifest(manifest, base, user)
    return InstallReport(rules_dir=str(rules_root), results=results)


def uninstall_rules(
    base: Path | None = None, force: bool = False, user: bool = False
) -> InstallReport:
    """Remove installed rules; user-modified files are left unless `force`."""
    rules_root = _rules_root(base, user)
    manifest = load_manifest(base, user)
    results: list[RuleReport] = []

    for name in list_packaged_rules():
        target = rules_root / f"{name}.mdc"
        recorded = manifest.hashes.get(name)
        if not target.is_file():
            action = RuleAction.MISSING
        elif _sha256(target.read_bytes()) == recorded or force:
            target.unlink()
            action = RuleAction.REMOVED
        else:
            action = RuleAction.LEFT_MODIFIED
        if action is not RuleAction.LEFT_MODIFIED:
            manifest.hashes.pop(name, None)
        results.append(RuleReport(rule=name, action=action, path=str(target)))

    save_manifest(manifest, base, user)
    return InstallReport(rules_dir=str(rules_root), results=results)


def rules_status(base: Path | None = None, user: bool = False) -> InstallReport:
    """Per-rule state: unchanged (as packaged), modified, or missing."""
    rules_root = _rules_root(base, user)
    results: list[RuleReport] = []
    for name, content in list_packaged_rules().items():
        target = rules_root / f"{name}.mdc"
        if not target.is_file():
            action = RuleAction.MISSING
        elif _sha256(target.read_bytes()) == _sha256(content.encode("utf-8")):
            action = RuleAction.UNCHANGED
        else:
            action = RuleAction.MODIFIED
        results.append(RuleReport(rule=name, action=action, path=str(target)))
    return InstallReport(rules_dir=str(rules_root), results=results)


def install_gateway(base: Path | None = None) -> RuleReport:
    """Write `.cursor/rules/researchforge.mdc` (alwaysApply: true) for this project.

    This is the cold-start hook: it is always in Cursor's context so the AI
    knows ResearchForge is present and which rules are available, even before
    any `.researchforge/` files or `researchforge.yaml` exist to trigger the
    glob-based rules.  Only writes to the project level (never `--user`).
    """
    rules_root = _rules_root(base, user=False)
    target = rules_root / f"{GATEWAY_RULE_NAME}.mdc"
    packaged_hash = _sha256(GATEWAY_RULE_CONTENT.encode("utf-8"))
    if target.is_file():
        if _sha256(target.read_bytes()) == packaged_hash:
            action = RuleAction.UNCHANGED
        else:
            action = RuleAction.UPDATED
    else:
        action = RuleAction.INSTALLED
    if action in (RuleAction.INSTALLED, RuleAction.UPDATED):
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(GATEWAY_RULE_CONTENT, encoding="utf-8")
    return RuleReport(rule=GATEWAY_RULE_NAME, action=action, path=str(target))


def uninstall_gateway(base: Path | None = None, force: bool = False) -> RuleReport:
    """Remove the project-level gateway rule if it is unmodified (or force)."""
    rules_root = _rules_root(base, user=False)
    target = rules_root / f"{GATEWAY_RULE_NAME}.mdc"
    if not target.is_file():
        return RuleReport(rule=GATEWAY_RULE_NAME, action=RuleAction.MISSING, path=str(target))
    if _sha256(target.read_bytes()) == _sha256(GATEWAY_RULE_CONTENT.encode("utf-8")) or force:
        target.unlink()
        return RuleReport(rule=GATEWAY_RULE_NAME, action=RuleAction.REMOVED, path=str(target))
    return RuleReport(rule=GATEWAY_RULE_NAME, action=RuleAction.LEFT_MODIFIED, path=str(target))
