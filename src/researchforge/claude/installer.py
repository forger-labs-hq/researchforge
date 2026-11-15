"""Install packaged Claude skills into a repository's `.claude/skills/`.

The skills are UX, not a security boundary — every gate they describe is
enforced by the Python engine regardless of what a skill (or Claude) says.
Installation is manifest-based so ResearchForge never overwrites or removes
Claude configuration it does not own: a sha256 per installed file is recorded
in `.researchforge/claude-skills-manifest.json`, and any file that no longer
matches its recorded hash is treated as user-owned.
"""

from __future__ import annotations

import hashlib
from enum import StrEnum
from importlib import resources
from pathlib import Path

from pydantic import BaseModel

from researchforge.config.paths import researchforge_dir

SKILLS_PACKAGE = "researchforge.claude.skills"
MANIFEST_FILENAME = "claude-skills-manifest.json"
CLAUDE_SKILLS_DIR = Path(".claude") / "skills"


class SkillAction(StrEnum):
    INSTALLED = "installed"
    UPDATED = "updated"
    UNCHANGED = "unchanged"
    SKIPPED_MODIFIED = "skipped_modified"
    REMOVED = "removed"
    LEFT_MODIFIED = "left_modified"
    MISSING = "missing"
    MODIFIED = "modified"


class SkillReport(BaseModel):
    skill: str
    action: SkillAction
    path: str


class InstallReport(BaseModel):
    skills_dir: str
    results: list[SkillReport]

    @property
    def conflicts(self) -> list[SkillReport]:
        return [r for r in self.results if r.action is SkillAction.SKIPPED_MODIFIED]


class SkillsManifest(BaseModel):
    """sha256 of each installed SKILL.md, keyed by skill name."""

    version: int = 1
    hashes: dict[str, str] = {}


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def manifest_path(base: Path | None = None, user: bool = False) -> Path:
    if user:
        # User-level installs have no project `.researchforge/`; keep the
        # manifest next to the user's Claude config instead.
        return Path.home() / ".claude" / f"researchforge-{MANIFEST_FILENAME}"
    return researchforge_dir(base) / MANIFEST_FILENAME


def _skills_root(base: Path | None, user: bool) -> Path:
    if user:
        return Path.home() / ".claude" / "skills"
    return (base if base is not None else Path.cwd()) / CLAUDE_SKILLS_DIR


def load_manifest(base: Path | None = None, user: bool = False) -> SkillsManifest:
    path = manifest_path(base, user)
    if not path.is_file():
        return SkillsManifest()
    return SkillsManifest.model_validate_json(path.read_text(encoding="utf-8"))


def save_manifest(manifest: SkillsManifest, base: Path | None = None, user: bool = False) -> None:
    path = manifest_path(base, user)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(manifest.model_dump_json(indent=2) + "\n", encoding="utf-8")


def list_packaged_skills() -> dict[str, str]:
    """Skill name -> SKILL.md content, from the wheel's packaged assets."""
    skills: dict[str, str] = {}
    root = resources.files(SKILLS_PACKAGE)
    for entry in root.iterdir():
        if entry.is_dir():
            skill_file = entry / "SKILL.md"
            if skill_file.is_file():
                skills[entry.name] = skill_file.read_text(encoding="utf-8")
    return dict(sorted(skills.items()))


def install_skills(
    base: Path | None = None, force: bool = False, user: bool = False
) -> InstallReport:
    """Copy packaged skills into `.claude/skills/`, never clobbering user edits.

    `user=True` installs into `~/.claude/skills/` (available in every project
    and Claude account on this machine) instead of the repository.

    Per skill: missing -> write; unchanged since our last install (manifest
    hash matches) -> update in place; anything else -> skip unless `force`.
    """
    skills_root = _skills_root(base, user)
    manifest = load_manifest(base, user)
    results: list[SkillReport] = []

    for name, content in list_packaged_skills().items():
        target = skills_root / name / "SKILL.md"
        packaged_hash = _sha256(content.encode("utf-8"))
        if target.is_file():
            current_hash = _sha256(target.read_bytes())
            if current_hash == packaged_hash:
                action = SkillAction.UNCHANGED
            elif current_hash == manifest.hashes.get(name) or force:
                action = SkillAction.UPDATED
            else:
                action = SkillAction.SKIPPED_MODIFIED
        else:
            action = SkillAction.INSTALLED

        if action in (SkillAction.INSTALLED, SkillAction.UPDATED):
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content, encoding="utf-8")
        if action is not SkillAction.SKIPPED_MODIFIED:
            manifest.hashes[name] = packaged_hash
        results.append(SkillReport(skill=name, action=action, path=str(target)))

    save_manifest(manifest, base, user)
    return InstallReport(skills_dir=str(skills_root), results=results)


def refresh_installed_skills(base: Path | None = None, user: bool = False) -> InstallReport | None:
    """Bring a previous install up to date with the packaged skills.

    `pip install --upgrade` replaces the packaged skills but not the copies in
    `.claude/skills/`. Runs only where the user installed before (a manifest
    exists), and only writes when a skill is new in this version or is an
    unmodified older copy; user-edited files are never touched (no `force`).
    Returns None when nothing needed updating.
    """
    if not manifest_path(base, user).is_file():
        return None
    skills_root = _skills_root(base, user)
    manifest = load_manifest(base, user)
    stale = False
    for name, content in list_packaged_skills().items():
        target = skills_root / name / "SKILL.md"
        if not target.is_file():
            # New in this version -> install; recorded but gone -> the user removed it.
            stale = name not in manifest.hashes
        else:
            current = _sha256(target.read_bytes())
            stale = current != _sha256(content.encode("utf-8")) and current == manifest.hashes.get(
                name
            )
        if stale:
            break
    if not stale:
        return None
    return install_skills(base, force=False, user=user)


def uninstall_skills(
    base: Path | None = None, force: bool = False, user: bool = False
) -> InstallReport:
    """Remove installed skills; user-modified files are left unless `force`."""
    skills_root = _skills_root(base, user)
    manifest = load_manifest(base, user)
    results: list[SkillReport] = []

    for name in list_packaged_skills():
        target = skills_root / name / "SKILL.md"
        recorded = manifest.hashes.get(name)
        if not target.is_file():
            action = SkillAction.MISSING
        elif _sha256(target.read_bytes()) == recorded or force:
            target.unlink()
            if not any(target.parent.iterdir()):
                target.parent.rmdir()
            action = SkillAction.REMOVED
        else:
            action = SkillAction.LEFT_MODIFIED
        if action is not SkillAction.LEFT_MODIFIED:
            manifest.hashes.pop(name, None)
        results.append(SkillReport(skill=name, action=action, path=str(target)))

    save_manifest(manifest, base, user)
    return InstallReport(skills_dir=str(skills_root), results=results)


def skills_status(base: Path | None = None, user: bool = False) -> InstallReport:
    """Per-skill state: unchanged (as packaged), modified, or missing."""
    skills_root = _skills_root(base, user)
    results: list[SkillReport] = []
    for name, content in list_packaged_skills().items():
        target = skills_root / name / "SKILL.md"
        if not target.is_file():
            action = SkillAction.MISSING
        elif _sha256(target.read_bytes()) == _sha256(content.encode("utf-8")):
            action = SkillAction.UNCHANGED
        else:
            action = SkillAction.MODIFIED
        results.append(SkillReport(skill=name, action=action, path=str(target)))
    return InstallReport(skills_dir=str(skills_root), results=results)
