"""Path conventions for ResearchForge's local `.researchforge/` project state.

Phase 0 only needs the directory itself and the sqlite database inside it.
Subdirectories such as `worktrees/`, `artifacts/`, `papers/`, and `reports/`
(see the phased spec's workspace-isolation section) are the responsibility of
later-phase execution code to create lazily on first use — they are
intentionally not created here.
"""

from __future__ import annotations

from pathlib import Path

RESEARCHFORGE_DIR_NAME = ".researchforge"
DB_FILENAME = "researchforge.db"
CONFIG_FILENAME = "config.json"
SYNTHESIS_DIR_NAME = "synthesis"
REPORTS_DIR_NAME = "reports"
AUTORUN_STATE_FILENAME = "autorun.json"
RESEARCH_LOG_FILENAME = "research-log.md"


def researchforge_dir(base: Path | None = None) -> Path:
    """The `.researchforge/` directory for the given base path (default: cwd)."""
    root = base if base is not None else Path.cwd()
    return root / RESEARCHFORGE_DIR_NAME


def db_path(base: Path | None = None) -> Path:
    """Path to the sqlite database file inside `.researchforge/`."""
    return researchforge_dir(base) / DB_FILENAME


def is_initialized(base: Path | None = None) -> bool:
    """Whether `.researchforge/` and its database both exist at `base`."""
    root = researchforge_dir(base)
    return root.is_dir() and db_path(base).is_file()


def find_project_root(start: Path | None = None) -> Path | None:
    """The nearest initialized project at `start` or above (git-style walk-up)."""
    current = (start if start is not None else Path.cwd()).resolve()
    for candidate in (current, *current.parents):
        if is_initialized(candidate):
            return candidate
    return None


def config_path(base: Path | None = None) -> Path:
    """Path to the optional settings-override file inside `.researchforge/`."""
    return researchforge_dir(base) / CONFIG_FILENAME


def synthesis_dir(base: Path | None = None) -> Path:
    """Directory for the Claude<->CLI synthesis handshake files."""
    return researchforge_dir(base) / SYNTHESIS_DIR_NAME


def reports_dir(base: Path | None = None) -> Path:
    """Directory for generated reports."""
    return researchforge_dir(base) / REPORTS_DIR_NAME


def contract_path(base: Path | None = None) -> Path:
    """The user-owned experiment contract at the repository root."""
    root = base if base is not None else Path.cwd()
    return root / "researchforge.yaml"


def worktrees_dir(base: Path | None = None) -> Path:
    """Directory holding baseline/experiment worktrees (created lazily)."""
    return researchforge_dir(base) / "worktrees"


def artifacts_dir(base: Path | None = None) -> Path:
    """Directory holding run artifacts (created lazily)."""
    return researchforge_dir(base) / "artifacts"


def experiments_dir(base: Path | None = None) -> Path:
    """Staging directory for the experiment-plan handshake files."""
    return researchforge_dir(base) / "experiments"


def experiment_artifacts_dir(base: Path | None = None) -> Path:
    """Directory holding experiment run artifacts (created lazily)."""
    return artifacts_dir(base) / "experiments"


def autorun_state_path(base: Path | None = None) -> Path:
    """Loop state for `researchforge autorun --resume`."""
    return researchforge_dir(base) / AUTORUN_STATE_FILENAME


def research_log_path(base: Path | None = None) -> Path:
    """The living research-log document read before every re-synthesis."""
    return researchforge_dir(base) / RESEARCH_LOG_FILENAME
