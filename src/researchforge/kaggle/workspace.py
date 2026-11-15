"""Where Kaggle overnight state lives on disk — shared by producers and the dashboard.

Every overnight command (prepare-round, execute-round, monitor, the launchd
watcher, ...) and the dashboard must agree on *which* `.overnight_<slug>/`
directory belongs to a competition. Historically each command used
``Path.cwd() / f".overnight_{slug}"``, so running the watcher from one folder
and the experiment loop from another silently split one competition across
several directories — and the dashboard, keyed by slug, showed only one of them.

This module is the single contract:

* :func:`resolve_artifacts_dir` — producers call this instead of building the
  path from ``cwd``. It prefers a directory that was actually set up, and
  registers whatever it returns.
* The registry (``~/.config/researchforge/overnight_registry.json``) is keyed by
  *path*, so several locations for one slug are all remembered.
* :func:`discover_locations` — the dashboard's scan: registry + the core
  project registry + a bounded walk of every known workspace root, including
  nested repos and ``.researchforge/worktrees/*`` checkouts.

All JSON state files are written with :func:`atomic_write_json` so a reader
polling every few seconds never sees a half-written file.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

OVERNIGHT_PREFIX = ".overnight_"
REPO_DIR_NAME = ".researchforge"

# Directories never worth descending into when scanning for projects.
_SKIP_DIRS = {
    "node_modules",
    "__pycache__",
    "site-packages",
    "dist",
    "build",
    "venv",
    "env",
    "Library",
    "Applications",
}
_DEFAULT_SCAN_DEPTH = 4


def config_dir() -> Path:
    return Path.home() / ".config" / "researchforge"


def registry_path() -> Path:
    return config_dir() / "overnight_registry.json"


def dashboard_config_path() -> Path:
    return config_dir() / "dashboard.json"


# ---------------------------------------------------------------------------
# Atomic JSON IO
# ---------------------------------------------------------------------------


def atomic_write_json(path: Path, data: object) -> None:
    """Write JSON so concurrent readers see either the old or the new file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=2, default=str)
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def read_json(path: Path, default: Any = None) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return default


# ---------------------------------------------------------------------------
# Location classification
# ---------------------------------------------------------------------------


def slug_of(artifacts_dir: Path) -> str:
    return artifacts_dir.name[len(OVERNIGHT_PREFIX) :]


def location_role(artifacts_dir: Path) -> str:
    """primary | backup | worktree — copies are shown, but never counted as live work."""
    if f"/{REPO_DIR_NAME}/worktrees/" in artifacts_dir.as_posix():
        return "worktree"
    name = artifacts_dir.name.lower()
    if any(tag in name for tag in ("backup", ".bak", ".old", ".orig")):
        return "backup"
    return "primary"


def has_setup(artifacts_dir: Path) -> bool:
    return (artifacts_dir / "overnight_setup.json").is_file()


# ---------------------------------------------------------------------------
# Registry (keyed by path)
# ---------------------------------------------------------------------------


def load_registry() -> list[dict[str, Any]]:
    """Return registry entries ``{slug, artifacts_dir, registered_at, last_active_at}``.

    Reads the v1 format (``{slug: path}``) transparently.
    """
    raw = read_json(registry_path(), {})
    entries: list[dict[str, Any]] = []
    if isinstance(raw, dict) and raw.get("version") == 2:
        for e in raw.get("entries", []):
            if isinstance(e, dict) and e.get("artifacts_dir"):
                entries.append(e)
    elif isinstance(raw, dict):
        for slug, p in raw.items():
            if isinstance(p, str):
                entries.append(
                    {"slug": slug, "artifacts_dir": p, "registered_at": "", "last_active_at": ""}
                )
    return entries


def _save_registry(entries: list[dict[str, Any]]) -> None:
    atomic_write_json(registry_path(), {"version": 2, "entries": entries})


def register_overnight_dir(slug: str, artifacts_dir: Path) -> None:
    """Upsert ``artifacts_dir`` in the global registry and mark it active now."""
    try:
        resolved = str(artifacts_dir.resolve())
        now = datetime.now().isoformat(timespec="seconds")
        entries = load_registry()
        for e in entries:
            if e["artifacts_dir"] == resolved:
                e["last_active_at"] = now
                e["slug"] = slug
                break
        else:
            entries.append(
                {
                    "slug": slug,
                    "artifacts_dir": resolved,
                    "registered_at": now,
                    "last_active_at": now,
                }
            )
        _save_registry(entries)
    except OSError:
        pass  # bookkeeping must never break a producer


# ---------------------------------------------------------------------------
# Producer-side resolution
# ---------------------------------------------------------------------------


def resolve_artifacts_dir(slug: str, cwd: Path | None = None, *, register: bool = True) -> Path:
    """Return the `.overnight_<slug>/` directory a command should read and write.

    Order:
      1. ``$RESEARCHFORGE_OVERNIGHT_DIR`` when it names this slug (launchd jobs).
      2. A *set-up* dir in ``cwd`` or an ancestor (running from a project subdir).
      3. The most recently set-up registered primary dir for the slug.
      4. Any existing dir in ``cwd`` / ancestors (watcher-only state).
      5. ``cwd/.overnight_<slug>`` — a brand new project.
    """
    cwd = (cwd or Path.cwd()).resolve()
    name = f"{OVERNIGHT_PREFIX}{slug}"
    chosen: Path | None = None

    env = os.environ.get("RESEARCHFORGE_OVERNIGHT_DIR", "")
    if env and Path(env).name == name:
        chosen = Path(env)

    local: list[Path] = []
    if chosen is None:
        home = Path.home()
        for d in [cwd, *cwd.parents]:
            cand = d / name
            if cand.is_dir():
                local.append(cand)
            if d == home or d == d.parent:
                break
        chosen = next((d for d in local if has_setup(d)), None)

    if chosen is None:
        registered = [Path(e["artifacts_dir"]) for e in load_registry() if e.get("slug") == slug]
        set_up = [
            p for p in registered if p.is_dir() and has_setup(p) and location_role(p) == "primary"
        ]
        if set_up:
            chosen = max(set_up, key=lambda p: (p / "overnight_setup.json").stat().st_mtime)
            if chosen.parent != cwd:
                print(f"[researchforge] using registered project dir {chosen}", file=sys.stderr)

    if chosen is None:
        chosen = local[0] if local else cwd / name

    if register and chosen.is_dir():
        register_overnight_dir(slug, chosen)
    return chosen


# ---------------------------------------------------------------------------
# Dashboard-side discovery
# ---------------------------------------------------------------------------


@dataclass
class ProjectLocation:
    """One project root on disk: overnight dirs and/or a `.researchforge/` repo project."""

    root: Path
    overnight_dirs: list[Path] = field(default_factory=list)
    repo_dir: Path | None = None
    registry_name: str = ""
    found_by: set[str] = field(default_factory=set)


def _core_registry_entries() -> list[dict[str, Any]]:
    try:
        from researchforge.config.registry import load_registry as _load_core

        return [e.model_dump() for e in _load_core()]
    except Exception:
        return []


def _configured_scan_roots() -> list[Path]:
    roots: list[Path] = []
    cfg = read_json(dashboard_config_path(), {})
    if isinstance(cfg, dict):
        roots += [Path(p).expanduser() for p in cfg.get("scan_roots", []) if isinstance(p, str)]
    env = os.environ.get("RESEARCHFORGE_SCAN_ROOTS", "")
    roots += [Path(p).expanduser() for p in env.split(os.pathsep) if p]
    return roots


def _hidden_roots() -> set[str]:
    """Project roots the user hid from the dashboard (``hidden_roots`` in dashboard.json).

    Hiding only filters the view: nothing on disk is touched, and removing the
    entry brings the project back.
    """
    cfg = read_json(dashboard_config_path(), {})
    if not isinstance(cfg, dict):
        return set()
    hidden: set[str] = set()
    for p in cfg.get("hidden_roots", []):
        if isinstance(p, str):
            try:
                hidden.add(str(Path(p).expanduser().resolve()))
            except OSError:
                continue
    return hidden


def scan_roots(cwd: Path) -> list[Path]:
    """Every root worth walking: cwd, configured roots, and every registered project root."""
    roots: list[Path] = [cwd, *_configured_scan_roots()]
    for e in load_registry():
        p = Path(e["artifacts_dir"])
        roots.append(p.parent)
    for e in _core_registry_entries():
        roots.append(Path(e["path"]))
    unique: list[Path] = []
    seen: set[str] = set()
    for r in roots:
        try:
            r = r.resolve()
        except OSError:
            continue
        if not r.is_dir() or str(r) in seen:
            continue
        seen.add(str(r))
        unique.append(r)
    # Drop roots already covered by an ancestor root (the walk will reach them).
    return [r for r in unique if not any(o != r and o in r.parents for o in unique)]


def _walk(root: Path, depth: int, hits: list[Path], errors: list[str]) -> None:
    try:
        it = os.scandir(root)
    except OSError as exc:
        errors.append(f"cannot scan {root}: {exc.strerror or exc}")
        return
    with it:
        for entry in it:
            try:
                if not entry.is_dir(follow_symlinks=False):
                    continue
            except OSError:
                continue
            name = entry.name
            if name.startswith(OVERNIGHT_PREFIX):
                hits.append(Path(entry.path))
                continue
            if name == REPO_DIR_NAME:
                hits.append(Path(entry.path))
                worktrees = Path(entry.path) / "worktrees"
                if worktrees.is_dir():
                    for wt in worktrees.iterdir():
                        if wt.is_dir():
                            hits.extend(p for p in wt.glob(f"{OVERNIGHT_PREFIX}*") if p.is_dir())
                continue
            if name.startswith(".") or name in _SKIP_DIRS or depth <= 0:
                continue
            if os.path.exists(os.path.join(entry.path, "pyvenv.cfg")):
                continue
            _walk(Path(entry.path), depth - 1, hits, errors)


def discover_locations(
    cwd: Path | None = None, depth: int = _DEFAULT_SCAN_DEPTH
) -> tuple[list[ProjectLocation], dict[str, Any]]:
    """Find every overnight dir and repo project reachable from the known roots.

    Returns ``(locations, report)`` where ``report`` describes how discovery ran
    (roots, counts, errors) so an empty dashboard can explain itself.
    """
    cwd = (cwd or Path.cwd()).resolve()
    roots = scan_roots(cwd)
    hits: list[Path] = []
    errors: list[str] = []
    for root in roots:
        _walk(root, depth, hits, errors)

    by_root: dict[str, ProjectLocation] = {}

    def loc_for(root: Path) -> ProjectLocation:
        key = str(root)
        if key not in by_root:
            by_root[key] = ProjectLocation(root=root)
        return by_root[key]

    for hit in hits:
        hit = hit.resolve()
        loc = loc_for(hit.parent)
        loc.found_by.add("scan")
        if hit.name == REPO_DIR_NAME:
            loc.repo_dir = hit
        elif hit not in loc.overnight_dirs:
            loc.overnight_dirs.append(hit)

    missing_registered: list[str] = []
    for e in load_registry():
        p = Path(e["artifacts_dir"])
        if p.is_dir():
            loc = loc_for(p.resolve().parent)
            loc.found_by.add("registry")
            if p.resolve() not in loc.overnight_dirs:
                loc.overnight_dirs.append(p.resolve())
        else:
            missing_registered.append(str(p))
    for e in _core_registry_entries():
        root = Path(e["path"])
        if (root / REPO_DIR_NAME).is_dir():
            loc = loc_for(root.resolve())
            loc.found_by.add("project-registry")
            loc.repo_dir = (root / REPO_DIR_NAME).resolve()
            loc.registry_name = e.get("name") or ""
        else:
            missing_registered.append(str(root))

    hidden = _hidden_roots()

    def is_hidden(root: Path) -> bool:
        if str(root) in hidden:
            return True
        # A project's worktree copies (<root>/.researchforge/worktrees/…) go with it.
        return REPO_DIR_NAME in root.parts and any(str(p) in hidden for p in root.parents)

    hidden_locs = [loc for loc in by_root.values() if is_hidden(loc.root)]
    locations = sorted(
        (loc for loc in by_root.values() if not is_hidden(loc.root)), key=lambda loc: str(loc.root)
    )
    for loc in locations:
        loc.overnight_dirs.sort()
    report = {
        "cwd": str(cwd),
        "scan_roots": [str(r) for r in roots],
        "scan_depth": depth,
        "registry_path": str(registry_path()),
        "hidden_roots": sorted(str(loc.root) for loc in hidden_locs),
        "locations_found": len(locations),
        "missing_registered_paths": missing_registered,
        "errors": errors[:20],
    }
    return locations, report
