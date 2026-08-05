"""Global project registry — every project on this machine, discoverable.

Projects are directory-scoped; the registry (in `~/.researchforge/`, or
`$RESEARCHFORGE_HOME`) is the machine-wide index the hub dashboard and
"did you mean" errors read. It is best-effort bookkeeping: corrupt or
stale entries are tolerated, and a project is only ever *really* defined
by its own `.researchforge/` directory.
"""

from __future__ import annotations

import json
import os
import re
from datetime import UTC, datetime
from pathlib import Path

from pydantic import BaseModel

REGISTRY_FILENAME = "projects.json"


class RegistryEntry(BaseModel):
    slug: str
    name: str
    path: str
    created_at: str
    last_active: str

    @property
    def exists(self) -> bool:
        return (Path(self.path) / ".researchforge").is_dir()


def researchforge_home() -> Path:
    env = os.environ.get("RESEARCHFORGE_HOME")
    return Path(env) if env else Path.home() / ".researchforge"


def registry_path() -> Path:
    return researchforge_home() / REGISTRY_FILENAME


def load_registry() -> list[RegistryEntry]:
    path = registry_path()
    if not path.is_file():
        return []
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        return [RegistryEntry.model_validate(item) for item in raw]
    except (ValueError, TypeError):
        return []


def _save(entries: list[RegistryEntry]) -> None:
    path = registry_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = [entry.model_dump() for entry in entries]
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _slugify(name: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")
    return slug or "project"


def _unique_slug(name: str, taken: set[str]) -> str:
    base = _slugify(name)
    if base not in taken:
        return base
    counter = 2
    while f"{base}-{counter}" in taken:
        counter += 1
    return f"{base}-{counter}"


def register_project(root: Path) -> RegistryEntry:
    """Upsert the project at `root` (resolved); returns its entry."""
    resolved = root.resolve()
    now = datetime.now(UTC).isoformat(timespec="seconds")
    entries = load_registry()
    for index, entry in enumerate(entries):
        if entry.path == str(resolved):
            updated = entry.model_copy(update={"last_active": now})
            entries[index] = updated
            _save(entries)
            return updated
    entry = RegistryEntry(
        slug=_unique_slug(resolved.name, {e.slug for e in entries}),
        name=resolved.name,
        path=str(resolved),
        created_at=now,
        last_active=now,
    )
    entries.append(entry)
    _save(entries)
    return entry


def touch_project(root: Path) -> RegistryEntry | None:
    """Record activity for an initialized project (registers it if unknown)."""
    if not (root / ".researchforge").is_dir():
        return None
    return register_project(root)


def find_by_slug(slug: str) -> RegistryEntry | None:
    return next((entry for entry in load_registry() if entry.slug == slug), None)


def prune_missing() -> list[RegistryEntry]:
    """Remove entries whose project folder no longer exists. Returns the removed entries."""
    entries = load_registry()
    missing = [e for e in entries if not e.exists]
    if missing:
        _save([e for e in entries if e.exists])
    return missing
