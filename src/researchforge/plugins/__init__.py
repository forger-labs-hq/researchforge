"""Plugin discovery: installed packages extend ResearchForge through entry points.

A package declares, in its ``pyproject.toml``::

    [project.entry-points."researchforge.plugins"]
    myplugin = "my_package.plugin:plugin"

The entry point resolves to a plugin object (or a zero-argument factory that
returns one) following :mod:`researchforge.plugins.api`. Discovery runs once
per process. A plugin that fails to import, declares a newer API, or raises
while registering is skipped and recorded in :func:`errors` — a plugin can
never break a command. With no plugin installed, nothing changes.
``RESEARCHFORGE_NO_PLUGINS=1`` turns discovery off.
"""

from __future__ import annotations

import os
import re
import sys
from dataclasses import dataclass
from importlib.metadata import EntryPoint, entry_points
from typing import Any

import typer

from researchforge.plugins.api import (
    API_VERSION,
    DashboardView,
    MonitorEvent,
    Plugin,
    PluginRequest,
    PluginResponse,
    Route,
)

__all__ = [
    "API_VERSION",
    "DashboardView",
    "LoadedPlugin",
    "MonitorEvent",
    "Plugin",
    "PluginError",
    "PluginRequest",
    "PluginResponse",
    "Route",
    "dashboard_views",
    "errors",
    "find_route",
    "load_plugins",
    "register_cli",
    "run_monitor_hooks",
]

GROUP = "researchforge.plugins"
_NAME = re.compile(r"^[a-z][a-z0-9_-]{0,39}$")


@dataclass(frozen=True)
class LoadedPlugin:
    name: str
    entry_point: str
    distribution: str
    version: str
    plugin: Any


@dataclass(frozen=True)
class PluginError:
    entry_point: str
    stage: str  # load | register_cli | dashboard_views | api_routes | monitor
    error: str


_loaded: list[LoadedPlugin] | None = None
_errors: list[PluginError] = []


def _describe(exc: BaseException) -> str:
    return f"{type(exc).__name__}: {exc}"


def _resolve(ep: EntryPoint) -> Any:
    obj = ep.load()
    if isinstance(obj, type) or (callable(obj) and not hasattr(obj, "name")):
        obj = obj()
    return obj


def load_plugins(refresh: bool = False) -> list[LoadedPlugin]:
    """Every installed plugin that loaded cleanly, in entry-point name order."""
    global _loaded
    if _loaded is not None and not refresh:
        return _loaded
    _loaded, _errors[:] = [], []
    if os.environ.get("RESEARCHFORGE_NO_PLUGINS"):
        return _loaded
    seen: set[str] = set()
    for ep in sorted(entry_points(group=GROUP), key=lambda e: e.name):
        label = f"{ep.name} = {ep.value}"
        try:
            plugin = _resolve(ep)
            name = str(getattr(plugin, "name", "") or "")
            if not _NAME.match(name):
                raise ValueError(f"plugin name {name!r} must match {_NAME.pattern}")
            if name in seen:
                raise ValueError(f"another plugin is already named {name!r}")
            wanted = int(getattr(plugin, "api_version", API_VERSION))
            if wanted > API_VERSION:
                raise ValueError(
                    f"needs plugin API {wanted}; this ResearchForge has {API_VERSION} — "
                    "upgrade researchforge"
                )
        except Exception as exc:
            _errors.append(PluginError(label, "load", _describe(exc)))
            continue
        seen.add(name)
        dist = ep.dist
        _loaded.append(
            LoadedPlugin(
                name=name,
                entry_point=label,
                distribution=dist.name if dist is not None else "",
                version=str(getattr(plugin, "version", "") or (dist.version if dist else "")),
                plugin=plugin,
            )
        )
    return _loaded


def errors() -> list[PluginError]:
    """Problems found while loading or calling plugins in this process."""
    load_plugins()
    return list(_errors)


def _record(lp: LoadedPlugin, stage: str, exc: BaseException) -> None:
    _errors.append(PluginError(lp.entry_point, stage, _describe(exc)))


def register_cli(app: typer.Typer) -> None:
    """Let each plugin add its commands; a failing plugin is skipped."""
    for lp in load_plugins():
        register = getattr(lp.plugin, "register_cli", None)
        if register is None:
            continue
        try:
            register(app)
        except Exception as exc:
            _record(lp, "register_cli", exc)


def dashboard_views() -> list[dict[str, str]]:
    """Plugin nav entries for the live dashboard, with the URL each one shows."""
    out: list[dict[str, str]] = []
    for lp in load_plugins():
        views = getattr(lp.plugin, "dashboard_views", None)
        if views is None:
            continue
        try:
            for v in views():
                if not isinstance(v, DashboardView):
                    raise TypeError(f"dashboard_views() returned {type(v).__name__}")
                path = v.path if v.path.startswith("/") else "/" + v.path
                out.append(
                    {
                        "key": f"plugin/{lp.name}/{v.id}",
                        "plugin": lp.name,
                        "id": v.id,
                        "label": v.label,
                        "url": f"/plugins/{lp.name}{path}",
                    }
                )
        except Exception as exc:
            _record(lp, "dashboard_views", exc)
    return out


def find_route(plugin_name: str, method: str, path: str) -> Route | None:
    """The plugin route serving ``method path`` (path below the plugin's mount)."""
    for lp in load_plugins():
        if lp.name != plugin_name:
            continue
        routes = getattr(lp.plugin, "api_routes", None)
        if routes is None:
            return None
        try:
            for route in routes():
                if route.method == method and route.path == path:
                    return route  # type: ignore[no-any-return]
        except Exception as exc:
            _record(lp, "api_routes", exc)
        return None
    return None


def run_monitor_hooks(event: MonitorEvent) -> list[tuple[str, str, str]]:
    """Call every plugin's monitor hook; returns (sensor name, status, detail)."""
    out: list[tuple[str, str, str]] = []
    for lp in load_plugins():
        hook = getattr(lp.plugin, "on_monitor_cycle", None)
        if hook is None:
            continue
        name = f"plugin:{lp.name}"
        try:
            result = hook(event)
        except Exception as exc:
            _record(lp, "monitor", exc)
            print(f"[plugins] {name} monitor hook failed: {_describe(exc)}", file=sys.stderr)
            out.append((name, "error", _describe(exc)))
            continue
        if result is not None:
            status, detail = result
            out.append((name, str(status), str(detail)))
    return out
