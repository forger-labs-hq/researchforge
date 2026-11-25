"""The plugin contract: what a ``researchforge.plugins`` entry point may provide.

A plugin is any object with a ``name``; every other member is optional, so a
plugin implements only what it adds. Subclassing :class:`Plugin` gives no-op
defaults for the rest. The contract is versioned by :data:`API_VERSION`: a
plugin that declares a newer ``api_version`` than this install supports is not
loaded, and says so in ``researchforge plugins``.

What a plugin can add:

* ``register_cli(app)`` — commands or command groups on the ``researchforge`` CLI;
* ``dashboard_views()`` — entries in the live dashboard's nav, each showing one of
  the plugin's own pages;
* ``api_routes()`` — HTTP routes on the live dashboard under ``/plugins/<name>/``
  (same localhost-only Host check; POSTs need the dashboard's per-process token);
* ``on_monitor_cycle(event)`` — called after every Kaggle ``monitor`` cycle and
  every repo auto-run monitor cycle; what it returns is recorded as a sensor.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import typer

API_VERSION = 1


@dataclass(frozen=True)
class DashboardView:
    """A nav entry in the live dashboard showing the plugin page at ``path``.

    ``path`` is relative to the plugin's mount, e.g. ``"/status"`` is served by
    the plugin's GET route ``/status`` at ``/plugins/<name>/status``.
    """

    id: str
    label: str
    path: str


@dataclass(frozen=True)
class PluginRequest:
    method: str
    # The path below the plugin's mount: "/status" for /plugins/<name>/status.
    path: str
    query: Mapping[str, list[str]] = field(default_factory=dict)
    body: bytes = b""
    headers: Mapping[str, str] = field(default_factory=dict)
    # The dashboard's per-process token: a plugin page embeds it to POST back.
    token: str = ""

    def json(self) -> Any:
        return json.loads(self.body or b"{}")


@dataclass(frozen=True)
class PluginResponse:
    status: int = 200
    body: bytes | str = b""
    content_type: str = "text/html; charset=utf-8"

    @classmethod
    def json(cls, data: Any, status: int = 200) -> PluginResponse:
        return cls(status, json.dumps(data, default=str), "application/json")

    def payload(self) -> bytes:
        return self.body.encode("utf-8") if isinstance(self.body, str) else self.body


Handler = Callable[[PluginRequest], PluginResponse]


@dataclass(frozen=True)
class Route:
    method: Literal["GET", "POST"]
    path: str
    handler: Handler


@dataclass(frozen=True)
class MonitorEvent:
    """One finished monitor cycle.

    ``kind`` is ``"kaggle"`` (``artifacts_dir`` is the competition's overnight
    directory, ``slug`` its slug) or ``"repo"`` (``artifacts_dir`` is the repo's
    ``.researchforge``). ``root`` is the project folder in both.
    """

    kind: Literal["kaggle", "repo"]
    root: Path
    artifacts_dir: Path
    slug: str = ""


# A hook returns (status, detail) like a sensor — status ok | skipped | error —
# or None when it has nothing to report.
MonitorResult = tuple[str, str] | None


class Plugin:
    """Optional base class: no-op defaults for everything but ``name``."""

    name: str = ""
    version: str = ""
    api_version: int = API_VERSION

    def register_cli(self, app: typer.Typer) -> None:
        return None

    def dashboard_views(self) -> list[DashboardView]:
        return []

    def api_routes(self) -> list[Route]:
        return []

    def on_monitor_cycle(self, event: MonitorEvent) -> MonitorResult:
        return None
