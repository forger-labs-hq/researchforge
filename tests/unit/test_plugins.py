"""Plugins: discovery through entry points, and every place they plug in."""

from __future__ import annotations

import http.server
import json
import sys
import threading
import types
import urllib.error
import urllib.request
from collections.abc import Iterator
from importlib.metadata import EntryPoint
from pathlib import Path
from typing import Any

import pytest
import typer
from typer.testing import CliRunner

from researchforge import plugins
from researchforge.plugins import (
    DashboardView,
    MonitorEvent,
    Plugin,
    PluginRequest,
    PluginResponse,
    Route,
)

seen_events: list[MonitorEvent] = []


class Demo(Plugin):
    name = "demo"
    version = "1.2.3"

    def register_cli(self, app: typer.Typer) -> None:
        @app.command("demo-hello")
        def hello() -> None:
            typer.echo("hello from demo")

    def dashboard_views(self) -> list[DashboardView]:
        return [DashboardView("status", "Demo status", "/status")]

    def api_routes(self) -> list[Route]:
        def status(req: PluginRequest) -> PluginResponse:
            return PluginResponse(body=f"<p>demo ok {req.query.get('x', [''])[0]}</p>")

        def echo(req: PluginRequest) -> PluginResponse:
            return PluginResponse.json({"got": req.json()})

        def boom(req: PluginRequest) -> PluginResponse:
            raise RuntimeError("kaput")

        return [
            Route("GET", "/status", status),
            Route("POST", "/echo", echo),
            Route("GET", "/boom", boom),
        ]

    def on_monitor_cycle(self, event: MonitorEvent) -> tuple[str, str]:
        seen_events.append(event)
        return "ok", f"saw {event.kind}"


class Broken(Plugin):
    name = "broken"

    def register_cli(self, app: typer.Typer) -> None:
        raise RuntimeError("cannot register")

    def on_monitor_cycle(self, event: MonitorEvent) -> tuple[str, str]:
        raise RuntimeError("hook failed")


class TooNew(Plugin):
    name = "toonew"
    api_version = plugins.API_VERSION + 1


@pytest.fixture
def installed(monkeypatch: pytest.MonkeyPatch) -> Iterator[list[EntryPoint]]:
    """Fake installed plugins: entry points resolving into a throwaway module."""
    mod = types.ModuleType("rf_fake_plugins")
    mod.demo = Demo()  # type: ignore[attr-defined]
    mod.broken = Broken  # a class: the loader instantiates it  # type: ignore[attr-defined]
    mod.toonew = TooNew()  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "rf_fake_plugins", mod)
    eps: list[EntryPoint] = [EntryPoint("demo", "rf_fake_plugins:demo", plugins.GROUP)]
    monkeypatch.setattr(plugins, "entry_points", lambda group: list(eps) if group else [])
    monkeypatch.delenv("RESEARCHFORGE_NO_PLUGINS", raising=False)
    seen_events.clear()
    yield eps
    plugins._errors.clear()


def _load(eps: list[EntryPoint], *more: str) -> list[plugins.LoadedPlugin]:
    for spec in more:
        name, _, value = spec.partition("=")
        eps.append(EntryPoint(name, value, plugins.GROUP))
    return plugins.load_plugins(refresh=True)


def test_no_plugins_changes_nothing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(plugins, "entry_points", lambda group: [])
    assert plugins.load_plugins(refresh=True) == []
    app = typer.Typer()
    plugins.register_cli(app)
    assert app.registered_commands == [] and app.registered_groups == []
    assert plugins.dashboard_views() == []
    assert plugins.find_route("demo", "GET", "/status") is None
    assert plugins.run_monitor_hooks(MonitorEvent("repo", Path("."), Path("."))) == []
    assert plugins.errors() == []


def test_a_plugin_adds_a_command(installed: list[EntryPoint]) -> None:
    (lp,) = _load(installed)
    assert (lp.name, lp.version) == ("demo", "1.2.3")
    app = typer.Typer()

    @app.command("builtin")
    def builtin() -> None:
        pass

    plugins.register_cli(app)
    result = CliRunner().invoke(app, ["demo-hello"])
    assert result.exit_code == 0 and "hello from demo" in result.output


def test_broken_and_too_new_plugins_are_skipped_not_fatal(installed: list[EntryPoint]) -> None:
    loaded = _load(
        installed,
        "broken=rf_fake_plugins:broken",
        "toonew=rf_fake_plugins:toonew",
        "missing=rf_no_such_module:plugin",
    )
    assert [lp.name for lp in loaded] == ["broken", "demo"]
    app = typer.Typer()
    app.command("builtin")(lambda: None)
    plugins.register_cli(app)  # broken raises inside; demo still registers
    assert CliRunner().invoke(app, ["demo-hello"]).exit_code == 0
    results = plugins.run_monitor_hooks(MonitorEvent("kaggle", Path("r"), Path("r/a"), "s"))
    assert ("plugin:broken", "error", "RuntimeError: hook failed") in results
    assert ("plugin:demo", "ok", "saw kaggle") in results
    stages = {(e.entry_point.split(" ")[0], e.stage) for e in plugins.errors()}
    assert {
        ("broken", "register_cli"),
        ("broken", "monitor"),
        ("toonew", "load"),
        ("missing", "load"),
    } <= stages
    too_new = next(e for e in plugins.errors() if e.entry_point.startswith("toonew"))
    assert "upgrade researchforge" in too_new.error


def test_switching_plugins_off(
    installed: list[EntryPoint], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("RESEARCHFORGE_NO_PLUGINS", "1")
    assert plugins.load_plugins(refresh=True) == []


def test_two_plugins_with_one_name(installed: list[EntryPoint]) -> None:
    loaded = _load(installed, "zdemo=rf_fake_plugins:demo")
    assert [lp.entry_point.split(" ")[0] for lp in loaded] == ["demo"]
    assert "already named" in plugins.errors()[0].error


def test_plugins_command_lists_what_each_adds(installed: list[EntryPoint]) -> None:
    from researchforge.plugins.cli import plugins_command

    _load(installed, "toonew=rf_fake_plugins:toonew")
    app = typer.Typer()
    app.command("plugins")(plugins_command)
    app.command("other")(lambda: None)
    out = CliRunner().invoke(app, ["plugins", "--json"])
    data = json.loads(out.output)
    assert data["plugins"][0]["name"] == "demo"
    assert data["plugins"][0]["adds"] == [
        "commands",
        "dashboard views",
        "API routes",
        "monitor hook",
    ]
    assert data["errors"][0]["stage"] == "load"


def test_monitor_hooks_run_after_a_repo_cycle(
    installed: list[EntryPoint], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from researchforge.autorun import unattended as ua

    _load(installed)
    monkeypatch.setattr(ua, "_watch_launch", lambda root, cfg: ("ok", ""))
    monkeypatch.setattr(ua, "start_if_ready", lambda root: ("skipped", "off"))
    monkeypatch.setattr(ua, "_papers", lambda root, cfg, state: ("skipped", ""))
    monkeypatch.setattr(ua, "_briefing", lambda root, cfg, state: ("skipped", ""))
    state = ua.monitor_cycle(tmp_path)
    assert state["sensors"]["plugin:demo"]["status"] == "ok"
    (event,) = seen_events
    assert (event.kind, event.root, event.artifacts_dir) == ("repo", tmp_path, ua.rf_dir(tmp_path))


class TestDashboard:
    @pytest.fixture
    def server(self, tmp_path: Path, installed: list[EntryPoint]) -> Iterator[tuple[Any, int]]:
        from researchforge.kaggle.dashboard import DashboardService, make_handler

        _load(installed)
        ws = tmp_path / "ws"
        ws.mkdir()
        service = DashboardService(ws)
        httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), make_handler(service, 0))
        port = httpd.server_address[1]
        httpd.RequestHandlerClass = make_handler(service, port)
        threading.Thread(target=httpd.serve_forever, daemon=True).start()
        yield service, port
        httpd.shutdown()

    @staticmethod
    def _call(
        port: int, path: str, body: dict[str, Any] | None = None, token: str = ""
    ) -> tuple[int, bytes]:
        req = urllib.request.Request(
            f"http://127.0.0.1:{port}{path}",
            method="POST" if body is not None else "GET",
            data=json.dumps(body).encode() if body is not None else None,
            headers={"X-RF-Token": token},
        )
        try:
            with urllib.request.urlopen(req, timeout=10) as r:
                return r.status, r.read()
        except urllib.error.HTTPError as e:
            return e.code, e.read()

    def test_views_and_routes_reach_the_plugin(self, server: tuple[Any, int]) -> None:
        service, port = server
        status, raw = self._call(port, "/api/state")
        state = json.loads(raw)
        assert state["plugins"]["views"] == [
            {
                "key": "plugin/demo/status",
                "plugin": "demo",
                "id": "status",
                "label": "Demo status",
                "url": "/plugins/demo/status",
            }
        ]
        assert self._call(port, "/plugins/demo/status?x=7") == (200, b"<p>demo ok 7</p>")
        assert self._call(port, "/plugins/demo/nope")[0] == 404
        assert self._call(port, "/plugins/other/status")[0] == 404
        code, raw = self._call(port, "/plugins/demo/boom")
        assert code == 500 and b"kaput" in raw

    def test_plugin_posts_need_the_dashboard_token(self, server: tuple[Any, int]) -> None:
        service, port = server
        assert self._call(port, "/plugins/demo/echo", {"a": 1}, "wrong")[0] == 403
        code, raw = self._call(port, "/plugins/demo/echo", {"a": 1}, service.token)
        assert code == 200 and json.loads(raw) == {"got": {"a": 1}}


def test_dashboard_without_plugins_reports_none(tmp_path: Path) -> None:
    from researchforge.kaggle.dashboard import DashboardService

    payload = DashboardService(tmp_path).state()
    assert payload["plugins"] == {"installed": [], "views": [], "errors": []}
