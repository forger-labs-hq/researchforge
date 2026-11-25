# Writing a ResearchForge plugin

ResearchForge loads plugins from installed Python packages. A plugin can add CLI commands,
pages on the live dashboard, HTTP routes on that dashboard, and a hook that runs after every
monitor cycle. With no plugin installed, ResearchForge behaves exactly as it does without
this feature.

## Declare the entry point

In your package's `pyproject.toml`:

```toml
[project.entry-points."researchforge.plugins"]
myplugin = "my_package.plugin:plugin"
```

The entry point resolves to a plugin object, a `Plugin` subclass (it is instantiated) or a
zero-argument factory that returns one. After `pip install`, `researchforge plugins` shows
it and what it adds.

## The plugin object

Only `name` is required (lowercase letters, digits, `-` and `_`). Everything else is
optional; subclassing `researchforge.plugins.Plugin` gives no-op defaults.

```python
import typer
from researchforge.plugins import (
    DashboardView, MonitorEvent, Plugin, PluginRequest, PluginResponse, Route,
)


class MyPlugin(Plugin):
    name = "myplugin"
    version = "0.1.0"

    def register_cli(self, app: typer.Typer) -> None:
        sub = typer.Typer(help="My plugin's commands.")

        @sub.command("hello")
        def hello() -> None:
            typer.echo("hello")

        app.add_typer(sub, name="myplugin")          # researchforge myplugin hello

    def dashboard_views(self) -> list[DashboardView]:
        return [DashboardView("status", "My plugin", "/status")]

    def api_routes(self) -> list[Route]:
        def status(req: PluginRequest) -> PluginResponse:
            return PluginResponse(body="<h1>All good</h1>")

        return [Route("GET", "/status", status)]

    def on_monitor_cycle(self, event: MonitorEvent) -> tuple[str, str] | None:
        return "ok", f"checked {event.root}"


plugin = MyPlugin()
```

### CLI commands: `register_cli(app)`

Called once when the `researchforge` CLI starts, after the built-in commands. Add commands
or a command group. Keep imports at module level light, because every CLI run imports your
plugin module.

### Dashboard routes: `api_routes()`

Routes are mounted on the live dashboard (`researchforge kaggle dashboard`, localhost:7842) at
`/plugins/<name>/<path>`. A route matches the method and the exact path below the mount. The
handler gets a `PluginRequest` (method, path, query, body, headers, and the dashboard's
per-process `token`) and returns a `PluginResponse` (`PluginResponse.json(data)` for JSON).

The dashboard's protections apply to plugin routes as well:

- The dashboard answers only `localhost` Host headers.
- A POST must carry the per-process token in an `X-RF-Token` header. A plugin page gets the
  token from `req.token` and embeds it for its own POSTs.
- A handler that raises is returned as a 500 that names the plugin. It does not stop the
  dashboard.

### Dashboard views: `dashboard_views()`

Each `DashboardView(id, label, path)` adds a nav entry to the dashboard. Selecting it shows
the plugin's GET route at `path` in a frame. The frame stays loaded while the dashboard
refreshes its data. The dashboard is dark-only, so style the page for it: set
`color-scheme: dark` and use a dark background (the dashboard's is `#0f1117`, with text in
`#e2e8f0`). Otherwise a viewer with a light system theme sees low-contrast text.

### Monitor hook: `on_monitor_cycle(event)`

Called after each Kaggle `monitor` cycle (`event.kind == "kaggle"`, `artifacts_dir` is the
competition's `.overnight_<slug>` folder) and after each repository auto-run monitor cycle
(`event.kind == "repo"`, `artifacts_dir` is the repository's `.researchforge`). If the hook
returns `(status, detail)`, it shows as the sensor `plugin:<name>` on the dashboard, with
status `ok`, `skipped` or `error`. If it raises, the error is recorded as that sensor's
status, and the cycle carries on.

## A stable API to build on: `researchforge.plugins.facade`

Use the facade rather than ResearchForge internals:

| Function | What it does |
|---|---|
| `list_projects()` / `find_project(id)` | Projects the dashboard knows, as dashboard payloads |
| `project_summary(project)` | State, best result, rounds, open questions |
| `open_questions(project)` | Open 🔔 questions; `approval: true` marks ones a person must answer |
| `answer_question(project, id, answer, comment, where)` | Answers a question, the same as the bell |
| `plans_awaiting_approval(project)` / `decide_plan(...)` | Repo plans waiting on **Approve plan**; approving needs the typed word `approve` |
| `directions(project)` / `add_direction(...)` / `reorder_directions(...)` | The **Up next** steering queue |

## Failure handling and switching plugins off

The following plugins are skipped, and `researchforge plugins` lists each one with the reason:

- a plugin that fails to import;
- a plugin that declares a newer `api_version` than this ResearchForge supports;
- a plugin whose name clashes with another plugin's;
- a plugin that raises while registering.

A skipped plugin never breaks a command. To start with every plugin switched off, set
`RESEARCHFORGE_NO_PLUGINS=1`.
