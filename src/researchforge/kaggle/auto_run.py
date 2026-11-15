"""The project's driver, and auto-run: rounds on untested ideas with no chat open.

The *driver* is one per-project choice — ``claude``, ``cursor`` or ``byok`` —
stored in ``auto_run.json`` and switchable at any time (``kaggle driver``, the
dashboard, ``/overnight driver …``). Everything unattended reads it: auto-run
launches, and, with ``route_sensors`` on, the AI calls the sensors make (notebook
ideas, arXiv scans, re-synthesis) through :func:`project_provider`. A switch takes
effect at the next launch or sensor call; a launch already running finishes on
the driver it started with unless it is stopped.

The watchers only fill ``pending_hypotheses.json``. Without this, a round runs only
when someone opens a chat (Claude Code or Cursor) or runs ``kaggle run`` by hand.
The ``auto_run_sensor`` closes that gap: each monitor cycle it checks the gates
below and, if they all pass, starts one *launch* — up to ``rounds_per_launch``
rounds driven by one of three drivers, unattended:

* ``claude`` — headless Claude Code (``claude -p``) following the overnight skill,
  allowed only ``researchforge kaggle …`` commands and file edits in the project;
* ``cursor`` — the Cursor agent CLI (``cursor-agent -p``) with the same prompt;
* ``byok``   — ``researchforge kaggle run`` with an AI API key (no chat agent).

Gates, all required: setup done, loop not paused, no open question, no round
already running, untested ideas in the pool, today's round count under
``max_rounds_per_day``, the Kaggle GPU cap not reached, and the driver ready
(binary found and signed in, or its key present).

The sensor never waits on a launch. It spawns a detached child
(``python -m researchforge.kaggle.auto_run_child <artifacts_dir> <slug>``) that runs the
driver with a time limit, records the outcome in ``auto_run_state.json`` and
sends a notification. Nothing here submits to Kaggle.

Unattended processes (launchd) see no shell profile, so keys and tokens come
from ``~/.config/researchforge/auto_run.env`` (``KEY=value`` lines, readable only
by you, mode 600): ``CLAUDE_CODE_OAUTH_TOKEN`` from ``claude setup-token``,
``CURSOR_API_KEY``, or ``ANTHROPIC_API_KEY`` / ``GEMINI_API_KEY`` /
``OPENAI_API_KEY`` for ``byok``. They are passed to the driver only.
"""

from __future__ import annotations

import contextlib
import json
import os
import re
import shutil
import signal
import sqlite3
import stat
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from researchforge.kaggle.workspace import atomic_write_json, read_json

SENSOR = "auto_run_sensor"
CONFIG_FILE = "auto_run.json"
STATE_FILE = "auto_run_state.json"
LOG_FILE = "auto_run.log"
DRIVERS = ("claude", "cursor", "byok")
BYOK_KEYS = ("ANTHROPIC_API_KEY", "GEMINI_API_KEY", "GOOGLE_API_KEY", "OPENAI_API_KEY")


# What detect_host reads: a process started on someone else's behalf (the dashboard's
# setup wizard) drops these, so setup does not claim it ran inside an agent.
HOST_MARKERS = ("CLAUDECODE", "AI_AGENT", "CURSOR_AGENT", "CURSOR_TRACE_ID")


def detect_host(env: dict[str, str] | None = None) -> tuple[str, str]:
    """The coding agent this process runs inside: ("claude" | "cursor" | "", why).

    Claude Code marks its shells with ``CLAUDECODE=1`` and ``AI_AGENT=claude-code_…``;
    Cursor's agent shells set ``CURSOR_AGENT`` (or an ``AI_AGENT`` naming Cursor) and
    its terminal ``CURSOR_TRACE_ID``.
    """
    env = dict(os.environ) if env is None else env
    agent = env.get("AI_AGENT", "").lower()
    if env.get("CLAUDECODE") == "1" or agent.startswith("claude"):
        return "claude", "you are running this from Claude Code"
    if env.get("CURSOR_AGENT") or agent.startswith("cursor") or env.get("CURSOR_TRACE_ID"):
        return "cursor", "you are running this from Cursor"
    return "", ""


def suggested_driver(artifacts_dir: Path, env: dict[str, str] | None = None) -> tuple[str, str]:
    """The driver that matches how this project already runs, and why.

    Setup's own choice wins: a project set up headless on an API key keeps using
    it. Otherwise the agent setup ran inside, then the agent running this now.
    """
    setup = read_json(artifacts_dir / "overnight_setup.json", {}) or {}
    mode = str(setup.get("ai_provider_mode") or "claude_code")
    if mode != "claude_code":
        return "byok", f"setup runs this project headless on your {mode} key"
    host = str(setup.get("agent_host") or "")
    if host in ("claude", "cursor"):
        name = "Claude Code" if host == "claude" else "Cursor"
        if setup.get("agent_host_source") == "picked":
            return host, f"you picked {name} for this project"
        return host, f"setup ran inside {name}"
    host, why = detect_host(env)
    if host:
        return host, why
    return "claude", "no agent or API key detected; Claude Code is the default"


def env_file() -> Path:
    return Path.home() / ".config" / "researchforge" / "auto_run.env"


@dataclass
class AutoRunConfig:
    driver: str = "claude"
    rounds_per_launch: int = 2
    max_rounds_per_day: int = 6
    max_minutes: int = 240  # a launch still running after this is stopped
    model: str = ""  # optional model id for the claude / cursor driver
    cli_path: str = ""  # optional explicit path to the claude / cursor-agent binary
    route_sensors: bool = True  # sensors' AI calls go through this driver too

    def save(self, artifacts_dir: Path) -> None:
        atomic_write_json(artifacts_dir / CONFIG_FILE, asdict(self))

    @classmethod
    def load(cls, artifacts_dir: Path) -> AutoRunConfig:
        raw = read_json(artifacts_dir / CONFIG_FILE, None)
        if not isinstance(raw, dict):  # never configured: match how the project runs
            raw = {"driver": suggested_driver(artifacts_dir)[0]}
        known = set(cls.__dataclass_fields__)
        cfg = cls(**{k: v for k, v in raw.items() if k in known})
        if cfg.driver not in DRIVERS:
            cfg.driver = "claude"
        cfg.rounds_per_launch = max(1, min(int(cfg.rounds_per_launch), 20))
        cfg.max_rounds_per_day = max(1, min(int(cfg.max_rounds_per_day), 100))
        cfg.max_minutes = max(10, min(int(cfg.max_minutes), 24 * 60))
        return cfg


def set_driver(
    artifacts_dir: Path,
    driver: str,
    *,
    model: str | None = None,
    route_sensors: bool | None = None,
) -> AutoRunConfig:
    """Switch the project's driver. Takes effect at the next launch or sensor call.

    The setup file is updated to match, so status output and later suggestions
    describe how the project now runs.
    """
    if driver not in DRIVERS:
        raise ValueError(f"driver must be one of {', '.join(DRIVERS)}")
    from researchforge.kaggle.run_config import record_change

    cfg = AutoRunConfig.load(artifacts_dir)
    record_change(artifacts_dir, "driver", cfg.driver, driver)
    if route_sensors is not None:
        record_change(artifacts_dir, "sensors_use_driver", cfg.route_sensors, route_sensors)
    if driver != cfg.driver:
        cfg.model = ""  # a model id belongs to one driver
    cfg.driver = driver
    if model is not None:
        cfg.model = model
    if route_sensors is not None:
        cfg.route_sensors = route_sensors
    cfg.save(artifacts_dir)
    setup_path = artifacts_dir / "overnight_setup.json"
    setup = read_json(setup_path, None)
    if isinstance(setup, dict):
        if driver == "byok":
            provider = api_provider()
            if provider is not None:
                setup["ai_provider_mode"] = provider.name.split("/")[0]
            setup["agent_host"] = ""
        else:
            setup["ai_provider_mode"] = "claude_code"  # an agent drives the loop
            setup["agent_host"] = driver
            setup["agent_host_source"] = "picked"  # chosen, not where setup happened to run
        atomic_write_json(setup_path, setup)
    return AutoRunConfig.load(artifacts_dir)


def setup_steps(driver: str) -> list[str]:
    """What a person does once so this driver can run unattended: copy-paste commands."""
    if driver == "claude":
        binary = find_binary("claude")
        return [
            "Get a token with `claude setup-token` (a browser opens once) and paste it in the "
            "dashboard's Keys & accounts — or in your own terminal run:  "
            "researchforge kaggle driver-login claude",
            "Claude Code found: "
            + (binary or "not installed — npm i -g @anthropic-ai/claude-code"),
            "Limits: only `researchforge kaggle …` commands and file edits inside the project",
        ]
    if driver == "cursor":
        binary = find_binary("cursor")
        return [
            "Install the Cursor agent CLI if needed:  curl https://cursor.com/install -fsS | bash"
            if not binary
            else f"Cursor agent CLI found: {binary}",
            "Paste a Cursor API key in the dashboard's Keys & accounts — or in your own "
            "terminal run:  researchforge kaggle driver-login cursor",
            "Limits: instructions only — headless Cursor needs --force, which allows any command",
        ]
    return [
        "Paste an Anthropic, Gemini or OpenAI key in the dashboard's Keys & accounts — or in "
        "your own terminal run:  researchforge kaggle driver-login byok",
        "Or use a local model: RESEARCHFORGE_LLM=ollama/<model>",
        "Billed per token by the provider; no chat agent involved",
    ]


@dataclass
class Readiness:
    ready: bool
    detail: str
    binary: str = ""
    fix: str = ""
    keys: list[str] = field(default_factory=list)  # names only, never values


# ---------------------------------------------------------------------------
# Keys and binaries
# ---------------------------------------------------------------------------


def load_env_file(path: Path | None = None) -> tuple[dict[str, str], str]:
    """``KEY=value`` pairs from the auto-run env file, and a problem if it is unsafe."""
    path = path or env_file()
    if not path.is_file():
        return {}, ""
    mode = stat.S_IMODE(path.stat().st_mode)
    if mode & 0o077:
        return {}, f"{path} is readable by others (mode {oct(mode)}); run: chmod 600 {path}"
    values: dict[str, str] = {}
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.removeprefix("export ").partition("=")
        values[key.strip()] = value.strip().strip("'\"")
    return values, ""


_LOGIN_FIX = (
    "dashboard → Keys & accounts, or in your own terminal: researchforge kaggle driver-login {}"
)
CLOUD_KEYS = ("RUNPOD_API_KEY", "LAMBDA_API_KEY", "VAST_API_KEY")
STORABLE_KEYS = ("CLAUDE_CODE_OAUTH_TOKEN", "CURSOR_API_KEY", *BYOK_KEYS, *CLOUD_KEYS)


def store_key(name: str, value: str, path: Path | None = None) -> Path:
    """Write or replace one ``KEY=value`` line in the env file, kept at mode 600."""
    if name not in STORABLE_KEYS:
        raise ValueError(f"key must be one of {', '.join(STORABLE_KEYS)}")
    if not value or "\n" in value:
        raise ValueError("the value must be one non-empty line")
    path = path or env_file()
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = path.read_text().splitlines() if path.is_file() else []
    kept = [ln for ln in lines if ln.strip().removeprefix("export ").partition("=")[0] != name]
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w") as fh:
        fh.write("\n".join([*kept, f"{name}={value}"]) + "\n")
    path.chmod(0o600)
    return path


def _newest(paths: list[Path]) -> Path | None:
    files = [p for p in paths if p.is_file() and os.access(p, os.X_OK)]
    return max(files, key=lambda p: p.stat().st_mtime) if files else None


def find_binary(driver: str, override: str = "") -> str:
    """The claude / cursor-agent executable, or "" if none is installed."""
    if override:
        return override if Path(override).is_file() else ""
    home = Path.home()
    if driver == "claude":
        # Inside Claude Code, the binary running right now.
        running = os.environ.get("CLAUDE_CODE_EXECPATH", "")
        if running and Path(running).is_file():
            return running
        on_path = shutil.which("claude")
        if on_path:
            return on_path
        fixed = [
            home / ".claude" / "local" / "claude",
            home / ".local" / "bin" / "claude",
            Path("/opt/homebrew/bin/claude"),
            Path("/usr/local/bin/claude"),
        ]
        hit = next((p for p in fixed if p.is_file()), None)
        if hit:
            return str(hit)
        # The Claude desktop app bundles one; its folder changes with each update.
        bundled = (home / "Library" / "Application Support" / "Claude" / "claude-code").glob(
            "*/*/claude.app/Contents/MacOS/claude"
        )
        newest = _newest(list(bundled))
        return str(newest) if newest else ""
    if driver == "cursor":
        on_path = shutil.which("cursor-agent")
        if on_path:
            return on_path
        fixed = [
            home / ".local" / "bin" / "cursor-agent",
            home / ".cursor" / "bin" / "cursor-agent",
        ]
        hit = next((p for p in fixed if p.is_file()), None)
        return str(hit) if hit else ""
    return ""


def readiness(cfg: AutoRunConfig, *, check_login: bool = True) -> Readiness:
    """Whether the configured driver can run unattended right now."""
    env, env_problem = load_env_file()
    if env_problem:
        return Readiness(False, env_problem, fix=f"chmod 600 {env_file()}")
    keys = sorted(k for k in env if env[k])
    if cfg.driver == "byok":
        have = [k for k in BYOK_KEYS if os.environ.get(k) or env.get(k)]
        local = os.environ.get("RESEARCHFORGE_LLM") or env.get("RESEARCHFORGE_LLM")
        if have or local:
            return Readiness(
                True, f"API key found: {', '.join(have) or 'RESEARCHFORGE_LLM'}", keys=keys
            )
        return Readiness(
            False,
            "no AI API key reaches the background monitor",
            fix=_LOGIN_FIX.format("byok"),
            keys=keys,
        )
    binary = find_binary(cfg.driver, cfg.cli_path)
    if cfg.driver == "claude":
        if not binary:
            return Readiness(
                False,
                "Claude Code CLI not found",
                fix="install it (npm i -g @anthropic-ai/claude-code) or set --cli-path",
            )
        if env.get("CLAUDE_CODE_OAUTH_TOKEN") or env.get("ANTHROPIC_API_KEY"):
            return Readiness(True, "token in the auto-run env file", binary, keys=keys)
        if not check_login:
            return Readiness(True, "binary found (sign-in not checked)", binary, keys=keys)
        try:
            out = subprocess.run(
                [binary, "auth", "status"], capture_output=True, text=True, timeout=20, check=False
            ).stdout
            logged_in = bool(json.loads(out or "{}").get("loggedIn"))
        except (OSError, subprocess.TimeoutExpired, ValueError):
            logged_in = False
        if logged_in:
            return Readiness(True, "signed in", binary, keys=keys)
        return Readiness(
            False,
            "Claude Code CLI is not signed in for unattended use",
            binary,
            fix=_LOGIN_FIX.format("claude"),
            keys=keys,
        )
    # cursor
    if not binary:
        return Readiness(
            False,
            "Cursor agent CLI (cursor-agent) not found",
            fix="install it: curl https://cursor.com/install -fsS | bash",
        )
    if env.get("CURSOR_API_KEY"):
        return Readiness(True, "CURSOR_API_KEY in the auto-run env file", binary, keys=keys)
    if not check_login:
        return Readiness(True, "binary found (sign-in not checked)", binary, keys=keys)
    try:
        proc = subprocess.run(
            [binary, "status"], capture_output=True, text=True, timeout=20, check=False
        )
        text = (proc.stdout + proc.stderr).lower()
        logged_in = proc.returncode == 0 and "not logged in" not in text
    except (OSError, subprocess.TimeoutExpired):
        logged_in = False
    if logged_in:
        return Readiness(True, "signed in", binary, keys=keys)
    return Readiness(
        False,
        "cursor-agent is not signed in",
        binary,
        fix=_LOGIN_FIX.format("cursor"),
        keys=keys,
    )


# ---------------------------------------------------------------------------
# AI calls through the driver (sensors, re-synthesis, the API-key loop)
# ---------------------------------------------------------------------------


@contextlib.contextmanager
def _env_file_applied() -> Any:
    """Keys from the auto-run env file, visible to provider construction only."""
    values, problem = load_env_file()
    added = {k: v for k, v in values.items() if v and not os.environ.get(k)} if not problem else {}
    os.environ.update(added)
    try:
        yield
    finally:
        for k in added:
            os.environ.pop(k, None)


def api_provider() -> Any:
    """An API-key provider from the environment or the auto-run env file, or None."""
    from researchforge.ai.providers import get_provider

    with _env_file_applied():
        try:
            return get_provider()
        except RuntimeError:
            return None


class AgentCLIProvider:
    """One-shot text generation through a coding agent's CLI, with every tool off.

    Lets sensors use the project's Claude Code or Cursor driver instead of an API
    key. Claude Code reports token usage in its JSON output, so these calls are
    metered like API calls; Cursor's are recorded with zero tokens.
    """

    def __init__(self, driver: str, binary: str, model: str = "", timeout: int = 600) -> None:
        self.driver, self.binary, self.model, self.timeout = driver, binary, model, timeout

    @property
    def name(self) -> str:
        return f"{self.driver}-cli/{self.model or 'default'}"

    def generate(self, system: str, user: str, *, max_tokens: int = 8192) -> str:
        from researchforge.ai.providers import _metered
        from researchforge.ai.usage import Usage

        if self.driver == "claude":
            cmd = [
                self.binary,
                "-p",
                user,
                "--system-prompt",
                system,
                "--tools",
                "",
                "--strict-mcp-config",
                "--no-session-persistence",
                "--output-format",
                "json",
            ]
        else:  # cursor: no --force, so it cannot write files or run commands
            cmd = [self.binary, "-p", f"{system}\n\n{user}", "--output-format", "text"]
        if self.model:
            cmd += ["--model", self.model]
        env = dict(os.environ)
        env.update(load_env_file()[0])
        with _metered(self.name) as meter:
            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self.timeout,
                env=env,
                stdin=subprocess.DEVNULL,
                check=False,
            )
            if proc.returncode != 0:
                raise RuntimeError(
                    f"{self.driver} exited {proc.returncode}: {(proc.stderr or proc.stdout)[-300:]}"
                )
            if self.driver != "claude":
                return proc.stdout.strip()
            try:
                out = json.loads(proc.stdout)
            except ValueError:
                return proc.stdout.strip()
            usage = out.get("usage") or {}
            meter.usage = Usage(
                input_tokens=int(usage.get("input_tokens") or 0),
                output_tokens=int(usage.get("output_tokens") or 0),
            )
            if out.get("is_error"):
                raise RuntimeError(f"claude: {str(out.get('result'))[:300]}")
            return str(out.get("result") or "")


def project_provider(artifacts_dir: Path, purpose: str, *, sensor: bool = True) -> Any:
    """The AI a project's unattended work should use, metered — or None.

    ``sensor=True`` (watcher ideas, arXiv, re-synthesis): the project's Claude Code
    or Cursor driver when ``route_sensors`` is on and it is ready, else an API key.
    ``sensor=False`` (the API-key loop's proposals): an API key only.
    """
    from researchforge.kaggle.token_log import metered

    cfg = AutoRunConfig.load(artifacts_dir)
    # Inside that same agent (a chat, or an auto-run launch) the agent does the work
    # itself; calling its CLI again would start a second agent inside the first.
    inside = detect_host()[0] == cfg.driver
    if sensor and cfg.route_sensors and cfg.driver in ("claude", "cursor") and not inside:
        ready = readiness(cfg)
        if ready.ready and ready.binary:
            agent = AgentCLIProvider(cfg.driver, ready.binary, cfg.model)
            return metered(agent, artifacts_dir, purpose)
    return metered(api_provider(), artifacts_dir, purpose)


# ---------------------------------------------------------------------------
# Which driver ran each round
# ---------------------------------------------------------------------------

ROUND_DRIVERS_FILE = "round_drivers.json"


def record_round_driver(artifacts_dir: Path, round_num: int, driver: str = "") -> str:
    """Note which driver ran a round: auto-run's, the agent in use, or "byok"."""
    driver = driver or os.environ.get("RESEARCHFORGE_DRIVER", "") or detect_host()[0] or "manual"
    data = read_json(artifacts_dir / ROUND_DRIVERS_FILE, {}) or {}
    if not isinstance(data, dict):
        data = {}
    data[str(round_num)] = {
        "driver": driver,
        "auto_run": os.environ.get("RESEARCHFORGE_AUTO_RUN") == "1",
        "at": datetime.now().isoformat(timespec="seconds"),
    }
    with contextlib.suppress(OSError):
        atomic_write_json(artifacts_dir / ROUND_DRIVERS_FILE, data)
    return driver


def round_drivers(artifacts_dir: Path) -> dict[str, dict[str, Any]]:
    data = read_json(artifacts_dir / ROUND_DRIVERS_FILE, {}) or {}
    return {k: v for k, v in data.items() if isinstance(v, dict)} if isinstance(data, dict) else {}


# ---------------------------------------------------------------------------
# Gates
# ---------------------------------------------------------------------------


def rounds_today(artifacts_dir: Path) -> int:
    db = artifacts_dir / "experiment_log.db"
    if not db.is_file():
        return 0
    today = datetime.now().date().isoformat()
    try:
        with sqlite3.connect(f"file:{db}?mode=ro", uri=True) as conn:
            row = conn.execute(
                "SELECT COUNT(*) FROM experiments WHERE started_at LIKE ?", (today + "%",)
            ).fetchone()
    except sqlite3.Error:
        return 0
    return int(row[0]) if row else 0


def rounds_total(artifacts_dir: Path) -> int:
    from researchforge.kaggle.experiment_loop import load_rounds

    return len(load_rounds(artifacts_dir / "experiment_log.db", include_archived=True))


def _pid_alive(pid: Any) -> bool:
    try:
        os.kill(int(pid), 0)
    except (ProcessLookupError, ValueError, TypeError):
        return False
    except PermissionError:
        return True
    return True


def round_in_flight(artifacts_dir: Path) -> bool:
    from researchforge.kaggle.experiment_loop import recover_interrupted_round

    recover_interrupted_round(artifacts_dir)  # a dead process's lock must not block forever
    if (artifacts_dir / "execute_round.lock").exists():
        return True
    inflight = read_json(artifacts_dir / "inflight_rounds.json", {}) or {}
    return (
        any(isinstance(v, dict) and _pid_alive(v.get("pid")) for v in inflight.values())
        if isinstance(inflight, dict)
        else False
    )


def gates(artifacts_dir: Path, cfg: AutoRunConfig) -> list[str]:
    """Reasons not to start a launch now; empty means go."""
    from researchforge.kaggle import control
    from researchforge.kaggle.hypothesis_pool import rank_untested, setup_direction

    reasons: list[str] = []
    setup = read_json(artifacts_dir / "overnight_setup.json", None)
    if not isinstance(setup, dict):
        return ["setup not done — run researchforge kaggle overnight first"]
    if control.LoopControl.load(artifacts_dir).paused:
        reasons.append("the loop is paused")
    # Only a question that holds the loop stops launches; one that blocks a few
    # ideas just takes them out of the pool until it is answered.
    holding = control.holding_questions(artifacts_dir)
    if holding:
        reasons.append(f"{len(holding)} question(s) put the loop on hold until you answer")
    if round_in_flight(artifacts_dir):
        reasons.append("a round is already running")
    limited = usage_limited_until(artifacts_dir)
    if limited and limited[1] == cfg.driver:
        reasons.append(
            f"the {limited[1]} driver hit its usage limit; it resets {limited[0]:%a %H:%M} "
            "(or switch driver in Run configs)"
        )
    if not rank_untested(artifacts_dir, setup_direction(artifacts_dir)):
        reasons.append("no untested ideas in the pool")
    if str(setup.get("selected_gpu") or "") == "colab_gpu":
        from researchforge.kaggle import colab

        ok, detail, _fix = colab.ready()
        if not ok:
            reasons.append(f"Colab is not ready: {detail} — set it up in Run configs")
    done = rounds_today(artifacts_dir)
    if done >= cfg.max_rounds_per_day:
        reasons.append(f"{done} rounds today reaches the daily limit of {cfg.max_rounds_per_day}")
    target = str(setup.get("selected_gpu") or "cpu")
    if control.kaggle_gpu_cap_reached(artifacts_dir, target):
        reasons.append("today's Kaggle GPU cap is reached")
    return reasons


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------


def agent_prompt(slug: str, rounds: int) -> str:
    return (
        f"Continue the ResearchForge overnight loop for the Kaggle competition {slug} in this "
        f"folder, unattended, for at most {rounds} round(s), then stop.\n"
        "Use the overnight skill's loop (THE FULL LOOP, Phase 3 onwards): run "
        f"`researchforge kaggle prepare-round {slug}`, write the proposal file it asks for, run "
        f"`researchforge kaggle execute-round {slug}`, read the result, repeat.\n"
        "Write each round for the compute in prepare-round's `compute`: where the code runs, "
        "its time limit and its rules. On Colab the whole experiment runs on the Colab VM; "
        "never push or wait on a Kaggle notebook from a round.\n"
        "A round can take 30 minutes or more: run execute-round in the foreground and wait for "
        "it to return. Never run it in the background or start a second one while it runs. On "
        "Kaggle compute, a Kaggle kernel still running when execute-round stops waiting is not "
        "a failure: the monitor collects its score when it finishes, so do not re-run that "
        "round.\n"
        "A round that writes its submission anywhere other than ./submission.csv (or "
        "submission.zip) must print `SUBMISSION_FILE: <path>` on its own line, so the round "
        "keeps that file and the person can approve submitting it.\n"
        "prepare-round's `public_notebooks_ahead` lists public notebooks that claim a better "
        "leaderboard score than our best submission. If our local score keeps rising while "
        "the leaderboard does not, run the leading one as a round. A round with no local "
        "score but a submission is kept and offered for submission by itself; to ask about "
        f"one, use `researchforge kaggle offer-submission {slug} <round> --note "
        "'<its LB claim>'`, never `kaggle ask`: only that question submits on a yes.\n"
        "To read a public notebook (or its outputs) the pool points at, run "
        f"`researchforge kaggle pull-kernel {slug} <owner/notebook>`; do not ask the person "
        "to fetch it. To run a public notebook as a round, use "
        f'`researchforge kaggle propose-from-kernel {slug} <owner/notebook> --title "…"` '
        "(add `--append <file>` for scoring code you wrote); it writes the proposal, with the "
        "notebook's datasets attached.\n"
        "Never ask the person to run a command, script or install on their computer: rounds run "
        "on the competition's chosen compute, never here. If an idea needs a step you cannot "
        "do with `researchforge kaggle` commands, skip it with "
        f'`researchforge kaggle skip-idea {slug} "<exact pool idea>" --reason "<why>"` and '
        "move on to the next idea.\n"
        "Nobody is watching, so never wait for a reply:\n"
        "- If you need a decision from the person, ask it with "
        f'`researchforge kaggle ask {slug} "<question>" --options "<a>,<b>"` and add '
        '`--blocks "<exact pool idea>"` for each idea that must wait for the answer; then '
        "carry on with ideas it does not block. Use `--hold` only when no round should run "
        "until they answer, and then stop.\n"
        "- If an answer means changing a setting (compute, the experiment budget, the time "
        "budget, the round time limit, the Kaggle GPU cap, the Colab plan or GPU, the metric "
        'direction), attach it: `--action "<answer> => compute:kaggle, max_experiments:40"`. '
        "Picking that answer then changes Run configs. Never ask the person to change a "
        "setting by hand.\n"
        "- If the loop is paused or a question holds it, stop.\n"
        "- Never submit to Kaggle, never run setup again (no `kaggle overnight`, no --refresh), "
        "never change settings, compute or the metric direction, and never archive or delete "
        "rounds.\n"
        "- Only edit files inside this project folder.\n"
        "- Prefer the top-ranked untested idea from prepare-round unless results rule it out, "
        "but the person's `directions` come first: answer each one with reply_needed once, "
        f'with `researchforge kaggle steer {slug} --reply <id> "<what you will do>"`.\n'
        "End with one line: how many rounds ran and the best score."
    )


def build_command(artifacts_dir: Path, slug: str, cfg: AutoRunConfig) -> list[str]:
    """The argv for one launch (also shown by ``auto-run --dry-run``)."""
    from researchforge.kaggle.control import person_notes

    prompt = agent_prompt(slug, cfg.rounds_per_launch)
    notes = person_notes(artifacts_dir)
    if notes:
        prompt += (
            "\n\nThe person's directions, then their recent comments (newest first) — follow "
            "them; they outrank the idea ranking:\n" + "\n".join(f"- {n}" for n in notes)
        )
    if cfg.driver == "claude":
        cmd = [
            find_binary("claude", cfg.cli_path) or "claude",
            "-p",
            prompt,
            "--permission-mode",
            "dontAsk",
            "--allowedTools",
            "Bash(researchforge kaggle:*)",
            "Read",
            "Glob",
            "Grep",
            "Edit(./**)",  # Edit rules cover every file-editing tool, Write included
            "--disallowedTools",
            "Bash(researchforge experiment approve:*)",  # approvals are the person's
            "Bash(researchforge kaggle respond:*)",  # so are answers (a yes submits)
            "--output-format",
            "text",
        ]
        return cmd + (["--model", cfg.model] if cfg.model else [])
    if cfg.driver == "cursor":
        cmd = [
            find_binary("cursor", cfg.cli_path) or "cursor-agent",
            "-p",
            prompt,
            "--output-format",
            "text",
            "--force",  # headless file writes and commands; the prompt sets the limits
        ]
        return cmd + (["--model", cfg.model] if cfg.model else [])
    return [
        sys.executable,
        "-m",
        "researchforge",
        "kaggle",
        "run",
        slug,
        "--rounds",
        str(cfg.rounds_per_launch),
    ]


def _shim_dir() -> Path:
    """A ``researchforge`` on PATH that runs *this* code, for the agent's shell commands."""
    d = Path.home() / ".local" / "share" / "researchforge" / "auto_run_bin"
    d.mkdir(parents=True, exist_ok=True)
    src = Path(__file__).resolve().parents[2]  # .../src
    shim = d / "researchforge"
    body = (
        "#!/bin/sh\n"
        f'PYTHONPATH="{src}${{PYTHONPATH:+:$PYTHONPATH}}" '
        f'exec "{sys.executable}" -m researchforge "$@"\n'
    )
    if not shim.is_file() or shim.read_text() != body:
        shim.write_text(body)
        shim.chmod(0o755)
    return d


def child_env(artifacts_dir: Path, driver: str = "") -> dict[str, str]:
    env = dict(os.environ)
    values, _problem = load_env_file()
    env.update(values)
    env["PATH"] = os.pathsep.join([str(_shim_dir()), env.get("PATH", "/usr/bin:/bin")])
    env["RESEARCHFORGE_OVERNIGHT_DIR"] = str(artifacts_dir)
    env["RESEARCHFORGE_NO_DASHBOARD"] = "1"
    env["RESEARCHFORGE_AUTO_RUN"] = "1"
    if driver:
        env["RESEARCHFORGE_DRIVER"] = driver
    if driver == "claude":
        # Claude Code stops a shell command after 10 minutes by default; a full round
        # runs longer, and stopping it mid-round is what once froze a loop.
        limit_ms = str(AutoRunConfig.load(artifacts_dir).max_minutes * 60 * 1000)
        env["BASH_DEFAULT_TIMEOUT_MS"] = limit_ms
        env["BASH_MAX_TIMEOUT_MS"] = limit_ms
    return env


# ---------------------------------------------------------------------------
# State, launch and the sensor
# ---------------------------------------------------------------------------


def load_state(artifacts_dir: Path) -> dict[str, Any]:
    state = read_json(artifacts_dir / STATE_FILE, {}) or {}
    return state if isinstance(state, dict) else {}


def _save_state(artifacts_dir: Path, state: dict[str, Any]) -> None:
    atomic_write_json(artifacts_dir / STATE_FILE, state)


def current_launch(artifacts_dir: Path) -> dict[str, Any] | None:
    """The launch still running, if any."""
    last = load_state(artifacts_dir).get("current")
    if isinstance(last, dict) and _pid_alive(last.get("pid")):
        return last
    return None


def launch(artifacts_dir: Path, slug: str, cfg: AutoRunConfig) -> dict[str, Any]:
    """Spawn the detached child that runs one launch. Returns its record."""
    log = (artifacts_dir / LOG_FILE).open("a")
    proc = subprocess.Popen(
        [sys.executable, "-m", "researchforge.kaggle.auto_run_child", str(artifacts_dir), slug],
        cwd=str(artifacts_dir.parent),
        env={**os.environ, "PYTHONPATH": str(Path(__file__).resolve().parents[2])},
        stdout=log,
        stderr=subprocess.STDOUT,
        stdin=subprocess.DEVNULL,
        start_new_session=True,
    )
    record = {
        "pid": proc.pid,
        "driver": cfg.driver,
        "rounds_requested": cfg.rounds_per_launch,
        "started_at": datetime.now().isoformat(timespec="seconds"),
        "status": "running",
    }
    state = load_state(artifacts_dir)
    state["current"] = record
    _save_state(artifacts_dir, state)
    return record


def sensor_check(artifacts_dir: Path, slug: str) -> tuple[str, str]:
    """One monitor cycle of ``auto_run_sensor``: report, or start a launch."""
    cfg = AutoRunConfig.load(artifacts_dir)
    running = current_launch(artifacts_dir)
    if running:
        return "ok", f"{running['driver']} launch running since {running['started_at']}"
    blockers = gates(artifacts_dir, cfg)
    if blockers:
        return "skipped", "; ".join(blockers)
    ready = readiness(cfg)
    if not ready.ready:
        # Waiting on the person, not broken: say what to do instead of failing each cycle.
        return "skipped", f"waiting for the {cfg.driver} driver: {ready.detail} — fix: {ready.fix}"
    rec = launch(artifacts_dir, slug, cfg)
    return (
        "ok",
        f"started {cfg.driver} for up to {cfg.rounds_per_launch} round(s), pid {rec['pid']}",
    )


def _finish(artifacts_dir: Path, slug: str, outcome: dict[str, Any]) -> None:
    from researchforge.kaggle import control

    state = load_state(artifacts_dir)
    record = {**(state.get("current") or {}), **outcome}
    history = [h for h in state.get("history", []) if isinstance(h, dict)][-19:]
    state["history"] = [*history, record]
    state["current"] = None
    state["last"] = record
    _save_state(artifacts_dir, state)
    ran = record.get("rounds_run", 0)
    if record.get("status") in ("finished", "stopped"):
        how = "" if record.get("status") == "finished" else ", then stopped on request"
        msg = f"Auto-run ({record.get('driver')}): {ran} round(s) ran{how}."
        control.notify(artifacts_dir, "briefing", msg)
    elif record.get("status") == "usage_limit":
        msg = f"Auto-run paused: {record.get('error', '')}"
        control.notify(artifacts_dir, "alert", msg[:300])
    else:
        msg = f"Auto-run ({record.get('driver')}) {record.get('status')}: {record.get('error', '')}"
        control.notify(artifacts_dir, "alert", msg[:300], priority="high")


class _StopRequested(Exception):
    """SIGTERM reached the launch child: someone asked to stop this launch."""


def _on_sigterm(signum: int, frame: Any) -> None:
    raise _StopRequested


def stop_launch(artifacts_dir: Path) -> dict[str, Any] | None:
    """Ask the running launch to stop. It stops its driver and records "stopped"."""
    running = current_launch(artifacts_dir)
    if running is None:
        return None
    with contextlib.suppress(ProcessLookupError, PermissionError):
        os.kill(int(running["pid"]), signal.SIGTERM)
    return running


def _stop(proc: subprocess.Popen[bytes]) -> None:
    """Stop the driver and everything it started (its own process group)."""
    for sig in (signal.SIGTERM, signal.SIGKILL):
        with contextlib.suppress(ProcessLookupError, PermissionError):
            os.killpg(proc.pid, sig)
        try:
            proc.wait(timeout=10)
            return
        except subprocess.TimeoutExpired:
            continue


_LIMIT_RE = re.compile(
    r"(hit your (?:weekly|daily|usage|session) limit|usage limit (?:reached|exceeded)|"
    r"rate limit(?:ed| reached| exceeded)|out of (?:credits|usage)|quota exceeded)",
    re.IGNORECASE,
)
_RESET_RE = re.compile(
    r"resets?\s+(?:on\s+|at\s+)?([A-Za-z]{3}\s+\d{1,2}\s+at\s+)?(\d{1,2})(?::(\d{2}))?\s*(am|pm)",
    re.I,
)


def usage_limit_in(text: str, now: datetime | None = None) -> tuple[str, datetime] | None:
    """A driver's "you've hit your limit" message, and when it resets (best effort).

    Returns (message, resets_at). Without a readable reset time it assumes six hours.
    """
    from datetime import timedelta

    hit = _LIMIT_RE.search(text)
    if not hit:
        return None
    now = now or datetime.now()
    line = next((ln.strip() for ln in text.splitlines() if hit.group(0) in ln), hit.group(0))
    resets = now + timedelta(hours=6)
    m = _RESET_RE.search(line)
    if m:
        hour = int(m.group(2)) % 12 + (12 if m.group(4).lower() == "pm" else 0)
        at = now.replace(hour=hour, minute=int(m.group(3) or 0), second=0, microsecond=0)
        while at <= now:
            at += timedelta(days=1)
        resets = at
    return line[:200], resets


def usage_limited_until(artifacts_dir: Path) -> tuple[datetime, str] | None:
    """When the driver's usage limit lifts, if a launch hit one that has not lifted yet."""
    hit = load_state(artifacts_dir).get("usage_limit")
    if not isinstance(hit, dict):
        return None
    try:
        until = datetime.fromisoformat(str(hit.get("until")))
    except ValueError:
        return None
    if until <= datetime.now():
        return None
    return until, str(hit.get("driver") or "")


def run_child(artifacts_dir: Path, slug: str) -> int:
    """Run one launch in the foreground with its time limit. Entry point of the child."""
    cfg = AutoRunConfig.load(artifacts_dir)
    before = rounds_total(artifacts_dir)
    log_path = artifacts_dir / LOG_FILE
    log_start = log_path.stat().st_size if log_path.exists() else 0
    cmd = build_command(artifacts_dir, slug, cfg)
    print(
        f"[{datetime.now().isoformat(timespec='seconds')}] auto-run {cfg.driver}: "
        f"{cmd[0]} … ({cfg.rounds_per_launch} round(s), limit {cfg.max_minutes} min)",
        flush=True,
    )
    outcome: dict[str, Any] = {}
    try:
        proc = subprocess.Popen(
            cmd,
            cwd=str(artifacts_dir.parent),
            env=child_env(artifacts_dir, cfg.driver),
            stdin=subprocess.DEVNULL,
            start_new_session=True,
        )
    except OSError as exc:
        proc = None
        outcome = {"status": "failed", "error": f"could not start {cmd[0]}: {exc}"}
    if proc is not None:
        signal.signal(signal.SIGTERM, _on_sigterm)
        try:
            code = proc.wait(timeout=cfg.max_minutes * 60)
            outcome = {"status": "finished" if code == 0 else "failed", "exit_code": code}
            if code != 0:
                outcome["error"] = f"{cfg.driver} exited with code {code}; see {LOG_FILE}"
        except subprocess.TimeoutExpired:
            _stop(proc)
            outcome = {"status": "timed_out", "error": f"stopped after {cfg.max_minutes} min"}
        except _StopRequested:
            signal.signal(signal.SIGTERM, signal.SIG_DFL)
            _stop(proc)
            outcome = {"status": "stopped", "error": "stopped on request"}
    outcome["rounds_run"] = max(0, rounds_total(artifacts_dir) - before)
    outcome["finished_at"] = datetime.now().isoformat(timespec="seconds")
    # A usage limit is not a crash: say so, and wait for the reset instead of retrying.
    try:
        with log_path.open("rb") as fh:
            fh.seek(log_start)
            said = fh.read().decode("utf-8", "replace")
    except OSError:
        said = ""
    limit = usage_limit_in(said) if outcome.get("status") != "finished" else None
    if limit is not None:
        message, until = limit
        outcome["status"] = "usage_limit"
        outcome["error"] = f"{message} — auto-run waits until {until:%a %H:%M}"
        state = load_state(artifacts_dir)
        state["usage_limit"] = {
            "driver": cfg.driver,
            "message": message,
            "until": until.isoformat(timespec="minutes"),
        }
        _save_state(artifacts_dir, state)
    _finish(artifacts_dir, slug, outcome)
    print(
        f"[{outcome['finished_at']}] auto-run {outcome['status']}: "
        f"{outcome['rounds_run']} round(s)",
        flush=True,
    )
    return 0 if outcome["status"] == "finished" else 1
