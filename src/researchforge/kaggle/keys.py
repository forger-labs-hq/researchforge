"""Keys and accounts: every credential ResearchForge uses, seen and set in one place.

The dashboard's Keys page and ``researchforge kaggle keys`` both use this module, so a
key set in one is ready in the other. A value goes to its usual home — Kaggle's own
files under ``~/.kaggle``, everything else in the auto-run env file (mode 600) that
unattended runs and the background monitor read — and is never read back out: status
reports whether a key is set, where, and what the last test call said.

An agent in a chat never handles a value. It asks which accounts the person wants,
opens this page (``keys open <name>``) and waits for it to turn ready (``keys wait``).
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from researchforge.kaggle.workspace import atomic_write_json, config_dir, read_json


@dataclass(frozen=True)
class Slot:
    name: str
    label: str
    group: str  # kaggle | ai | driver | compute
    env: str = ""  # the env var it is stored as ("" = its own files or sign-in)
    get_it: str = ""  # where the person gets the value
    used_for: str = ""
    settable: bool = True  # False: signed in elsewhere (Colab, AWS); status only


SLOTS: dict[str, Slot] = {
    s.name: s
    for s in (
        Slot(
            "kaggle",
            "Kaggle API token",
            "kaggle",
            get_it="kaggle.com/settings → API → Create New Token (a KGAT_… token, or the "
            "legacy kaggle.json username + key)",
            used_for="competition data, notebooks, Kaggle GPU rounds and submissions",
        ),
        Slot(
            "anthropic",
            "Anthropic API key",
            "ai",
            env="ANTHROPIC_API_KEY",
            get_it="console.anthropic.com → API keys",
            used_for="the API-key driver and sensors' AI calls; billed per token",
        ),
        Slot(
            "gemini",
            "Gemini API key",
            "ai",
            env="GEMINI_API_KEY",
            get_it="aistudio.google.com → Get API key",
            used_for="the API-key driver and sensors' AI calls; billed per token",
        ),
        Slot(
            "openai",
            "OpenAI API key",
            "ai",
            env="OPENAI_API_KEY",
            get_it="platform.openai.com → API keys",
            used_for="the API-key driver and sensors' AI calls; billed per token",
        ),
        Slot(
            "claude",
            "Claude Code token",
            "driver",
            env="CLAUDE_CODE_OAUTH_TOKEN",
            get_it="run `claude setup-token` in a terminal, sign in, and paste the token it prints",
            used_for="unattended rounds with the Claude Code driver, on your Claude plan",
        ),
        Slot(
            "cursor",
            "Cursor API key",
            "driver",
            env="CURSOR_API_KEY",
            get_it="cursor.com → Dashboard → Integrations → API keys",
            used_for="unattended rounds with the Cursor agent driver",
        ),
        Slot(
            "colab",
            "Google Colab",
            "compute",
            get_it="sign in to Google from this page; it uses Google's Colab CLI",
            used_for="rounds on Colab GPUs",
            settable=False,
        ),
        Slot(
            "runpod",
            "RunPod API key",
            "compute",
            env="RUNPOD_API_KEY",
            get_it="runpod.io → Settings → API Keys",
            used_for="rounds on paid RunPod GPUs",
        ),
        Slot(
            "lambda",
            "Lambda API key",
            "compute",
            env="LAMBDA_API_KEY",
            get_it="cloud.lambdalabs.com → API keys",
            used_for="rounds on paid Lambda GPUs",
        ),
        Slot(
            "vast",
            "vast.ai API key",
            "compute",
            env="VAST_API_KEY",
            get_it="cloud.vast.ai → Account → API Keys",
            used_for="rounds on paid vast.ai GPUs",
        ),
        Slot(
            "aws",
            "AWS credentials",
            "compute",
            get_it="run `aws configure` in your own terminal",
            used_for="rounds on EC2 GPUs",
            settable=False,
        ),
    )
}

ENV_SLOT_KEYS = tuple(s.env for s in SLOTS.values() if s.env)


def _kaggle_dir() -> Path:
    return Path.home() / ".kaggle"  # looked up per call: tests and other homes stay apart


def _tests_path() -> Path:
    return config_dir() / "key_tests.json"


def _slot(name: str) -> Slot:
    slot = SLOTS.get(name.strip().lower())
    if slot is None:
        raise KeyError(f"unknown key {name!r}; one of: {', '.join(SLOTS)}")
    return slot


# ---------------------------------------------------------------------------
# Status (names and places only, never values)
# ---------------------------------------------------------------------------


def _where(slot: Slot) -> str:
    """Where the value is set, or "" when it is not."""
    from researchforge.kaggle.auto_run import env_file, load_env_file

    if slot.name == "kaggle":
        token = _kaggle_dir() / "access_token"
        if os.environ.get("KAGGLE_API_TOKEN"):
            return "KAGGLE_API_TOKEN in the environment"
        if token.is_file() and token.read_text().strip():
            return str(token)
        legacy = read_json(_kaggle_dir() / "kaggle.json", None)
        if isinstance(legacy, dict) and legacy.get("username") and legacy.get("key"):
            return str(_kaggle_dir() / "kaggle.json")
        return ""
    if slot.name == "colab":
        from researchforge.kaggle import colab

        st = colab.cached("status", ttl=120.0)
        return (
            f"Colab CLI{' as ' + st['account'] if st.get('account') else ''}"
            if st.get("signed_in")
            else ""
        )
    if slot.name == "aws":
        # Configured, not verified: proving it needs a network call, which Test makes.
        configured = (
            os.environ.get("AWS_ACCESS_KEY_ID") or (Path.home() / ".aws" / "credentials").is_file()
        )
        return "aws CLI" if shutil.which("aws") and configured else ""
    values, _problem = load_env_file()
    if values.get(slot.env):
        return str(env_file())
    if os.environ.get(slot.env):
        return "this process's environment only (unattended runs may not see it)"
    return ""


def _aws_identity() -> tuple[bool, str]:
    try:
        proc = subprocess.run(
            ["aws", "sts", "get-caller-identity", "--output", "json"],
            capture_output=True,
            text=True,
            timeout=20,
            stdin=subprocess.DEVNULL,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return False, str(exc)
    if proc.returncode != 0:
        return False, (proc.stderr.strip().splitlines() or ["not signed in"])[-1]
    try:
        return True, str(json.loads(proc.stdout).get("Arn", "signed in"))
    except ValueError:
        return True, "signed in"


def status(name: str | None = None) -> list[dict[str, Any]]:
    """One row per key: label, whether it is set and where, and the last test result."""
    tests = read_json(_tests_path(), {}) or {}
    rows = []
    for slot in SLOTS.values() if name is None else [_slot(name)]:
        where = _where(slot)
        test = tests.get(slot.name) if isinstance(tests, dict) else None
        rows.append(
            {
                **asdict(slot),
                "set": bool(where),
                "where": where,
                # A test of an older value says nothing about the one set now.
                "test": test if isinstance(test, dict) and where else None,
            }
        )
    return rows


def ready(name: str) -> bool:
    row = status(name)[0]
    return bool(row["set"]) and (row["test"] is None or bool(row["test"].get("ok")))


# ---------------------------------------------------------------------------
# Set / remove
# ---------------------------------------------------------------------------


def set_key(name: str, value: str, username: str = "") -> str:
    """Store one key where ResearchForge reads it. Returns where (never the value)."""
    from researchforge.kaggle.auto_run import store_key

    slot = _slot(name)
    value = value.strip()
    if not slot.settable:
        raise ValueError(f"{slot.label} is signed in, not pasted: {slot.get_it}")
    if not value or "\n" in value:
        raise ValueError("paste the value as one line")
    _forget_test(slot.name)
    if slot.name == "kaggle":
        return _set_kaggle(value, username.strip())
    return str(store_key(slot.env, value))


def _set_kaggle(value: str, username: str) -> str:
    _kaggle_dir().mkdir(mode=0o700, exist_ok=True)
    legacy: dict[str, str] | None = None
    if value.startswith("{"):
        try:
            raw = json.loads(value)
        except ValueError:
            raise ValueError("that is not valid kaggle.json content") from None
        if not (isinstance(raw, dict) and raw.get("username") and raw.get("key")):
            raise ValueError("kaggle.json needs both username and key")
        legacy = {"username": str(raw["username"]), "key": str(raw["key"])}
    elif username:
        legacy = {"username": username, "key": value}
    elif not value.startswith("KGAT_"):
        raise ValueError(
            "a legacy Kaggle key needs your Kaggle username too; a new API token starts with KGAT_"
        )
    path = _kaggle_dir() / ("kaggle.json" if legacy else "access_token")
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w") as fh:
        fh.write(json.dumps(legacy) if legacy else value)
    path.chmod(0o600)
    return str(path)


def remove_key(name: str) -> str:
    """Remove a key from the auto-run env file. Kaggle's own files are left alone."""
    from researchforge.kaggle.auto_run import env_file

    slot = _slot(name)
    if not slot.env:
        raise ValueError(f"{slot.label} is not stored by ResearchForge: {slot.get_it}")
    path = env_file()
    if not path.is_file():
        return "not set"
    lines = path.read_text().splitlines()
    kept = [ln for ln in lines if ln.strip().removeprefix("export ").partition("=")[0] != slot.env]
    if len(kept) == len(lines):
        return "not set"
    fd = os.open(path, os.O_WRONLY | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w") as fh:
        fh.write("\n".join(kept) + ("\n" if kept else ""))
    _forget_test(slot.name)
    return f"removed from {path}"


# ---------------------------------------------------------------------------
# Test: one harmless real call with the stored value
# ---------------------------------------------------------------------------


def _forget_test(name: str) -> None:
    tests = read_json(_tests_path(), {}) or {}
    if isinstance(tests, dict) and tests.pop(name, None) is not None:
        atomic_write_json(_tests_path(), tests)


def _value(slot: Slot) -> str:
    from researchforge.kaggle.auto_run import load_env_file

    return load_env_file()[0].get(slot.env) or os.environ.get(slot.env, "")


def _http_ok(url: str, headers: dict[str, str]) -> tuple[bool, str]:
    req = urllib.request.Request(url, headers={"User-Agent": "researchforge", **headers})
    try:
        with urllib.request.urlopen(req, timeout=20) as resp:
            return True, f"accepted (HTTP {resp.status})"
    except urllib.error.HTTPError as exc:
        return False, f"refused (HTTP {exc.code})"
    except (urllib.error.URLError, OSError) as exc:
        return False, f"could not reach the service: {exc}"


def _run(cmd: list[str], env: dict[str, str] | None = None, timeout: int = 60) -> tuple[bool, str]:
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            stdin=subprocess.DEVNULL,
            env={**os.environ, **(env or {})},
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return False, str(exc)
    out = (proc.stdout + proc.stderr).strip().splitlines()
    return proc.returncode == 0, (out[-1] if out else f"exit {proc.returncode}")[:200]


def _test(slot: Slot) -> tuple[bool, str]:
    if slot.name == "kaggle":
        ok, last = _run(["kaggle", "competitions", "list", "--page-size", "1"])
        return ok, "Kaggle accepted it" if ok else last
    if slot.name == "colab":
        from researchforge.kaggle import colab

        colab.forget_cache()
        st = colab.status()
        return bool(st["signed_in"]), str(st["detail"])
    if slot.name == "aws":
        return _aws_identity()
    value = _value(slot)
    if not value:
        return False, "not set"
    if slot.name == "anthropic":
        return _http_ok(
            "https://api.anthropic.com/v1/models",
            {"x-api-key": value, "anthropic-version": "2023-06-01"},
        )
    if slot.name == "gemini":
        return _http_ok(
            "https://generativelanguage.googleapis.com/v1beta/models", {"x-goog-api-key": value}
        )
    if slot.name == "openai":
        return _http_ok("https://api.openai.com/v1/models", {"Authorization": f"Bearer {value}"})
    if slot.name in ("runpod", "lambda", "vast"):
        from researchforge.kaggle.cloud_gpu_config import list_gpu_types

        try:
            types = list_gpu_types(slot.name, value)
        except Exception as exc:
            return False, f"refused: {exc}"[:200]
        return bool(types), f"accepted — {len(types)} GPU types listed" if types else "no answer"
    from researchforge.kaggle.auto_run import find_binary

    binary = find_binary(slot.name)
    if not binary:
        return False, f"the {slot.name} CLI is not installed"
    if slot.name == "claude":
        # One tiny prompt on the token: the only check that proves it works headless.
        ok, last = _run(
            [binary, "-p", "Reply with the single word OK", "--output-format", "text"],
            {"CLAUDE_CODE_OAUTH_TOKEN": value},
            timeout=120,
        )
        return ok, "Claude Code answered on this token" if ok else last
    ok, last = _run([binary, "status"], {"CURSOR_API_KEY": value})
    return ok and "not logged in" not in last.lower(), last


def test_key(name: str) -> dict[str, Any]:
    """Make one harmless call with the stored key and remember the result."""
    slot = _slot(name)
    ok, detail = _test(slot)
    result = {"ok": ok, "detail": detail, "at": datetime.now().isoformat(timespec="seconds")}
    tests = read_json(_tests_path(), {}) or {}
    tests = tests if isinstance(tests, dict) else {}
    tests[slot.name] = result
    atomic_write_json(_tests_path(), tests)
    return result


def page_url(name: str = "", port: int | None = None) -> str:
    """The dashboard's Keys page, scrolled to one key."""
    from researchforge.kaggle.dashboard import _DEFAULT_PORT

    return f"http://localhost:{port or _DEFAULT_PORT}/#/keys" + (f"?n={name}" if name else "")
