"""Google Colab as an unattended compute target, through Google's Colab CLI.

Each round rents a fresh Colab VM (``colab new``), uploads the round's script and
starts it in the background there, checks on it every minute with short ``colab
exec`` calls, then downloads its output and submission file and releases the VM
(``colab stop``). ResearchForge wraps each round's script so the VM first
downloads the competition data (with your Kaggle key, passed as environment
variables), then runs the round from that folder. The output is read for the
metric the same way as a local run.

Setup is part of the flow, never a manual step: choosing Colab (at setup or in
Run configs) offers to install the CLI (``uv tool install google-colab-cli``),
then to sign in — a Google link plus a pasted code, from the dashboard or with
``researchforge kaggle driver-login colab``. Google's CLI keeps the token in the
macOS keychain. The plan and GPU are per-project settings (``colab_config.json``).

Without the CLI or a sign-in, Colab stays a manual target (a generated notebook
you run yourself), and auto-run does not start rounds on it.
"""

from __future__ import annotations

import contextlib
import json
import os
import re
import secrets
import shutil
import subprocess
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from researchforge.kaggle.workspace import atomic_write_json, read_json

CONFIG_FILE = "colab_config.json"
PACKAGE = "google-colab-cli"
GPUS = ("T4", "L4", "G4", "A100", "H100")
PLANS = ("free", "pro", "pro_plus", "payg", "enterprise")
PLAN_LABELS = {
    "free": "Free",
    "pro": "Colab Pro",
    "pro_plus": "Colab Pro+",
    "payg": "Pay as you go",
    "enterprise": "Colab Enterprise (Google Cloud)",
}
# What each plan can ask for. Free Colab gives T4 when available; the larger GPUs
# and high-RAM shapes spend compute units, which free accounts do not have.
PLAN_GPUS = {"free": ("T4",), "pro": GPUS, "pro_plus": GPUS, "payg": GPUS, "enterprise": ()}
PLAN_HIGH_MEM = {"pro", "pro_plus"}


@dataclass
class ColabConfig:
    plan: str = "free"
    gpu: str = "T4"
    high_mem: bool = False

    def save(self, artifacts_dir: Path) -> None:
        atomic_write_json(artifacts_dir / CONFIG_FILE, asdict(self))

    @classmethod
    def load(cls, artifacts_dir: Path) -> ColabConfig:
        raw = read_json(artifacts_dir / CONFIG_FILE, {}) or {}
        cfg = cls(**{k: v for k, v in raw.items() if k in cls.__dataclass_fields__})
        if cfg.plan not in PLANS:
            cfg.plan = "free"
        allowed = PLAN_GPUS.get(cfg.plan, ())
        if cfg.gpu not in allowed:
            cfg.gpu = allowed[0] if allowed else ""
        if cfg.plan not in PLAN_HIGH_MEM:
            cfg.high_mem = False
        return cfg


def set_config(
    artifacts_dir: Path,
    *,
    plan: str | None = None,
    gpu: str | None = None,
    high_mem: bool | None = None,
    via: str = "",
) -> ColabConfig:
    """Change the plan / GPU / high-RAM choice; refuses what the plan cannot get."""
    from researchforge.kaggle.run_config import record_change

    cfg = ColabConfig.load(artifacts_dir)
    before = asdict(cfg)
    if plan is not None:
        if plan not in PLANS:
            raise ValueError(f"plan must be one of {', '.join(PLANS)}")
        cfg.plan = plan
    if gpu is not None:
        gpu = gpu.upper()
        if gpu not in PLAN_GPUS.get(cfg.plan, ()):
            allowed = ", ".join(PLAN_GPUS.get(cfg.plan, ())) or "none"
            raise ValueError(f"{PLAN_LABELS[cfg.plan]} can use: {allowed}")
        cfg.gpu = gpu
    if high_mem is not None:
        if high_mem and cfg.plan not in PLAN_HIGH_MEM:
            raise ValueError("high-RAM machines need Colab Pro or Pro+")
        cfg.high_mem = high_mem
    cfg.save(artifacts_dir)
    cfg = ColabConfig.load(artifacts_dir)
    for key, value in asdict(cfg).items():
        record_change(artifacts_dir, f"colab_{key}", before.get(key), value, via)
    return cfg


# ---------------------------------------------------------------------------
# The CLI: find, install, sign-in status, usage
# ---------------------------------------------------------------------------


def find_cli() -> str:
    found = shutil.which("colab")
    if found:
        return found
    local = Path.home() / ".local" / "bin" / "colab"
    return str(local) if local.is_file() else ""


def find_uv() -> str:
    found = shutil.which("uv")
    if found:
        return found
    local = Path.home() / ".local" / "bin" / "uv"
    return str(local) if local.is_file() else ""


def install() -> tuple[bool, str]:
    """Install Google's Colab CLI with uv (or pip). Returns (ok, last output)."""
    uv = find_uv()
    cmd = (
        [uv, "tool", "install", PACKAGE]
        if uv
        else ["python3", "-m", "pip", "install", "--user", PACKAGE]
    )
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=900, check=False)
    except (OSError, subprocess.TimeoutExpired) as exc:
        return False, str(exc)
    out = (proc.stdout + proc.stderr).strip().splitlines()
    ok = proc.returncode == 0 and bool(find_cli())
    return ok, out[-1] if out else ""


def status(*, check_login: bool = True) -> dict[str, Any]:
    """Installed? Signed in, as whom? Never blocks on a sign-in prompt."""
    cli = find_cli()
    if not cli:
        return {
            "installed": False,
            "signed_in": False,
            "account": "",
            "detail": "Google's Colab CLI is not installed",
            "fix": "install it from Run configs, or: uv tool install google-colab-cli",
        }
    if not check_login:
        return {
            "installed": True,
            "signed_in": None,
            "account": "",
            "detail": "installed",
            "fix": "",
        }
    try:
        proc = subprocess.run(
            [cli, "whoami"],
            capture_output=True,
            text=True,
            timeout=30,
            stdin=subprocess.DEVNULL,
            check=False,
        )
        out = proc.stdout + proc.stderr
    except (OSError, subprocess.TimeoutExpired) as exc:
        out, proc = str(exc), None
    if "authorize" in out.lower() or proc is None or proc.returncode != 0:
        return {
            "installed": True,
            "signed_in": False,
            "account": "",
            "detail": "not signed in to Colab",
            "fix": "sign in from Run configs, or in your own terminal: "
            "researchforge kaggle driver-login colab",
        }
    match = re.search(r"[\w.+-]+@[\w-]+\.[\w.]+", out)
    return {
        "installed": True,
        "signed_in": True,
        "account": match.group(0) if match else "",
        "detail": f"signed in{' as ' + match.group(0) if match else ''}",
        "fix": "",
    }


def usage() -> dict[str, str]:
    """Compute-unit balance and burn rate, as Colab reports them ({} if unknown)."""
    cli = find_cli()
    if not cli:
        return {}
    try:
        proc = subprocess.run(
            [cli, "usage"],
            capture_output=True,
            text=True,
            timeout=30,
            stdin=subprocess.DEVNULL,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return {}
    if proc.returncode != 0:
        return {}
    out: dict[str, str] = {}
    for line in proc.stdout.splitlines():
        if "balance" in line.lower():
            out["balance"] = line.split(":", 1)[-1].strip()
        elif "rate" in line.lower():
            out["rate"] = line.split(":", 1)[-1].strip()
    return out


_STATUS_CACHE: dict[str, tuple[float, Any]] = {}


def cached(name: str, ttl: float = 120.0) -> Any:
    """status() / usage() for views that refresh often; each call spawns the CLI."""
    import time

    hit = _STATUS_CACHE.get(name)
    if hit is None or time.time() - hit[0] > ttl:
        hit = (time.time(), status() if name == "status" else usage())
        _STATUS_CACHE[name] = hit
    return hit[1]


def forget_cache() -> None:
    _STATUS_CACHE.clear()


def ready() -> tuple[bool, str, str]:
    """Can rounds run on Colab unattended right now? (ok, detail, fix)."""
    st = status()
    return bool(st["signed_in"]), str(st["detail"]), str(st["fix"])


class SignIn:
    """A Colab sign-in driven from the dashboard: show Google's link, take the code.

    Google's CLI prints an authorization URL and waits for the code on stdin; the
    dashboard keeps that process between the two requests.
    """

    def __init__(self) -> None:
        self._proc: subprocess.Popen[str] | None = None
        self._lines: list[str] = []
        self._lock = threading.Lock()

    def start(self, timeout: float = 30.0) -> str:
        cli = find_cli()
        if not cli:
            raise ValueError("install Google's Colab CLI first")
        self.cancel()
        self._lines = []
        self._proc = subprocess.Popen(
            [cli, "whoami"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        proc = self._proc

        def pump() -> None:
            assert proc.stdout is not None
            for line in proc.stdout:
                with self._lock:
                    self._lines.append(line)

        threading.Thread(target=pump, daemon=True).start()
        deadline = timeout
        while deadline > 0:
            with self._lock:
                text = "".join(self._lines)
            url = re.search(r"https://accounts\.google\.com/\S+", text)
            if url:
                return url.group(0)
            if proc.poll() is not None:
                if proc.returncode == 0:
                    return ""  # already signed in
                raise ValueError(text.strip()[-300:] or "the Colab CLI exited")
            threading.Event().wait(0.2)
            deadline -= 0.2
        raise ValueError("the Colab CLI did not print a sign-in link")

    def finish(self, code: str, timeout: float = 90.0) -> dict[str, Any]:
        proc = self._proc
        if proc is None or proc.poll() is not None:
            raise ValueError("start the sign-in first")
        assert proc.stdin is not None
        proc.stdin.write(code.strip() + "\n")
        proc.stdin.flush()
        try:
            proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            self.cancel()
            raise ValueError("Colab did not accept the code in time") from None
        self._proc = None
        if proc.returncode != 0:
            with self._lock:
                tail = "".join(self._lines).strip()[-300:]
            raise ValueError(f"sign-in failed: {tail}")
        return status()

    def cancel(self) -> None:
        if self._proc is not None and self._proc.poll() is None:
            with contextlib.suppress(OSError):
                self._proc.kill()
        self._proc = None


# ---------------------------------------------------------------------------
# Running a round
# ---------------------------------------------------------------------------

_WRAPPER = """\
# Generated by ResearchForge: one round on a fresh Colab VM.
import os, pathlib, subprocess, sys, zipfile

DATA = pathlib.Path("/content/rf_data")
if not any(DATA.glob("*")):
    DATA.mkdir(parents=True, exist_ok=True)
    print("[researchforge] downloading competition data on the VM", flush=True)
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "kaggle"], check=True)
    got = subprocess.run(["kaggle", "competitions", "download", "-c", {slug!r}, "-p", str(DATA)],
                         capture_output=True, text=True)
    if got.returncode != 0:
        sys.exit("[researchforge] data download failed: " + (got.stdout + got.stderr)[-500:])
    for z in DATA.glob("*.zip"):
        zipfile.ZipFile(z).extractall(DATA)
        z.unlink()
os.chdir(DATA)
sys.argv = ["experiment.py"] + sys.argv[1:]
CODE = {code!r}
exec(compile(CODE, "experiment.py", "exec"), {{"__name__": "__main__"}})
"""


def kaggle_env() -> dict[str, str]:
    """Kaggle credentials for the VM, the way this machine signs in to Kaggle.

    Kaggle's current CLI signs in with an API token (``KAGGLE_API_TOKEN``, or the
    ``access_token`` file it writes); the legacy username + key in ``kaggle.json``
    may be revoked. The token is preferred, the legacy pair is the fallback.
    """
    home = Path.home()
    token = os.environ.get("KAGGLE_API_TOKEN", "")
    if not token:
        for path in (
            home / ".kaggle" / "access_token",
            home / ".config" / "kaggle" / "access_token",
        ):
            with contextlib.suppress(OSError):
                token = path.read_text().strip()
            if token:
                break
    if token:
        return {"KAGGLE_API_TOKEN": token}
    user, key = os.environ.get("KAGGLE_USERNAME", ""), os.environ.get("KAGGLE_KEY", "")
    if not (user and key):
        for path in (home / ".kaggle" / "kaggle.json", home / ".config" / "kaggle" / "kaggle.json"):
            try:
                creds = json.loads(path.read_text())
            except (OSError, ValueError):
                continue
            user, key = str(creds.get("username") or ""), str(creds.get("key") or "")
            if user and key:
                break
    return {"KAGGLE_USERNAME": user, "KAGGLE_KEY": key} if user and key else {}


# A round runs detached on the VM, so it does not depend on one long-held connection:
# `colab run` keeps a single websocket whose access token expires after about 30
# minutes and is never renewed, which cut every longer round off ("Connection was
# lost"). Each short `colab exec` below renews the token.
REMOTE_SCRIPT = "/content/rf_round.py"
REMOTE_DIR = "/content/rf_round"
REMOTE_DATA = "/content/rf_data"  # the wrapper's working folder (see _WRAPPER)
POLL_SECONDS = 60.0
# Checks that may fail in a row (a dropped connection, a slow VM) before the round is
# given up as lost: about a quarter of an hour without contact.
MAX_FAILED_POLLS = 15

_START = f"""\
import os, pathlib, subprocess, sys, time
R = pathlib.Path({REMOTE_DIR!r})
R.mkdir(parents=True, exist_ok=True)
for f in ("rc", "out.log", "pid"):
    (R / f).unlink(missing_ok=True)
(R / "started").write_text(str(time.time()))
cmd = (f"{{sys.executable}} -u {REMOTE_SCRIPT} > {{R}}/out.log 2>&1; "
       f"echo $? > {{R}}/rc.tmp && mv {{R}}/rc.tmp {{R}}/rc")
p = subprocess.Popen(["bash", "-c", cmd], start_new_session=True, env=dict(os.environ),
                     cwd="/content", stdin=subprocess.DEVNULL,
                     stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
(R / "pid").write_text(str(p.pid))
print("RF_STARTED", p.pid, flush=True)
"""

_POLL = f"""\
import json, os, pathlib
R = pathlib.Path({REMOTE_DIR!r})
rc = (R / "rc").read_text().strip() if (R / "rc").exists() else None
alive = False
if (R / "pid").exists():
    try:  # a finished child the kernel never reaped is a zombie ("Z"), not alive
        pid = int((R / "pid").read_text())
        stat = pathlib.Path("/proc/" + str(pid) + "/stat").read_text()
        alive = stat.rsplit(")", 1)[1].split()[0] != "Z"
    except (OSError, ValueError, IndexError):
        pass
log = R / "out.log"
print("RF_POLL " + json.dumps({{
    "done": rc is not None or not alive,
    "rc": int(rc) if rc and rc.lstrip("-").isdigit() else None,
    "bytes": log.stat().st_size if log.exists() else 0,
}}), flush=True)
"""

_STOP = f"""\
import os, pathlib, signal, time
R = pathlib.Path({REMOTE_DIR!r})
try:
    pid = int((R / "pid").read_text())
    os.killpg(pid, signal.SIGTERM)
    time.sleep(5)
    os.killpg(pid, signal.SIGKILL)
except (OSError, ValueError):
    pass
print("RF_STOPPED", flush=True)
"""

# The file the round wrote for Kaggle: the one it named, else a fresh submission.*.
_FIND_SUBMISSION = f"""\
import json, pathlib, re
R, D = pathlib.Path({REMOTE_DIR!r}), pathlib.Path({REMOTE_DATA!r})
log = R / "out.log"
text = log.read_text(errors="replace") if log.exists() else ""
started = float((R / "started").read_text()) if (R / "started").exists() else 0.0
named = re.findall(r"^SUBMISSION_FILE:\\s*(\\S.*?)\\s*$", text, re.M)
found = []
if named:
    p = pathlib.Path(named[-1])
    found.append(p if p.is_absolute() else D / p)
found += sorted(D.glob("submission.*"), key=lambda p: p.stat().st_mtime, reverse=True)
pick = next((p for p in found if p.is_file() and p.stat().st_mtime >= started), None)
print("RF_SUB " + json.dumps(str(pick) if pick else ""), flush=True)
"""


def _cli(args: list[str], timeout: float) -> tuple[int | None, str, str]:
    """Run one Colab CLI command: (exit code or None when it could not finish, out, err)."""
    try:
        proc = subprocess.run(
            [find_cli() or "colab", *args],
            capture_output=True,
            text=True,
            timeout=timeout,
            stdin=subprocess.DEVNULL,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return None, "", f"colab {args[0]} did not answer within {int(timeout)}s"
    except OSError as exc:
        return None, "", str(exc)
    return proc.returncode, proc.stdout or "", proc.stderr or ""


def _exec(
    session: str, code: str, workdir: Path, timeout: float, env: dict[str, str] | None = None
) -> tuple[int | None, str, str]:
    """Run a short snippet in the session's kernel; returns what ``_cli`` does."""
    snippet = workdir / f"rf_{secrets.token_hex(3)}.py"
    snippet.write_text(code)
    args = ["exec", "-s", session, "-f", str(snippet), "--timeout", str(int(timeout))]
    for k, v in (env or {}).items():
        args += ["--env", f"{k}={v}"]
    try:
        return _cli(args, timeout + 60)
    finally:
        snippet.unlink(missing_ok=True)


def _marker(out: str, name: str) -> Any:
    for line in reversed(out.splitlines()):
        if line.startswith(name + " "):
            with contextlib.suppress(ValueError):
                return json.loads(line[len(name) + 1 :])
    return None


def run_round(
    code: str,
    slug: str,
    cfg: ColabConfig,
    workdir: Path,
    timeout_seconds: float,
    *,
    poll_seconds: float | None = None,
    sleep: Any = time.sleep,
) -> dict[str, Any]:
    """Run one round's script on a fresh Colab VM, detached, and release the VM.

    The VM downloads the competition data and runs the round in the background;
    this side checks on it every ``poll_seconds`` until it ends or ``timeout_seconds``
    pass, so a dropped connection costs one check, not the round. The round's output
    comes back as ``stdout_text``; a submission file it wrote is downloaded into
    ``workdir`` (``submission_file``) so the round can be offered for submission.
    """
    poll_seconds = POLL_SECONDS if poll_seconds is None else poll_seconds
    workdir.mkdir(parents=True, exist_ok=True)
    wrapper = workdir / "colab_round.py"
    wrapper.write_text(_WRAPPER.format(slug=slug, code=code))
    env = kaggle_env()
    fail: dict[str, Any] = {
        "ok": False,
        "exit_code": None,
        "stdout_text": "",
        "submission_file": "",
    }
    if not env:
        return {**fail, "stderr_text": "no Kaggle key found to download the data on the VM"}
    session = f"rf-{secrets.token_hex(4)}"
    new = ["new", "-s", session]
    if cfg.gpu:
        new += ["--gpu", cfg.gpu]
    if cfg.high_mem:
        new.append("--high-mem")
    rc, out, err = _cli(new, 900)
    if rc != 0:
        return {**fail, "stderr_text": f"Colab could not start a VM: {(err or out).strip()[-400:]}"}
    notes: list[str] = []
    exit_code: int | None = None
    log_text = ""
    submission = ""
    try:
        rc, out, err = _cli(["upload", "-s", session, str(wrapper), REMOTE_SCRIPT], 300)
        if rc != 0:
            return {
                **fail,
                "stderr_text": f"upload to the VM failed: {(err or out).strip()[-400:]}",
            }
        rc, out, err = _exec(session, _START, workdir, 120, env)
        if rc != 0 or "RF_STARTED" not in out:
            return {
                **fail,
                "stderr_text": f"the round did not start: {(err or out).strip()[-400:]}",
            }
        deadline = time.monotonic() + timeout_seconds
        failed = 0
        while True:
            left = deadline - time.monotonic()
            if left <= 0:
                _exec(session, _STOP, workdir, 60)
                notes.append(f"stopped at the round time limit ({int(timeout_seconds // 60)} min)")
                break
            sleep(min(poll_seconds, left))
            rc, out, err = _exec(session, _POLL, workdir, 90)
            info = _marker(out, "RF_POLL") if rc == 0 else None
            if not isinstance(info, dict):
                failed += 1
                if failed >= MAX_FAILED_POLLS:
                    notes.append(
                        f"lost contact with the Colab VM for {failed} checks in a row: "
                        f"{(err or out).strip()[-200:]}"
                    )
                    break
                continue
            failed = 0
            if info.get("done"):
                exit_code = info.get("rc")
                if exit_code is None:
                    notes.append("the round's process ended without an exit code")
                break
        local_log = workdir / "out.log"
        rc, out, err = _cli(
            ["download", "-s", session, f"{REMOTE_DIR}/out.log", str(local_log)], 600
        )
        if rc == 0 and local_log.is_file():
            log_text = local_log.read_text(errors="replace")
        else:
            notes.append(f"could not fetch the round's output: {(err or out).strip()[-200:]}")
        if exit_code == 0:
            rc, out, _err = _exec(session, _FIND_SUBMISSION, workdir, 60)
            remote = _marker(out, "RF_SUB") if rc == 0 else ""
            if isinstance(remote, str) and remote:
                local = workdir / Path(remote).name
                rc, out, err = _cli(["download", "-s", session, remote, str(local)], 900)
                if rc == 0 and local.is_file():
                    submission = str(local)
                else:
                    notes.append(f"could not fetch {remote}: {(err or out).strip()[-200:]}")
    finally:
        _cli(["stop", "-s", session], 120)
    return {
        "ok": exit_code == 0,
        "exit_code": exit_code,
        "stdout_text": log_text,
        "stderr_text": "; ".join(notes) or (log_text.strip()[-400:] if exit_code else ""),
        "submission_file": submission,
    }
