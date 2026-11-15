"""Live local dashboard for ResearchForge (Kaggle overnight + repo projects).

A small service, not a static page:

* ``GET /`` serves one static app shell (``dashboard_app.html``). The app keeps
  its view / project / search in the URL hash, so reloads never reset to Home.
* ``GET /api/state`` returns the canonical payload from
  :func:`researchforge.kaggle.dashboard_state.build_state` with an ETag; the
  app polls it every few seconds and re-renders only when it changed.
* ``POST /api/*`` performs real actions — launchd install/uninstall through the
  CLI, sensor config in ``scheduler_config.json``, a one-off monitor cycle,
  a submissions refresh. POSTs require the per-process token embedded in the
  page and a localhost Host header (no CSRF / DNS-rebinding from other sites).

Usage:
    researchforge kaggle dashboard            # foreground on localhost:7842
    researchforge kaggle dashboard --daemon   # background (PID file)
    researchforge kaggle dashboard --stop
    researchforge kaggle dashboard --json     # print the payload
"""

from __future__ import annotations

import contextlib
import json
import os
import secrets
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any

from researchforge.kaggle import dashboard_state as _state
from researchforge.kaggle import keys as _keys
from researchforge.kaggle import new_project as _new
from researchforge.kaggle import sensors as _sensors
from researchforge.kaggle.workspace import (  # noqa: F401 (re-export)
    OVERNIGHT_PREFIX,
    config_dir,
    read_json,
    register_overnight_dir,
)

_DEFAULT_PORT = 7842


def _pid_file() -> Path:
    return config_dir() / "dashboard.pid"


_APP_HTML = Path(__file__).with_name("dashboard_app.html")
_STATE_TTL = 2.0  # rebuild the payload at most this often
_SUBMISSIONS_TTL = 15 * 60  # background Kaggle CLI refresh per slug


def _keys_fingerprint(rows: list[dict[str, Any]]) -> str:
    return "".join(
        ("1" if r["set"] else "0") + str((r.get("test") or {}).get("at", "")) for r in rows
    )


def collect_all(cwd: Path | None = None) -> dict[str, Any]:
    """Return the canonical dashboard payload (used by ``--json``)."""
    return _state.build_state(cwd or Path.cwd())


# ---------------------------------------------------------------------------
# Service
# ---------------------------------------------------------------------------


class DashboardService:
    """Owns the cached state, background submission fetches and actions."""

    def __init__(self, cwd: Path) -> None:
        self.cwd = cwd
        self.token = secrets.token_urlsafe(24)
        self.started_at = datetime.now().isoformat(timespec="seconds")
        self._lock = threading.Lock()
        self._state: dict[str, Any] | None = None
        self._state_at = 0.0
        self._subs: dict[str, dict[str, Any]] = {}
        self._subs_lock = threading.Lock()
        self._subs_version = 0
        self._actions: list[dict[str, Any]] = []
        self._actions_version = 0
        self._jobs: dict[str, dict[str, Any]] = {}  # setup wizard runs, newest last
        self._jobs_version = 0
        self._stop = threading.Event()

    # ── state ────────────────────────────────────────────────────────────
    def state(self, force: bool = False) -> dict[str, Any]:
        with self._lock:
            if force or self._state is None or time.time() - self._state_at > _STATE_TTL:
                with self._subs_lock:
                    live = dict(self._subs)
                payload = _state.build_state(self.cwd, live)
                payload["actions"] = self._actions[-20:]
                payload["keys"] = _keys.status()  # names and places only, never values
                payload["jobs"] = list(self._jobs.values())[-10:]
                payload["new_project"] = {
                    "default_root": str(self.cwd),
                    "sensors": _sensors.BUILTIN_SENSORS,
                    "computes": list(_new.COMPUTES),
                    "cloud_providers": list(_new.CLOUD_PROVIDERS),
                }
                payload["server"] = {
                    "pid": os.getpid(),
                    "started_at": self.started_at,
                    "cwd": str(self.cwd),
                }
                self._state, self._state_at = payload, time.time()
            return self._state

    def etag(self, payload: dict[str, Any]) -> str:
        # The token suffix makes a restarted server never answer 304 to an old tab.
        parts = (
            payload["fingerprint"],
            self._subs_version,
            self._actions_version,
            self._jobs_version,
            _keys_fingerprint(payload.get("keys") or []),
            self.token[:6],
        )
        return '"' + "-".join(str(x) for x in parts) + '"'

    def find_project(self, pid: str) -> dict[str, Any] | None:
        for p in self.state()["projects"]:
            if p["id"] == pid:
                found: dict[str, Any] = p
                return found
        return None

    # ── submissions (background, never on the request path) ──────────────
    def _fetch_subs(self, slug: str) -> None:
        from researchforge.kaggle.kernel_watcher import fetch_submissions

        rows, error = fetch_submissions(slug, timeout=30)
        with self._subs_lock:
            prev = self._subs.get(slug, {})
            self._subs[slug] = {
                "fetched_at": datetime.now().isoformat(timespec="seconds"),
                "error": error,
                "submissions": rows if not error else prev.get("submissions", []),
            }
            self._subs_version += 1

    def _subs_loop(self) -> None:
        while not self._stop.is_set():
            try:
                slugs = sorted(
                    {
                        p["slug"]
                        for p in self.state()["projects"]
                        if p["slug"] and p["role"] == "primary"
                    }
                )
                for slug in slugs:
                    if self._stop.is_set():
                        return
                    last = self._subs.get(slug, {}).get("fetched_at")
                    if (
                        last
                        and time.time() - datetime.fromisoformat(last).timestamp()
                        < _SUBMISSIONS_TTL
                    ):
                        continue
                    self._fetch_subs(slug)
            except Exception:
                pass
            self._stop.wait(30)

    def start_background(self) -> None:
        threading.Thread(target=self._subs_loop, name="rf-submissions", daemon=True).start()

    # ── actions ──────────────────────────────────────────────────────────
    def _record_action(self, entry: dict[str, Any]) -> dict[str, Any]:
        self._actions.append(entry)
        self._actions = self._actions[-50:]
        self._actions_version += 1
        return entry

    def _cli(
        self, project: dict[str, Any], args: list[str], timeout: int = 60
    ) -> subprocess.CompletedProcess[str]:
        """Run the real CLI pinned to this project's artifacts dir."""
        art = Path(project["overnight"]["artifacts_dir"])
        env = {
            **os.environ,
            "RESEARCHFORGE_OVERNIGHT_DIR": str(art),
            "RESEARCHFORGE_NO_DASHBOARD": "1",
        }
        return subprocess.run(
            [sys.executable, "-m", "researchforge", "kaggle", *args],
            capture_output=True,
            text=True,
            cwd=str(art.parent),
            env=env,
            timeout=timeout,
            check=False,
        )

    def action_scheduler(
        self, project: dict[str, Any], enabled: bool, interval_hours: float
    ) -> dict[str, Any]:
        slug = project["slug"]
        args = (
            [
                "scheduler-install",
                slug,
                "--interval-hours",
                str(max(0.25, min(float(interval_hours), 48.0))),
            ]
            if enabled
            else ["scheduler-uninstall", slug]
        )
        res = self._cli(project, args)
        _state.invalidate_launchd(slug)
        out = (res.stdout + res.stderr).strip()
        self._record_action(
            {
                "project": project["id"],
                "kind": "scheduler-install" if enabled else "scheduler-uninstall",
                "at": datetime.now().isoformat(timespec="seconds"),
                "returncode": res.returncode,
                "output": out[-600:],
            }
        )
        status = _state.launchd_status(slug, ttl=0)
        if res.returncode != 0:
            raise RuntimeError(out.splitlines()[-1] if out else f"exit {res.returncode}")
        if enabled and not status["loaded"]:
            raise RuntimeError(
                "plist written but launchd did not load it — " + (status["last_error"] or "see log")
            )
        return {"scheduler": status, "output": out.splitlines()[-1] if out else ""}

    def action_monitor(self, project: dict[str, Any]) -> dict[str, Any]:
        art = Path(project["overnight"]["artifacts_dir"])
        entry = self._record_action(
            {
                "project": project["id"],
                "kind": "monitor --once",
                "at": datetime.now().isoformat(timespec="seconds"),
                "returncode": None,
                "output": "",
            }
        )

        def run() -> None:
            try:
                res = self._cli(project, ["monitor", project["slug"], "--once"], timeout=900)
                entry["returncode"] = res.returncode
                entry["output"] = (res.stdout + res.stderr).strip()[-600:]
            except Exception as exc:
                entry["returncode"] = -1
                entry["output"] = str(exc)
            entry["finished_at"] = datetime.now().isoformat(timespec="seconds")
            self._actions_version += 1

        threading.Thread(target=run, daemon=True).start()
        return {"artifacts_dir": str(art)}

    def action_sensor(self, project: dict[str, Any], name: str, enabled: bool) -> dict[str, Any]:
        art = Path(project["overnight"]["artifacts_dir"])
        _sensors.set_enabled(art, project["slug"], name, enabled)
        return {"sensor": name, "enabled": enabled}

    def action_answer(self, project: dict[str, Any], body: dict[str, Any]) -> dict[str, Any]:
        """Answer one open question from the bell or the Home page."""
        from researchforge.kaggle import control

        qid, answer = str(body.get("id") or ""), str(body.get("answer") or "").strip()
        where = str(body.get("where") or "kaggle")
        if not qid or not (answer or body.get("dismiss")):
            raise ValueError("id and answer are required")
        if where == "kaggle":
            if not project.get("overnight"):
                raise ValueError("this project has no overnight loop")
            d = Path(project["overnight"]["artifacts_dir"])
        else:
            d = Path(project["root"]) / ".researchforge"
        item = next((q for q in control.open_interactions(d) if q.id == qid), None)
        if item is None:
            raise KeyError(f"no open question {qid}")
        comment = str(body.get("comment") or "")
        if body.get("dismiss"):
            control.dismiss(d, qid, "dismissed on the dashboard", comment)
            return {"dismissed": qid}
        if item.answer_key.startswith("judge:"):
            from researchforge.autorun.kaggle_mode import record_decision

            if answer not in ("yes", "no", "skip"):
                raise ValueError("answer must be yes, no or skip")
            record_decision(d, item.answer_key[6:], answer)  # type: ignore[arg-type]
            if comment:
                items = control.load_interactions(d)
                for i in items:
                    if i.id == qid:
                        i.comment = comment.strip()
                control._save_interactions(d, items)
        elif item.kind == "submission_approval" and where == "kaggle":
            from researchforge.kaggle.submission import answer as answer_submission

            line = answer_submission(d, str(project.get("slug") or ""), qid, answer, comment)
            return {"answered": qid, "answer": answer, "output": line}
        else:
            applied = control.respond(d, qid, answer, comment).applied
            return {"answered": qid, "answer": answer, "applied": applied}
        return {"answered": qid, "answer": answer}

    def action_plan_approve(self, project: dict[str, Any], body: dict[str, Any]) -> dict[str, Any]:
        """Approve or reject a repo plan for the person, with their comment.

        Approving needs the typed word "approve" — the same as `researchforge
        experiment approve`. Refused for a competition, whose only engine is its
        Kaggle loop. Rejecting cancels the plan.
        """
        from researchforge.kaggle import control
        from researchforge.kaggle.dashboard_state import _plans_awaiting_approval

        plan_id = str(body.get("plan") or "")
        decision = str(body.get("decision") or "approve")
        comment = str(body.get("comment") or "")
        if project.get("overnight"):
            raise ValueError(
                "this competition runs only on its Kaggle loop; repo plans are not run for it"
            )
        repo_dir = Path(project["root"]) / ".researchforge"
        if plan_id not in _plans_awaiting_approval(repo_dir):
            raise KeyError(f"{plan_id} is not waiting for approval")
        if decision == "approve":
            if str(body.get("typed") or "").strip().lower() != "approve":
                raise ValueError('type "approve" to approve a plan')
            cmd = ["experiment", "approve", plan_id, "--yes"]
        elif decision == "reject":
            cmd = ["experiment", "cancel", plan_id, "--yes"]
        else:
            raise ValueError("decision must be approve or reject")
        env = {k: v for k, v in os.environ.items() if not k.startswith("RESEARCHFORGE_AUTO")}
        proc = subprocess.run(
            [sys.executable, "-m", "researchforge", *cmd],
            cwd=project["root"],
            env=env,
            capture_output=True,
            text=True,
            timeout=120,
            check=False,
        )
        if proc.returncode != 0:
            raise ValueError((proc.stdout + proc.stderr).strip()[-300:] or f"{decision} failed")
        control.record_plan_feedback(repo_dir, plan_id, decision, comment)
        return {"plan": plan_id, "decision": decision, "output": proc.stdout.strip()[-300:]}

    def action_repo_auto(self, project: dict[str, Any], body: dict[str, Any]) -> dict[str, Any]:
        """Unattended auto-run of a repo project: settings, approval, launches, service.

        A standing approval needs the typed word "approve", like Approve plan.
        """
        from researchforge.autorun import unattended as ua
        from researchforge.kaggle import control

        if project.get("overnight") or not project.get("repo"):
            raise ValueError("repo auto-run is for repository projects; competitions use auto-run")
        root = Path(project["root"])
        op = str(body.get("op") or "")
        if op == "settings":
            changes = body.get("changes") or {}
            if not isinstance(changes, dict):
                raise ValueError("changes must be an object")
            return {"config": asdict(ua.update_settings(root, changes))}
        if op == "approval":
            cfg = ua.set_approval(
                root, str(body.get("mode") or ""), typed=str(body.get("typed") or "")
            )
            return {"config": asdict(cfg)}
        if op == "start":
            status, detail = ua.start_if_ready(root)
            if status != "ok" or "started" not in detail:
                raise ValueError(f"not starting: {detail}")
            return {"output": detail}
        if op == "stop":
            stopped = ua.stop_launch(root)
            if stopped is None:
                raise ValueError("no launch is running")
            return {"stopped": stopped}
        if op == "pause":
            control.pause(ua.rf_dir(root), str(body.get("text") or "paused from the dashboard"))
            return {"output": "paused"}
        if op == "resume":
            control.resume(ua.rf_dir(root))
            return {"output": "resumed"}
        if op == "service":
            if bool(body.get("enabled")):
                return {"output": ua.install_service(root)}
            return {"output": ua.uninstall_service(root)}
        if op == "monitor":  # papers can take minutes: run it beside the request
            threading.Thread(target=ua.monitor_cycle, args=(root,), daemon=True).start()
            return {"output": "monitor cycle started"}
        raise ValueError(
            "op must be settings, approval, start, stop, pause, resume, service or monitor"
        )

    def action_round(self, project: dict[str, Any], body: dict[str, Any]) -> dict[str, Any]:
        """Archive, restore or delete one round of the overnight loop."""
        from researchforge.kaggle.experiment_loop import delete_round, set_archived

        art = Path(project["overnight"]["artifacts_dir"])
        try:
            num = int(str(body.get("round", "")).lstrip("Rr"))
        except ValueError:
            raise ValueError("round must be a round number, e.g. 7 or R7") from None
        op = str(body.get("op") or "archive")
        if op == "delete":
            if not delete_round(art, num):
                raise KeyError(f"no round {num}")
        elif op in ("archive", "restore"):
            set_archived(art, num, op == "archive", str(body.get("reason") or "dashboard"))
        else:
            raise ValueError("op must be archive, restore or delete")
        return {"round": num, "op": op}

    _colab_signin: Any = None

    def action_colab(self, project: dict[str, Any], body: dict[str, Any]) -> dict[str, Any]:
        """Install Google's Colab CLI, or sign in to it, from the dashboard."""
        from researchforge.kaggle import colab

        op = str(body.get("op") or "")
        colab.forget_cache()
        if op == "install":
            if colab.find_cli():
                return {"installed": True, "detail": "already installed"}
            ok, last = colab.install()
            if not ok:
                raise ValueError(f"install failed: {last}")
            return {"installed": True, "detail": last}
        if op == "signin":
            if self._colab_signin is None:
                self._colab_signin = colab.SignIn()
            url = self._colab_signin.start()
            return {"url": url, "signed_in": not url}
        if op == "code":
            if self._colab_signin is None:
                raise ValueError("start the sign-in first")
            st = self._colab_signin.finish(str(body.get("code") or ""))
            return {"signed_in": bool(st["signed_in"]), "account": st["account"]}
        if op == "cancel":
            if self._colab_signin is not None:
                self._colab_signin.cancel()
            return {"cancelled": True}
        raise ValueError("op must be install, signin, code or cancel")

    def action_driver(self, project: dict[str, Any], body: dict[str, Any]) -> dict[str, Any]:
        """Switch the project's driver; the next launch and sensor call use it."""
        from researchforge.kaggle import auto_run as ar

        art = Path(project["overnight"]["artifacts_dir"])
        model = body.get("model")
        route = body.get("route_sensors")
        cfg = ar.set_driver(
            art,
            str(body.get("driver") or ar.AutoRunConfig.load(art).driver),
            model=None if model is None else str(model),
            route_sensors=None if route is None else bool(route),
        )
        return {"driver": asdict(cfg)}

    def action_autorun(self, project: dict[str, Any], body: dict[str, Any]) -> dict[str, Any]:
        """Save auto-run limits (rounds per launch, daily limit)."""
        from researchforge.kaggle import auto_run as ar

        art = Path(project["overnight"]["artifacts_dir"])
        cfg = ar.AutoRunConfig.load(art)
        if body.get("rounds_per_launch") is not None:
            cfg.rounds_per_launch = int(body["rounds_per_launch"])
        if body.get("max_rounds_per_day") is not None:
            cfg.max_rounds_per_day = int(body["max_rounds_per_day"])
        cfg.save(art)
        return {"auto_run": asdict(ar.AutoRunConfig.load(art))}

    def action_autorun_stop(self, project: dict[str, Any]) -> dict[str, Any]:
        from researchforge.kaggle import auto_run as ar

        stopped = ar.stop_launch(Path(project["overnight"]["artifacts_dir"]))
        if stopped is None:
            raise ValueError("no launch is running")
        return {"stopped": stopped}

    def action_autorun_start(self, project: dict[str, Any]) -> dict[str, Any]:
        """Start one launch now, through the same gates as the sensor."""
        from researchforge.kaggle import auto_run as ar

        art = Path(project["overnight"]["artifacts_dir"])
        cfg = ar.AutoRunConfig.load(art)
        if ar.current_launch(art):
            raise ValueError("a launch is already running")
        blockers = ar.gates(art, cfg)
        ready = ar.readiness(cfg)
        if blockers or not ready.ready:
            raise ValueError(
                "not starting: " + "; ".join(blockers or [f"{ready.detail} — {ready.fix}"])
            )
        return {"launch": ar.launch(art, project["slug"], cfg)}

    def action_sensor_add(
        self, project: dict[str, Any], name: str, command: str, timeout: int
    ) -> dict[str, Any]:
        art = Path(project["overnight"]["artifacts_dir"])
        _sensors.add_custom(art, project["slug"], name, command, timeout)
        return {"sensor": name}

    def action_sensor_remove(self, project: dict[str, Any], name: str) -> dict[str, Any]:
        art = Path(project["overnight"]["artifacts_dir"])
        if not _sensors.remove_custom(art, project["slug"], name):
            raise KeyError(f"no custom sensor named {name}")
        return {"sensor": name}

    def action_submissions(self, project: dict[str, Any]) -> dict[str, Any]:
        self._fetch_subs(project["slug"])
        sub = self._subs.get(project["slug"], {})
        if sub.get("error"):
            raise RuntimeError(sub["error"])
        return {"count": len(sub.get("submissions", []))}

    # ── steering and loop control ────────────────────────────────────────
    def action_steer(self, project: dict[str, Any], body: dict[str, Any]) -> dict[str, Any]:
        """Add, answer or remove a direction for the proposer."""
        from researchforge.kaggle import steering

        art = Path(project["overnight"]["artifacts_dir"])
        op = str(body.get("op") or "add")
        if op == "add":
            d = steering.add(art, str(body.get("text") or ""), str(body.get("kind") or "next"))
        elif op == "reply":
            d = steering.reply(
                art, str(body.get("id") or ""), str(body.get("text") or ""), "person"
            )
        elif op == "remove":
            d = steering.remove(art, str(body.get("id") or ""))
        else:
            raise ValueError("op must be add, reply or remove")
        return {"direction": asdict(d)}

    def action_loop(self, project: dict[str, Any], body: dict[str, Any]) -> dict[str, Any]:
        """Pause, resume, re-synthesize, or add an idea — the chat's steering commands."""
        op, slug = str(body.get("op") or ""), project["slug"]
        text = str(body.get("text") or "").strip()
        if op == "pause":
            args = ["pause", slug, "--reason", text or "paused from the dashboard"]
        elif op == "resume":
            args = ["resume", slug]
        elif op == "add-idea":
            if not text:
                raise ValueError("write the idea first")
            args = ["add-hypothesis", slug, text]
        elif op == "resynthesize":
            args = ["resynthesize", slug]
        else:
            raise ValueError("op must be pause, resume, add-idea or resynthesize")
        res = self._cli(project, args, timeout=900 if op == "resynthesize" else 60)
        out = (res.stdout + res.stderr).strip()
        if res.returncode != 0:
            raise ValueError(out.splitlines()[-1] if out else f"exit {res.returncode}")
        return {"output": out.splitlines()[-1] if out else ""}

    # ── keys & accounts (global) ─────────────────────────────────────────
    def action_keys(self, path: str, body: dict[str, Any]) -> dict[str, Any]:
        name = str(body.get("name") or "")
        if path == "/api/keys/set":
            where = _keys.set_key(
                name, str(body.get("value") or ""), str(body.get("username") or "")
            )
            test = _keys.test_key(name)
            return {"where": where, "test": test}
        if path == "/api/keys/test":
            return {"test": _keys.test_key(name)}
        if path == "/api/keys/remove":
            return {"output": _keys.remove_key(name)}
        if path == "/api/keys/colab":
            return self.action_colab({}, body)
        raise KeyError(f"unknown endpoint {path}")

    # ── new competition (global) ─────────────────────────────────────────
    def _draft_dir(self, body: dict[str, Any]) -> Path:
        """A draft's artifacts dir from the request, checked: a .overnight_* dir with a draft."""
        art = Path(str(body.get("artifacts_dir") or "")).expanduser()
        if not art.name.startswith(OVERNIGHT_PREFIX) or _new.load_draft(art) is None:
            raise ValueError("no setup draft there")
        _new.check_root(str(art.parent))
        return art

    def action_new(self, path: str, body: dict[str, Any]) -> dict[str, Any]:
        if path == "/api/new/facts":
            slug = _new.parse_slug(str(body.get("competition") or ""))
            done = _new.existing_setup(slug)
            if done is not None:
                raise ValueError(f"{slug} is already set up at {done.parent} — open it there")
            # Each competition in its own folder unless the person picks one.
            root = _new.check_root(str(body.get("root") or "").strip() or str(self.cwd / slug))
            art = _new.artifacts_dir_for(root, slug)
            _new.save_draft(
                art, {"slug": slug, "root": str(root), "via": "dashboard", "status": "facts"}
            )
            register_overnight_dir(slug, art)  # so the dashboard (and a restart) finds the draft
            return self._start_job("facts", f"Gather facts: {slug}", art, [["gather-facts", slug]])
        if path == "/api/new/cloud-gpus":
            from researchforge.kaggle.cloud_gpu_config import CloudGPUConfig, list_gpu_types

            provider = str(body.get("provider") or "")
            if provider not in _new.CLOUD_PROVIDERS:
                raise ValueError(f"provider must be one of {', '.join(_new.CLOUD_PROVIDERS)}")
            key = CloudGPUConfig(provider=provider).api_key()
            if not key:
                raise ValueError(f"set the {provider} API key in Keys & accounts first")
            return {"gpus": list_gpu_types(provider, key)[:60]}
        art = self._draft_dir(body)
        draft = _new.load_draft(art) or {}
        if path == "/api/new/draft":
            form = body.get("form")
            if not isinstance(form, dict):
                raise ValueError("form must be an object")
            return {"draft": _new.save_draft(art, {"form": form, "status": "answering"})}
        if path == "/api/new/discard":
            (art / _new.DRAFT_FILE).unlink(missing_ok=True)
            return {"discarded": str(art)}
        if path == "/api/new/setup":
            form = body.get("form") if isinstance(body.get("form"), dict) else draft.get("form")
            if not isinstance(form, dict) or not draft.get("facts"):
                raise ValueError("gather the facts first")
            missing = _new.missing_keys(form)
            if missing:
                raise ValueError(
                    "set these in Keys & accounts first: "
                    + ", ".join(_keys.SLOTS[m].label for m in missing)
                )
            cmds = _new.setup_commands(draft["slug"], form, draft["facts"])
            _new.save_draft(art, {"form": form, "status": "setting_up"})
            steps: list[Any] = [*cmds]
            if str(form.get("ntfy_topic") or "").strip():
                steps.append(("Set the ntfy topic", lambda: self._set_ntfy(art, form)))
            return self._start_job("setup", f"Set up {draft['slug']}", art, steps)
        raise KeyError(f"unknown endpoint {path}")

    @staticmethod
    def _set_ntfy(art: Path, form: dict[str, Any]) -> str:
        from researchforge.kaggle.run_config import set_field

        set_field(art, "ntfy_topic", str(form["ntfy_topic"]).strip(), via="dashboard")
        return "ntfy topic saved"

    def _start_job(self, kind: str, title: str, art: Path, steps: list[Any]) -> dict[str, Any]:
        """Run setup commands one after another in the background; the payload shows progress.

        A step is a ``researchforge kaggle`` argv list, or ``(label, callable)``. The job is
        mirrored into the draft, so another surface (or a restarted dashboard) sees it.
        """
        if any(
            j["status"] == "running" and j["artifacts_dir"] == str(art) for j in self._jobs.values()
        ):
            raise ValueError("a setup step is already running for this competition")
        job: dict[str, Any] = {
            "id": f"job-{secrets.token_hex(4)}",
            "kind": kind,
            "title": title,
            "artifacts_dir": str(art),
            "status": "running",
            "started_at": datetime.now().isoformat(timespec="seconds"),
            "finished_at": None,
            "error": "",
            "steps": [
                {
                    "label": s[0]
                    if isinstance(s, tuple)
                    else "researchforge kaggle "
                    + " ".join(a if len(a) < 80 else a[:77] + "…" for a in s),
                    "status": "pending",
                    "output": "",
                }
                for s in steps
            ],
        }
        self._jobs[job["id"]] = job

        def changed(persist: bool = True) -> None:
            self._jobs_version += 1
            with self._lock:
                self._state = None
            if persist:
                with contextlib.suppress(OSError):
                    _new.save_draft(art, {"job": job})

        def run() -> None:
            from researchforge.kaggle.auto_run import HOST_MARKERS

            # Setup from the dashboard runs inside no agent, whatever started the dashboard.
            env = {k: v for k, v in os.environ.items() if k not in HOST_MARKERS}
            env |= {"RESEARCHFORGE_OVERNIGHT_DIR": str(art), "RESEARCHFORGE_NO_DASHBOARD": "1"}
            for step, rec in zip(steps, job["steps"], strict=True):
                rec["status"] = "running"
                changed()
                try:
                    if isinstance(step, tuple):
                        rec["output"], rc = str(step[1]()), 0
                    else:
                        rc = self._stream(step, art.parent, env, rec, changed)
                except Exception as exc:
                    rec["output"], rc = f"{type(exc).__name__}: {exc}", 1
                rec["status"] = "ok" if rc == 0 else "error"
                if rc != 0:
                    lines = [ln for ln in rec["output"].splitlines() if ln.strip()]
                    job["error"] = lines[-1] if lines else f"exit {rc}"
                    break
            job["status"] = "error" if job["error"] else "done"
            job["finished_at"] = datetime.now().isoformat(timespec="seconds")
            keep = True
            with contextlib.suppress(Exception):
                keep = self._job_finished(job, art)
            changed(persist=keep)  # a finished setup removed the draft: do not write it back

        threading.Thread(target=run, name=f"rf-{job['id']}", daemon=True).start()
        return {"job": job["id"]}

    @staticmethod
    def _stream(
        argv: list[str], cwd: Path, env: dict[str, str], rec: dict[str, Any], changed: Any
    ) -> int:
        cwd.mkdir(parents=True, exist_ok=True)
        proc = subprocess.Popen(
            [sys.executable, "-m", "researchforge", "kaggle", *argv],
            cwd=str(cwd),
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        last = 0.0
        assert proc.stdout is not None
        for line in proc.stdout:
            rec["output"] = (rec["output"] + line)[-6000:]
            if time.time() - last > 2:
                last = time.time()
                changed(persist=False)
        return proc.wait()

    def _job_finished(self, job: dict[str, Any], art: Path) -> bool:
        """Update the draft for a finished job. Returns False once the draft is gone."""
        if job["status"] != "done":
            _new.save_draft(art, {"status": "facts" if job["kind"] == "facts" else "answering"})
            return True
        if job["kind"] == "facts":
            facts = read_json(art / "facts.json", None)
            if not isinstance(facts, dict):
                raise ValueError("gather-facts wrote no facts.json")
            draft = _new.load_draft(art) or {}
            form = draft.get("form") or _new.default_form(facts)
            _new.save_draft(art, {"facts": facts, "form": form, "status": "answering"})
            return True
        # Set up: the draft has done its job; the project now shows like any other.
        (art / _new.DRAFT_FILE).unlink(missing_ok=True)
        return False


# ---------------------------------------------------------------------------
# Daemon helpers
# ---------------------------------------------------------------------------


def _write_pid(port: int) -> None:
    _pid_file().parent.mkdir(parents=True, exist_ok=True)
    _pid_file().write_text(json.dumps({"pid": os.getpid(), "port": port}))


def _read_pid() -> dict[str, Any] | None:
    try:
        data: dict[str, Any] = json.loads(_pid_file().read_text())
        return data
    except (OSError, ValueError):
        return None


def _health(port: int, timeout: float = 1.0) -> dict[str, Any] | None:
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/api/health", timeout=timeout) as r:
            data = json.loads(r.read())
            return data if data.get("service") == "researchforge-dashboard" else None
    except (urllib.error.URLError, OSError, ValueError):
        return None


def is_running(port: int | None = None) -> tuple[bool, int]:
    """Return (running, port). A live PID file *or* a healthy dashboard on the port counts."""
    info = _read_pid()
    if info:
        pid, p = info.get("pid", 0), info.get("port", _DEFAULT_PORT)
        try:
            os.kill(pid, 0)
            return True, p
        except (ProcessLookupError, PermissionError, TypeError):
            _pid_file().unlink(missing_ok=True)
    p = port or _DEFAULT_PORT
    return (_health(p) is not None), p


def start_daemon(port: int = _DEFAULT_PORT, cwd: Path | None = None, wait: float = 6.0) -> int:
    """Start the dashboard in the background and wait until it answers. Returns the port."""
    running, existing = is_running(port)
    if running:
        return existing
    cwd = cwd or Path.cwd()
    log = config_dir() / "dashboard.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("ab") as fh:
        subprocess.Popen(
            [
                sys.executable,
                "-m",
                "researchforge",
                "kaggle",
                "dashboard",
                "--port",
                str(port),
                "--no-open",
                "--daemon-internal",
            ],
            start_new_session=True,
            stdout=fh,
            stderr=fh,
            cwd=str(cwd),
        )
    deadline = time.time() + wait
    while time.time() < deadline:
        if _health(port, timeout=0.5):
            return port
        time.sleep(0.25)
    raise RuntimeError(f"dashboard did not start on port {port} — see {log}")


def stop_daemon() -> bool:
    """Stop the background dashboard. Returns True if a process was signalled."""
    info = _read_pid()
    if not info:
        h = _health(_DEFAULT_PORT)
        if h and h.get("pid"):
            info = {"pid": h["pid"]}
        else:
            return False
    try:
        os.kill(int(info["pid"]), 15)
        return True
    except (ProcessLookupError, PermissionError, ValueError, KeyError):
        return False
    finally:
        _pid_file().unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# HTTP server
# ---------------------------------------------------------------------------


def make_handler(service: DashboardService, port: int) -> Any:
    import http.server
    import urllib.parse

    allowed_hosts = {
        f"localhost:{port}",
        f"127.0.0.1:{port}",
        f"[::1]:{port}",
        "localhost",
        "127.0.0.1",
    }

    def page() -> bytes:
        # Read per request: the shell is small, and an upgraded install serves
        # the new UI without restarting a long-lived daemon.
        return (
            _APP_HTML.read_text(encoding="utf-8")
            .replace("__RF_TOKEN__", service.token)
            .encode("utf-8")
        )

    class Handler(http.server.BaseHTTPRequestHandler):
        server_version = "ResearchForgeDashboard/1"

        def log_message(self, fmt: Any, *args: Any) -> None:  # quiet
            pass

        def _send(
            self, status: int, body: bytes, ctype: str, headers: dict[str, Any] | None = None
        ) -> None:
            self.send_response(status)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            for k, v in (headers or {}).items():
                self.send_header(k, v)
            self.end_headers()
            if self.command != "HEAD":
                self.wfile.write(body)

        def _json(
            self, data: dict[str, Any], status: int = 200, headers: dict[str, Any] | None = None
        ) -> None:
            self._send(status, json.dumps(data, default=str).encode(), "application/json", headers)

        def _host_ok(self) -> bool:
            return self.headers.get("Host", "") in allowed_hosts

        def do_GET(self) -> None:  # noqa: N802
            if not self._host_ok():
                return self._json({"ok": False, "error": "bad host"}, 403)
            path = urllib.parse.urlparse(self.path).path
            if path in ("/", "/index.html"):
                return self._send(
                    200,
                    page(),
                    "text/html; charset=utf-8",
                    {
                        "Content-Security-Policy": (
                            "default-src 'self'; script-src 'unsafe-inline'; "
                            "style-src 'unsafe-inline'; img-src 'self' data:"
                        )
                    },
                )
            if path == "/api/health":
                return self._json(
                    {
                        "ok": True,
                        "service": "researchforge-dashboard",
                        "pid": os.getpid(),
                        "port": port,
                        "started_at": service.started_at,
                    }
                )
            if path in ("/api/state", "/json"):
                try:
                    payload = service.state()
                except Exception as exc:
                    return self._json({"ok": False, "error": f"state build failed: {exc}"}, 500)
                tag = service.etag(payload)
                if self.headers.get("If-None-Match") == tag:
                    self.send_response(304)
                    self.send_header("ETag", tag)
                    self.end_headers()
                    return
                # Same-origin only (no CORS headers): lets an open tab pick up the
                # token of a restarted server without exposing it to other sites.
                return self._json(payload, headers={"ETag": tag, "X-RF-Token": service.token})
            return self._json({"ok": False, "error": "not found"}, 404)

        do_HEAD = do_GET

        def do_POST(self) -> None:  # noqa: N802
            if not self._host_ok() or self.headers.get("X-RF-Token") != service.token:
                return self._json({"ok": False, "error": "forbidden"}, 403)
            path = urllib.parse.urlparse(self.path).path
            try:
                n = int(self.headers.get("Content-Length") or 0)
                body = json.loads(self.rfile.read(n) or b"{}") if n else {}
            except ValueError:
                return self._json({"ok": False, "error": "invalid JSON body"}, 400)
            if path.startswith(("/api/keys/", "/api/new/")):
                try:
                    result = (
                        service.action_keys(path, body)
                        if path.startswith("/api/keys/")
                        else service.action_new(path, body)
                    )
                except (KeyError, ValueError) as exc:
                    return self._json({"ok": False, "error": str(exc).strip("'")}, 400)
                except Exception as exc:
                    return self._json({"ok": False, "error": str(exc)}, 500)
                service.state(force=True)
                return self._json({"ok": True, **result})
            project = service.find_project(str(body.get("project", "")))
            if project is None:
                return self._json({"ok": False, "error": "unknown project"}, 404)
            if path in ("/api/question/answer", "/api/plan/approve", "/api/repo/auto"):
                try:
                    if path == "/api/question/answer":
                        result = service.action_answer(project, body)
                    elif path == "/api/plan/approve":
                        result = service.action_plan_approve(project, body)
                    else:
                        result = service.action_repo_auto(project, body)
                except (KeyError, ValueError, RuntimeError) as exc:
                    return self._json({"ok": False, "error": str(exc).strip("'")}, 400)
                service.state(force=True)
                return self._json({"ok": True, **result})
            if not project.get("overnight"):
                return self._json(
                    {
                        "ok": False,
                        "error": "automation is only available for Kaggle overnight projects",
                    },
                    400,
                )
            try:
                if path == "/api/scheduler":
                    result = service.action_scheduler(
                        project, bool(body.get("enabled")), float(body.get("interval_hours") or 6)
                    )
                elif path == "/api/monitor/run":
                    result = service.action_monitor(project)
                elif path == "/api/sensor":
                    result = service.action_sensor(
                        project, str(body.get("name", "")), bool(body.get("enabled"))
                    )
                elif path == "/api/sensor/add":
                    result = service.action_sensor_add(
                        project,
                        str(body.get("name", "")),
                        str(body.get("command", "")),
                        int(body.get("timeout") or 120),
                    )
                elif path == "/api/autorun":
                    result = service.action_autorun(project, body)
                elif path == "/api/autorun/start":
                    result = service.action_autorun_start(project)
                elif path == "/api/autorun/stop":
                    result = service.action_autorun_stop(project)
                elif path == "/api/config":
                    from researchforge.kaggle.run_config import set_field

                    before, after = set_field(
                        Path(project["overnight"]["artifacts_dir"]),
                        str(body.get("field", "")),
                        body.get("value"),
                        via="dashboard",
                    )
                    result = {"field": body.get("field"), "from": before, "to": after}
                elif path == "/api/notify-test":
                    from researchforge.kaggle.run_config import send_test_alert

                    result = send_test_alert(Path(project["overnight"]["artifacts_dir"]))
                elif path == "/api/colab":
                    result = service.action_colab(project, body)
                elif path == "/api/round":
                    result = service.action_round(project, body)
                elif path == "/api/driver":
                    result = service.action_driver(project, body)
                elif path == "/api/sensor/remove":
                    result = service.action_sensor_remove(project, str(body.get("name", "")))
                elif path == "/api/submissions/refresh":
                    result = service.action_submissions(project)
                elif path == "/api/steer":
                    result = service.action_steer(project, body)
                elif path == "/api/loop":
                    result = service.action_loop(project, body)
                else:
                    return self._json({"ok": False, "error": f"unknown endpoint {path}"}, 404)
            except (KeyError, ValueError) as exc:
                return self._json({"ok": False, "error": str(exc).strip("'")}, 400)
            except Exception as exc:
                return self._json({"ok": False, "error": str(exc)}, 500)
            service.state(force=True)
            return self._json({"ok": True, **result})

    return Handler


def serve(
    port: int = _DEFAULT_PORT,
    open_browser: bool = False,
    cwd: Path | None = None,
    daemon_internal: bool = False,
) -> None:
    """Serve the dashboard on localhost:<port> until interrupted."""
    import http.server

    cwd = cwd or Path.cwd()
    service = DashboardService(cwd)
    try:
        server = http.server.ThreadingHTTPServer(("127.0.0.1", port), make_handler(service, port))
    except OSError as exc:
        if _health(port):
            print(f"✅ ResearchForge Dashboard already running → http://localhost:{port}")
            return
        raise SystemExit(f"❌ Cannot bind localhost:{port}: {exc}") from exc
    server.daemon_threads = True
    import signal

    def _sigterm(*_a: Any) -> None:
        raise KeyboardInterrupt

    with contextlib.suppress(ValueError):  # not the main thread (tests)
        signal.signal(signal.SIGTERM, _sigterm)
    _write_pid(port)
    service.start_background()
    threading.Thread(target=service.state, daemon=True).start()  # warm the cache

    url = f"http://localhost:{port}"
    if not daemon_internal:
        print(f"✅ ResearchForge Dashboard → {url}")
        print("   Press Ctrl+C to stop.")
    if open_browser:
        import webbrowser

        threading.Timer(0.5, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n   Dashboard stopped.")
    finally:
        service._stop.set()
        info = _read_pid()
        if info and info.get("pid") == os.getpid():
            _pid_file().unlink(missing_ok=True)
