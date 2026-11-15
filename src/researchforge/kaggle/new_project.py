"""A new competition from the dashboard: the same facts, questions and setup as /overnight.

The wizard runs the commands the overnight skill runs — ``gather-facts``, then
``kaggle overnight … --answers-json``, then the sensors, driver, auto-run and scheduler
— so a project set up here is the same as one set up from a chat. What has been
answered so far is kept in ``setup_draft.json``: a setup started on the dashboard can
be finished in Claude Code or Cursor (the skill's Phase 0 reads the draft), and one
started in a chat can be finished here.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from researchforge.kaggle.workspace import OVERNIGHT_PREFIX, atomic_write_json, read_json

DRAFT_FILE = "setup_draft.json"
COMPUTES = ("local", "kaggle", "colab", "ssh", "cloud")
CLOUD_PROVIDERS = ("runpod", "lambda", "vast")  # AWS needs more than a form: use a chat
_FACT_COMPUTE = {"kaggle_gpu": "kaggle", "cpu": "local", "colab_gpu": "colab"}


def parse_slug(text: str) -> str:
    from researchforge.kaggle.orchestrator import OvernightOrchestrator

    slug = OvernightOrchestrator._parse_slug(text.strip())
    if not slug or "/" in slug or " " in slug:
        raise ValueError("give a Kaggle competition URL or slug")
    return slug


def artifacts_dir_for(root: Path, slug: str) -> Path:
    return root / f"{OVERNIGHT_PREFIX}{slug}"


def existing_setup(slug: str) -> Path | None:
    """The set-up project for this slug on this machine, if there is one."""
    from researchforge.kaggle.workspace import has_setup, load_registry

    for entry in load_registry():
        d = Path(str(entry.get("artifacts_dir", "")))
        if entry.get("slug") == slug and has_setup(d):
            return d
    return None


def check_root(root: str) -> Path:
    path = Path(root).expanduser()
    if not path.is_absolute():
        raise ValueError("the folder must be an absolute path")
    home = Path.home().resolve()
    resolved = path.resolve()
    if resolved != home and home not in resolved.parents:
        raise ValueError(f"pick a folder inside {home}")
    if resolved.exists() and not resolved.is_dir():
        raise ValueError(f"{resolved} is a file, not a folder")
    return resolved


# ---------------------------------------------------------------------------
# The draft
# ---------------------------------------------------------------------------


def load_draft(artifacts_dir: Path) -> dict[str, Any] | None:
    raw = read_json(artifacts_dir / DRAFT_FILE, None)
    return raw if isinstance(raw, dict) else None


def save_draft(artifacts_dir: Path, patch: dict[str, Any]) -> dict[str, Any]:
    draft = {**(load_draft(artifacts_dir) or {}), **patch}
    draft["updated_at"] = datetime.now().isoformat(timespec="seconds")
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_json(artifacts_dir / DRAFT_FILE, draft)
    return draft


def default_form(facts: dict[str, Any]) -> dict[str, Any]:
    """The questions with the answers /overnight would suggest."""
    from researchforge.kaggle.sensors import BUILTIN_SENSORS, DEFAULT_OFF

    rec = str(facts.get("recommended_compute") or "")
    compute = _FACT_COMPUTE.get(rec, rec if rec in COMPUTES else "local")
    if facts.get("is_code_competition"):
        compute = "kaggle"
    return {
        "compute": compute,
        "colab": {"plan": "free", "gpu": "T4", "high_mem": False},
        "ssh": {"host": "", "key": ""},
        "cloud": {"provider": "runpod", "gpu_type": "", "max_price": ""},
        "time_hours": 8,
        "max_experiments": 20,
        "sensors": {s["name"]: s["name"] not in DEFAULT_OFF for s in BUILTIN_SENSORS},
        "auto_run": {
            "driver": str(facts.get("auto_run_driver") or "claude"),
            "rounds": 2,
            "daily_max": 6,
        },
        "scheduler": {"on": True, "interval_hours": 6},
        "answers": {"known_approaches": "", "forbidden_features": "", "external_data": "unknown"},
        "metric_direction": str(facts.get("metric_direction") or ""),
        "ntfy_topic": "",
    }


# ---------------------------------------------------------------------------
# What setup runs
# ---------------------------------------------------------------------------


def _int(value: Any, low: int, high: int, what: str) -> int:
    try:
        n = int(value)
    except (TypeError, ValueError):
        raise ValueError(f"{what} must be a whole number") from None
    if not low <= n <= high:
        raise ValueError(f"{what} must be between {low} and {high}")
    return n


def setup_commands(slug: str, form: dict[str, Any], facts: dict[str, Any]) -> list[list[str]]:
    """The ``researchforge kaggle …`` argv lists setup runs, in order. ValueError if invalid."""
    from researchforge.kaggle.auto_run import DRIVERS
    from researchforge.kaggle.sensors import BUILTIN_NAMES

    compute = str(form.get("compute") or "")
    if compute not in COMPUTES:
        raise ValueError(f"compute must be one of {', '.join(COMPUTES)}")
    if facts.get("is_code_competition") and compute != "kaggle":
        raise ValueError("a code competition only accepts Kaggle notebooks: pick Kaggle GPU")
    cmds: list[list[str]] = []
    if compute == "colab":
        c = form.get("colab") or {}
        if not c.get("plan") or not c.get("gpu"):
            raise ValueError("pick your Colab plan and GPU")
        cmds.append(
            ["colab-setup", slug, "--plan", str(c["plan"]), "--gpu", str(c["gpu"])]
            + (["--high-mem"] if c.get("high_mem") else ["--no-high-mem"])
        )
    elif compute == "ssh":
        s = form.get("ssh") or {}
        host = str(s.get("host") or "").strip()
        if "@" not in host:
            raise ValueError("give the SSH host as user@server")
        cmds.append(
            ["ssh-setup", slug, "--host", host]
            + (["--key", str(s["key"]).strip()] if str(s.get("key") or "").strip() else [])
        )
    elif compute == "cloud":
        c = form.get("cloud") or {}
        provider = str(c.get("provider") or "")
        if provider not in CLOUD_PROVIDERS:
            raise ValueError(
                f"cloud provider must be one of {', '.join(CLOUD_PROVIDERS)} "
                "(AWS needs a key pair, image and security group: set it up from a chat)"
            )
        if not str(c.get("gpu_type") or "").strip():
            raise ValueError("pick a GPU type (List GPUs shows them with prices)")
        cmd = ["cloud-gpu-setup", slug, "--provider", provider, "--gpu-type", str(c["gpu_type"])]
        if str(c.get("max_price") or "").strip():
            cmd += ["--max-price", str(float(c["max_price"]))]
        cmds.append(cmd)

    answers = {k: v for k, v in (form.get("answers") or {}).items() if str(v).strip()}
    cmds.append(
        [
            "overnight", slug,
            "--gpu", compute,
            "--time-hours", str(_int(form.get("time_hours"), 1, 168, "time budget")),
            "--max-experiments", str(_int(form.get("max_experiments"), 1, 1000, "max experiments")),
            "--answers-json", json.dumps(answers),
        ]
    )  # fmt: skip
    direction = str(form.get("metric_direction") or "")
    if direction and direction != str(facts.get("metric_direction") or ""):
        if direction not in ("maximize", "minimize"):
            raise ValueError("metric direction must be maximize or minimize")
        cmds.append(["settings", slug, "--metric-direction", direction])

    sensors = form.get("sensors") or {}
    off = sorted(n for n, on in sensors.items() if n in BUILTIN_NAMES and not on)
    on_ = sorted(n for n, on in sensors.items() if n in BUILTIN_NAMES and on)
    toggles = [a for n in off for a in ("--disable", n)] + [
        a for n in on_ if n != "auto_run_sensor" for a in ("--enable", n)
    ]
    if toggles:
        cmds.append(["sensors", slug, *toggles])

    ar = form.get("auto_run") or {}
    driver = str(ar.get("driver") or "claude")
    if driver not in DRIVERS:
        raise ValueError(f"driver must be one of {', '.join(DRIVERS)}")
    cmds.append(["driver", slug, driver])
    if sensors.get("auto_run_sensor"):
        cmds.append(
            [
                "auto-run", slug,
                "--driver", driver,
                "--rounds", str(_int(ar.get("rounds"), 1, 20, "rounds per launch")),
                "--daily-max", str(_int(ar.get("daily_max"), 1, 100, "rounds per day")),
                "--on",
            ]
        )  # fmt: skip
    sch = form.get("scheduler") or {}
    if sch.get("on"):
        hours = float(sch.get("interval_hours") or 6)
        if not 0.25 <= hours <= 48:
            raise ValueError("the scheduler interval must be between 0.25 and 48 hours")
        cmds.append(["scheduler-install", slug, "--interval-hours", str(hours)])
    return cmds


def missing_keys(form: dict[str, Any]) -> list[str]:
    """Keys the chosen setup needs that are not set yet (names, for the Keys page)."""
    from researchforge.kaggle import keys

    need = ["kaggle"]
    compute = form.get("compute")
    if compute == "colab":
        need.append("colab")
    if compute == "cloud":
        need.append(str((form.get("cloud") or {}).get("provider") or "runpod"))
    if (form.get("sensors") or {}).get("auto_run_sensor"):
        from researchforge.kaggle.auto_run import AutoRunConfig, readiness

        driver = str((form.get("auto_run") or {}).get("driver") or "claude")
        # A signed-in CLI is as good as a stored token; readiness checks both.
        if not readiness(AutoRunConfig(driver=driver)).ready:
            return [k for k in need if not keys.ready(k)] + [
                "anthropic" if driver == "byok" else driver
            ]
    return [k for k in need if not keys.ready(k)]
