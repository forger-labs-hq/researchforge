"""A small, stable API over projects, 🔔 questions and steering, for plugins.

Plugins call these instead of reaching into dashboard or loop internals, so the
internals can keep changing. The live dashboard uses the same functions, so a
plugin answering a question goes through exactly the code path the bell does.

Projects are the dashboard's project payloads (dicts from
:func:`researchforge.kaggle.dashboard_state.build_state`); pass one back to the
functions that act on it.
"""

from __future__ import annotations

import os
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any, Literal

Where = Literal["kaggle", "repo"]


def list_projects(cwd: Path | None = None) -> list[dict[str, Any]]:
    """Every project the dashboard knows about (registry, cwd, overnight copies)."""
    from researchforge.kaggle.dashboard_state import build_state

    projects: list[dict[str, Any]] = build_state(cwd or Path.cwd())["projects"]
    return projects


def find_project(project_id: str, cwd: Path | None = None) -> dict[str, Any] | None:
    """A project by its dashboard id, slug or name (id wins)."""
    projects = list_projects(cwd)
    for key in ("id", "slug", "name"):
        for p in projects:
            if p.get(key) == project_id:
                return p
    return None


def project_summary(project: dict[str, Any]) -> dict[str, Any]:
    """The few fields a status line needs."""
    return {
        "id": project["id"],
        "name": project.get("name"),
        "slug": project.get("slug"),
        "category": project.get("category"),
        "state": project.get("state"),
        "role": project.get("role"),
        "location": project.get("location"),
        "best": project.get("best"),
        "our_best_lb": project.get("our_best_lb"),
        "last_activity_at": project.get("last_activity_at"),
        "rounds": len(project.get("experiments") or []),
        "open_questions": len(project.get("questions") or []),
    }


def _artifacts(project: dict[str, Any], where: Where) -> Path:
    if where == "kaggle":
        if not project.get("overnight"):
            raise ValueError("this project has no overnight loop")
        return Path(project["overnight"]["artifacts_dir"])
    return Path(project["root"]) / ".researchforge"


def _places(project: dict[str, Any]) -> list[Where]:
    out: list[Where] = []
    if project.get("overnight"):
        out.append("kaggle")
    if project.get("repo"):
        out.append("repo")
    return out


def is_approval(kind: str, answer_key: str) -> bool:
    """Questions only a person may answer: submissions and the loop's judge decisions."""
    return kind == "submission_approval" or answer_key.startswith("judge:")


def open_questions(project: dict[str, Any]) -> list[dict[str, Any]]:
    """The project's open 🔔 questions, each with where it lives and whether it is an approval."""
    from researchforge.kaggle import control

    out: list[dict[str, Any]] = []
    for where in _places(project):
        for item in control.open_interactions(_artifacts(project, where)):
            row = asdict(item)
            row["where"] = where
            row["approval"] = is_approval(item.kind, item.answer_key)
            out.append(row)
    return out


def answer_question(
    project: dict[str, Any],
    question_id: str,
    answer: str = "",
    comment: str = "",
    where: Where = "kaggle",
    dismiss: bool = False,
) -> dict[str, Any]:
    """Answer (or dismiss) one open question — the same path as the dashboard's bell."""
    from researchforge.kaggle import control

    answer = answer.strip()
    if not question_id or not (answer or dismiss):
        raise ValueError("id and answer are required")
    d = _artifacts(project, where)
    item = next((q for q in control.open_interactions(d) if q.id == question_id), None)
    if item is None:
        raise KeyError(f"no open question {question_id}")
    if dismiss:
        control.dismiss(d, question_id, "dismissed on the dashboard", comment)
        return {"dismissed": question_id}
    if item.answer_key.startswith("judge:"):
        from researchforge.autorun.kaggle_mode import record_decision

        if answer not in ("yes", "no", "skip"):
            raise ValueError("answer must be yes, no or skip")
        record_decision(d, item.answer_key[6:], answer)  # type: ignore[arg-type]
        if comment:
            items = control.load_interactions(d)
            for i in items:
                if i.id == question_id:
                    i.comment = comment.strip()
            control._save_interactions(d, items)
    elif item.kind == "submission_approval" and where == "kaggle":
        from researchforge.kaggle.submission import answer as answer_submission

        line = answer_submission(d, str(project.get("slug") or ""), question_id, answer, comment)
        return {"answered": question_id, "answer": answer, "output": line}
    else:
        applied = control.respond(d, question_id, answer, comment).applied
        return {"answered": question_id, "answer": answer, "applied": applied}
    return {"answered": question_id, "answer": answer}


def decide_plan(
    project: dict[str, Any], plan_id: str, decision: str, typed: str = "", comment: str = ""
) -> dict[str, Any]:
    """Approve or reject a repo plan for a person, with their comment.

    Approving needs the typed word "approve" — the same as `researchforge
    experiment approve`. Refused for a competition, whose only engine is its
    Kaggle loop. Rejecting cancels the plan.
    """
    from researchforge.kaggle import control
    from researchforge.kaggle.dashboard_state import _plans_awaiting_approval

    if project.get("overnight"):
        raise ValueError(
            "this competition runs only on its Kaggle loop; repo plans are not run for it"
        )
    repo_dir = Path(project["root"]) / ".researchforge"
    if plan_id not in _plans_awaiting_approval(repo_dir):
        raise KeyError(f"{plan_id} is not waiting for approval")
    if decision == "approve":
        if typed.strip().lower() != "approve":
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


def plans_awaiting_approval(project: dict[str, Any]) -> list[str]:
    """Repo plans waiting for a person's Approve plan."""
    from researchforge.kaggle.dashboard_state import _plans_awaiting_approval

    if project.get("overnight") or not project.get("repo"):
        return []
    return list(_plans_awaiting_approval(Path(project["root"]) / ".researchforge"))


def _steering_dir(project: dict[str, Any]) -> Path:
    if not project.get("overnight"):
        raise ValueError("steering directions are for competitions with an overnight loop")
    return Path(project["overnight"]["artifacts_dir"])


def directions(project: dict[str, Any]) -> list[dict[str, Any]]:
    """The Up next queue: the person's directions and pinned ideas, in run order."""
    from researchforge.kaggle import steering

    return steering.up_next(_steering_dir(project))


def add_direction(
    project: dict[str, Any], text: str, kind: str = "next", rounds: int = 1, via: str = "plugin"
) -> dict[str, Any]:
    """Put a direction at the top of Up next (the newest one is followed first)."""
    from researchforge.kaggle import steering

    return asdict(steering.add(_steering_dir(project), text, kind, via=via, rounds=rounds))


def reorder_directions(project: dict[str, Any], ids: list[str]) -> list[dict[str, Any]]:
    """Put these directions first, in this order."""
    from researchforge.kaggle import steering

    art = _steering_dir(project)
    steering.reorder(art, ids)
    return steering.up_next(art)
