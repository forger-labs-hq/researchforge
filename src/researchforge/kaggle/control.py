"""Human control of a running overnight loop — without stopping it.

Two files in ``.overnight_<slug>/``:

``loop_control.json``
    What the human has decided: paused or not, notification level, submission
    policy, and the daily Kaggle GPU cap. Every loop entry point
    (prepare-round, execute-round, run-parallel) reads it before doing work.

``pending_interaction.json``
    Questions the loop needs a human for. Asking never blocks: the request is
    recorded, a notification goes out, and the loop keeps working on anything the
    question does not block. ``kaggle respond`` answers it, and the experiments it
    blocked become runnable again.
"""

from __future__ import annotations

import re
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

from researchforge.kaggle.workspace import atomic_write_json, read_json

CONTROL_FILE = "loop_control.json"
INTERACTIONS_FILE = "pending_interaction.json"

NotificationLevel = Literal["verbose", "balanced", "minimal"]
SubmissionPolicy = Literal["aggressive", "balanced", "conservative"]
InteractionKind = Literal["submission_approval", "domain_question", "compute_question", "other"]

NOTIFICATION_LEVELS: tuple[NotificationLevel, ...] = ("verbose", "balanced", "minimal")
SUBMISSION_POLICIES: tuple[SubmissionPolicy, ...] = ("aggressive", "balanced", "conservative")

# What each notification level lets through. Categories are assigned by the sender.
#   result       — every experiment result, including no-change and failures
#   improvement  — a result that beat the best so far
#   ideas        — new ideas reached the pool (kernel watcher, arXiv, pool growth)
#   kernel       — a public kernel that beats your baseline
#   merge        — two independent wins that can be combined
#   candidate    — a submission candidate is ready
#   interaction  — the loop needs your answer
#   briefing     — the scheduled morning summary
#   alert        — something is wrong (hung experiment, sensor error)
_LEVEL_ALLOWS: dict[NotificationLevel, frozenset[str]] = {
    "verbose": frozenset(
        {
            "result",
            "improvement",
            "ideas",
            "kernel",
            "merge",
            "candidate",
            "interaction",
            "briefing",
            "alert",
        }
    ),
    "balanced": frozenset(
        {"improvement", "kernel", "merge", "candidate", "interaction", "briefing", "alert"}
    ),
    "minimal": frozenset({"candidate", "interaction", "alert"}),
}


@dataclass
class LoopControl:
    """The human's standing decisions for one competition's loop."""

    paused: bool = False
    paused_at: str = ""
    pause_reason: str = ""
    notification_level: NotificationLevel = "balanced"
    submission_policy: SubmissionPolicy = "balanced"
    # Max experiments per local calendar day that may run on Kaggle GPU. 0 = no cap.
    kaggle_gpu_daily_cap: int = 0
    updated_at: str = ""

    def save(self, artifacts_dir: Path) -> None:
        self.updated_at = datetime.now().isoformat(timespec="seconds")
        atomic_write_json(artifacts_dir / CONTROL_FILE, asdict(self))

    @classmethod
    def load(cls, artifacts_dir: Path) -> LoopControl:
        raw = read_json(artifacts_dir / CONTROL_FILE, {}) or {}
        known = set(cls.__dataclass_fields__)
        ctl = cls(**{k: v for k, v in raw.items() if k in known})
        # An edited file with an unknown value falls back to the default, never crashes.
        if ctl.notification_level not in NOTIFICATION_LEVELS:
            ctl.notification_level = "balanced"
        if ctl.submission_policy not in SUBMISSION_POLICIES:
            ctl.submission_policy = "balanced"
        return ctl


def pause(artifacts_dir: Path, reason: str = "") -> LoopControl:
    ctl = LoopControl.load(artifacts_dir)
    ctl.paused = True
    ctl.paused_at = datetime.now().isoformat(timespec="seconds")
    ctl.pause_reason = reason
    ctl.save(artifacts_dir)
    return ctl


def resume(artifacts_dir: Path) -> LoopControl:
    ctl = LoopControl.load(artifacts_dir)
    ctl.paused = False
    ctl.paused_at = ""
    ctl.pause_reason = ""
    ctl.save(artifacts_dir)
    return ctl


def should_notify(artifacts_dir: Path | None, category: str) -> bool:
    """True if this competition's notification level lets ``category`` through."""
    level: NotificationLevel = (
        LoopControl.load(artifacts_dir).notification_level if artifacts_dir else "balanced"
    )
    return category in _LEVEL_ALLOWS[level]


# ---------------------------------------------------------------------------
# Pending interactions
# ---------------------------------------------------------------------------


@dataclass
class Interaction:
    id: str
    kind: InteractionKind
    question: str
    options: list[str] = field(default_factory=list)
    default: str = ""
    # Hypothesis ideas (exact pool text) that must wait for the answer.
    blocks: list[str] = field(default_factory=list)
    created_at: str = ""
    status: Literal["open", "answered", "dismissed"] = "open"
    answer: str = ""
    answered_at: str = ""
    # Where the answer is applied: a setup question id goes into user_answers.json.
    answer_key: str = ""
    # Who asked: "setup", "auto-run (claude)", "you", … — shown next to the question.
    source: str = ""
    # True: no new round starts until it is answered. False: only `blocks` ideas wait.
    holds_loop: bool = False
    # The person's own words beside their answer; the next proposal reads it.
    comment: str = ""
    # Settings an answer changes, applied when the person picks it: {answer: {field: value}}.
    actions: dict[str, dict[str, Any]] = field(default_factory=dict)
    # What picking the answer changed, e.g. "compute: colab_gpu → kaggle_gpu".
    applied: list[str] = field(default_factory=list)


def load_interactions(artifacts_dir: Path) -> list[Interaction]:
    raw = read_json(artifacts_dir / INTERACTIONS_FILE, []) or []
    known = set(Interaction.__dataclass_fields__)
    out: list[Interaction] = []
    for item in raw if isinstance(raw, list) else []:
        if isinstance(item, dict) and item.get("id") and item.get("question"):
            out.append(Interaction(**{k: v for k, v in item.items() if k in known}))
    return out


def _save_interactions(artifacts_dir: Path, items: list[Interaction]) -> None:
    atomic_write_json(artifacts_dir / INTERACTIONS_FILE, [asdict(i) for i in items])


def open_interactions(artifacts_dir: Path) -> list[Interaction]:
    return [i for i in load_interactions(artifacts_dir) if i.status == "open"]


def request_interaction(
    artifacts_dir: Path,
    question: str,
    *,
    kind: InteractionKind = "other",
    options: list[str] | None = None,
    default: str = "",
    blocks: list[str] | None = None,
    answer_key: str = "",
    source: str = "",
    holds_loop: bool = False,
    actions: dict[str, dict[str, Any]] | None = None,
) -> Interaction:
    """Record a question for the human. Never blocks; an identical open question is reused."""
    actions = check_actions(options or [], actions or {})
    items = load_interactions(artifacts_dir)
    for existing in items:
        if existing.status == "open" and existing.question == question:
            return existing
    item = Interaction(
        id=f"q-{uuid.uuid4().hex[:6]}",
        kind=kind,
        question=question,
        options=list(options or []),
        default=default,
        blocks=list(blocks or []),
        created_at=datetime.now().isoformat(timespec="seconds"),
        answer_key=answer_key,
        source=source,
        holds_loop=holds_loop,
        actions=actions,
    )
    items.append(item)
    _save_interactions(artifacts_dir, items)
    return item


# Run configs fields an answer may change. Topics, drivers and auto-run limits stay
# with the person's own Run configs.
ACTION_FIELDS = (
    "compute",
    "max_experiments",
    "time_budget_hours",
    "round_time_limit_minutes",
    "kaggle_gpu_daily_cap",
    "colab_plan",
    "colab_gpu",
    "colab_high_mem",
    "metric_direction",
)


def check_actions(
    options: list[str], actions: dict[str, dict[str, Any]]
) -> dict[str, dict[str, Any]]:
    """Validate the settings each answer changes: known answers, known fields."""
    out: dict[str, dict[str, Any]] = {}
    for answer, changes in actions.items():
        if answer not in options:
            raise ValueError(f"a settings change is for {answer!r}, which is not an answer")
        if not isinstance(changes, dict) or not changes:
            raise ValueError(f"name the settings {answer!r} changes")
        for name in changes:
            if name not in ACTION_FIELDS:
                raise ValueError(
                    f"an answer cannot change {name!r}; it can change {', '.join(ACTION_FIELDS)}"
                )
        out[answer] = dict(changes)
    return out


def parse_action(text: str) -> tuple[str, dict[str, str]]:
    """``"<answer> => field:value, field:value"`` from the command line."""
    answer, sep, rest = text.rpartition("=>")
    if not sep or not answer.strip():
        raise ValueError(f'write it as "<answer> => field:value", not {text!r}')
    changes: dict[str, str] = {}
    for part in rest.split(","):
        name, colon, value = part.partition(":")
        if not colon or not name.strip() or not value.strip():
            raise ValueError(f"write each change as field:value, not {part.strip()!r}")
        changes[name.strip()] = value.strip()
    return answer.strip(), changes


def _apply_actions(artifacts_dir: Path, item: Interaction, answer: str) -> list[str]:
    """Apply the settings ``answer`` changes, through the same setters as Run configs."""
    import os

    from researchforge.kaggle import run_config

    changes = item.actions.get(answer) or {}
    if not changes:
        return []
    if os.environ.get("RESEARCHFORGE_AUTO_RUN") == "1":
        raise ValueError(
            "this answer changes Run configs, so only the person can pick it "
            "(on the dashboard or with kaggle respond in their own terminal)"
        )
    done: list[str] = []
    for name, value in changes.items():
        try:
            before, after = run_config.set_field(
                artifacts_dir, name, value, via=f"answer to {item.id}"
            )
        except ValueError as exc:
            made = f" (already changed: {'; '.join(done)})" if done else ""
            raise ValueError(f"could not set {name} to {value}: {exc}{made}") from None
        if str(before) != str(after):  # an answer picked after the change was made by hand
            done.append(f"{name}: {before} → {after}")
    return done


def ask(
    artifacts_dir: Path,
    question: str,
    *,
    options: list[str] | None = None,
    default: str = "",
    blocks: list[str] | None = None,
    holds_loop: bool = False,
    source: str = "",
    actions: dict[str, dict[str, Any]] | None = None,
) -> Interaction:
    """Ask the person a question mid-loop and push it to their phone.

    ``blocks``: pool ideas that wait for the answer (every other idea keeps
    running). ``holds_loop``: nothing new starts until it is answered.
    ``actions``: the Run configs settings an answer changes, applied when the
    person picks it — so "switch to Kaggle GPU" does what it says.

    A yes/no question about submitting a round ("Submit R21's output to Kaggle?")
    becomes that round's submission question, so a "yes" submits it. ValueError
    when that round left nothing Kaggle takes.
    """
    round_num = submission_round(question, options)
    if round_num is not None:
        return ask_submission(artifacts_dir, round_num, question, source=source)
    item = request_interaction(
        artifacts_dir,
        question,
        options=options,
        default=default,
        blocks=blocks,
        source=source,
        holds_loop=holds_loop,
        actions=actions,
    )
    slug = artifacts_dir.name.removeprefix(".overnight_")
    opts = f" [{' / '.join(item.options)}]" if item.options else ""
    notify(
        artifacts_dir,
        "interaction",
        f"Question{' (loop on hold)' if holds_loop else ''}: {question}{opts}\n"
        f'Answer on the dashboard, or: researchforge kaggle respond {slug} {item.id} "<answer>"',
        priority="high" if holds_loop else "default",
    )
    return item


def holding_questions(artifacts_dir: Path) -> list[Interaction]:
    """Open questions that hold the whole loop (see ``ask``)."""
    return [i for i in open_interactions(artifacts_dir) if i.holds_loop]


def person_notes(artifacts_dir: Path, limit: int = 8) -> list[str]:
    """The person's directions, then their recent comments (answers, dismissals, repo plans).

    Unattended launches and prepare-round pass these to whoever proposes next, so a
    comment like "stop trying LoRA, the adapter is broken" is acted on.
    """
    notes: list[tuple[str, str]] = []
    for i in load_interactions(artifacts_dir):
        if i.comment:
            notes.append(
                (i.answered_at, f'On "{i.question[:120]}" ({i.answer or i.status}): {i.comment}')
            )
    feedback = artifacts_dir.parent / ".researchforge" / PLAN_FEEDBACK_FILE
    if feedback.is_file():
        import json

        for line in feedback.read_text(encoding="utf-8").splitlines():
            try:
                f = json.loads(line)
            except ValueError:
                continue
            if f.get("comment"):
                notes.append(
                    (
                        str(f.get("at")),
                        f"On repo {f.get('plan')} ({f.get('decision')}): {f['comment']}",
                    )
                )
    notes.sort(key=lambda n: n[0] or "", reverse=True)
    from researchforge.kaggle.steering import prompt_lines

    # Directions first and never cut: the person set them to steer every proposal.
    return prompt_lines(artifacts_dir) + [n for _at, n in notes[:limit]]


PLAN_FEEDBACK_FILE = "plan_feedback.jsonl"


def record_plan_feedback(repo_dir: Path, plan_id: str, decision: str, comment: str) -> None:
    import json

    with (repo_dir / PLAN_FEEDBACK_FILE).open("a", encoding="utf-8") as fh:
        fh.write(
            json.dumps(
                {
                    "at": datetime.now().isoformat(timespec="seconds"),
                    "plan": plan_id,
                    "decision": decision,
                    "comment": comment.strip(),
                }
            )
            + "\n"
        )


def dismiss(
    artifacts_dir: Path, interaction_id: str, reason: str = "", comment: str = ""
) -> Interaction:
    """Close a question without answering it. Ideas it blocked run again; nothing is applied.

    For questions that no longer matter (the situation changed, or the person is
    fixing it another way). Submission questions dismissed this way mean "not now".
    """
    items = load_interactions(artifacts_dir)
    match = next((i for i in items if i.id == interaction_id), None)
    if match is None:
        raise KeyError(f"No interaction with id {interaction_id!r}")
    if match.status != "open":
        raise ValueError(f"{interaction_id} is already {match.status}")
    match.status = "dismissed"
    match.answer = reason
    match.comment = comment.strip()
    match.answered_at = datetime.now().isoformat(timespec="seconds")
    _save_interactions(artifacts_dir, items)
    return match


def respond(
    artifacts_dir: Path, interaction_id: str, answer: str, comment: str = ""
) -> Interaction:
    """Answer an open interaction. Domain answers are merged into user_answers.json."""
    items = load_interactions(artifacts_dir)
    match = next((i for i in items if i.id == interaction_id), None)
    if match is None:
        raise KeyError(f"No interaction with id {interaction_id!r}")
    if match.status != "open":
        raise ValueError(f"{interaction_id} was already {match.status}: {match.answer!r}")
    if match.options and answer not in match.options:
        raise ValueError(f"Answer must be one of {match.options}")
    # Settings first: if one is refused, the question stays open with the reason.
    match.applied = _apply_actions(artifacts_dir, match, answer)
    match.status = "answered"
    match.answer = answer
    match.comment = comment.strip()
    match.answered_at = datetime.now().isoformat(timespec="seconds")
    _save_interactions(artifacts_dir, items)

    if match.answer_key and match.kind == "domain_question":
        from researchforge.kaggle.human_questions import UserAnswers

        answers = UserAnswers.load(artifacts_dir)
        answers.answers[match.answer_key] = answer
        answers.save(artifacts_dir)
        # Proposals read the answers embedded in overnight_setup.json, so update that copy too.
        setup_path = artifacts_dir / "overnight_setup.json"
        setup = read_json(setup_path, None)
        if isinstance(setup, dict):
            ua = setup.get("user_answers") or {"answers": {}, "asked_at": ""}
            ua.setdefault("answers", {})[match.answer_key] = answer
            setup["user_answers"] = ua
            atomic_write_json(setup_path, setup)
    return match


def blocked_ideas(artifacts_dir: Path) -> set[str]:
    """Hypothesis ideas that wait on an unanswered question."""
    return {idea for i in open_interactions(artifacts_dir) for idea in i.blocks}


# ---------------------------------------------------------------------------
# Kaggle GPU daily cap
# ---------------------------------------------------------------------------

_KAGGLE_TARGETS = ("kaggle_gpu", "kaggle_gpu_offline", "competition_gpu")


def kaggle_gpu_runs_today(artifacts_dir: Path) -> int:
    """Experiments started today on Kaggle compute, from compute_usage.json."""
    usage = read_json(artifacts_dir / "compute_usage.json", {}) or {}
    today = datetime.now().date().isoformat()
    day = usage.get(today, {}) if isinstance(usage, dict) else {}
    return sum(int(day.get(t, 0)) for t in _KAGGLE_TARGETS)


def record_compute_use(artifacts_dir: Path, target: str) -> None:
    usage: dict[str, Any] = read_json(artifacts_dir / "compute_usage.json", {}) or {}
    today = datetime.now().date().isoformat()
    day = usage.setdefault(today, {})
    day[target] = int(day.get(target, 0)) + 1
    atomic_write_json(artifacts_dir / "compute_usage.json", usage)


def kaggle_gpu_cap_reached(artifacts_dir: Path, target: str) -> bool:
    if target not in _KAGGLE_TARGETS:
        return False
    cap = LoopControl.load(artifacts_dir).kaggle_gpu_daily_cap
    return cap > 0 and kaggle_gpu_runs_today(artifacts_dir) >= cap


# ---------------------------------------------------------------------------
# Notifications and submission offers
# ---------------------------------------------------------------------------


def split_options(text: str) -> list[str]:
    """A question's answers from one string.

    Split on ``|`` when there is one, else on commas outside brackets and double quotes, so an
    answer like "Switch to local (runs run_benchmark.py here, which pushes kernels)"
    stays whole instead of turning into two half-answers.
    """
    if "|" in text:
        return [o.strip() for o in text.split("|") if o.strip()]
    out: list[str] = []
    buf: list[str] = []
    depth, quote = 0, ""
    for ch in text:
        if quote:
            quote = "" if ch == quote else quote
        elif ch == '"':
            quote = ch
        elif ch in "([{":
            depth += 1
        elif ch in ")]}":
            depth = max(0, depth - 1)
        elif ch == "," and depth == 0:
            out.append("".join(buf))
            buf = []
            continue
        buf.append(ch)
    out.append("".join(buf))
    return [o.strip() for o in out if o.strip()]


def notify_topic(artifacts_dir: Path | None) -> tuple[str, str]:
    """The ntfy topic this competition's alerts go to, and where it is set.

    ``NTFY_TOPIC`` in the environment wins, then the project's own topic (Run configs),
    then the machine-wide one (``researchforge notify setup``, or Run configs). Where is
    "env", "project", "machine", or "" when no topic is set and alerts are not sent.
    """
    import os

    env = os.environ.get("NTFY_TOPIC", "").strip()
    if env:
        return env, "env"
    if artifacts_dir is not None and artifacts_dir.name == ".researchforge":
        repo = read_json(artifacts_dir / "auto_run.json", {}) or {}  # a repo project's own
        own = str(repo.get("ntfy_topic") or "").strip() if isinstance(repo, dict) else ""
        if own:
            return own, "project"
    elif artifacts_dir is not None:
        setup = read_json(artifacts_dir / "overnight_setup.json", {}) or {}
        setup = setup if isinstance(setup, dict) else {}
        answers = (setup.get("user_answers") or {}).get("answers") or {}
        # A topic the person gave as a setup answer counts: that is where they subscribed.
        own = str(setup.get("ntfy_topic") or answers.get("ntfy_topic") or "").strip()
        if own:
            return own, "project"
    from researchforge.notifications.service import _load_notify_config

    machine = str(_load_notify_config().get("ntfy_topic") or "").strip()
    return (machine, "machine") if machine else ("", "")


def notify(
    artifacts_dir: Path, category: str, message: str, *, topic: str = "", priority: str = "default"
) -> bool:
    """Push a notification if this competition's level allows ``category``. Never raises."""
    import os

    if not should_notify(artifacts_dir, category):
        return False
    topic = topic or notify_topic(artifacts_dir)[0]
    if not topic:
        return False
    try:
        from researchforge.notifications.service import Channel, Notification, send_notification

        old = os.environ.get("NTFY_TOPIC")
        os.environ["NTFY_TOPIC"] = topic
        try:
            slug = (
                artifacts_dir.parent.name  # a repo project: its folder's name
                if artifacts_dir.name == ".researchforge"
                else artifacts_dir.name.removeprefix(".overnight_")
            )
            return send_notification(
                Notification(
                    title=f"ResearchForge: {slug}",
                    message=message,
                    channel=Channel.NTFY,
                    priority=priority,
                )
            )
        finally:
            if old is None:
                os.environ.pop("NTFY_TOPIC", None)
            else:
                os.environ["NTFY_TOPIC"] = old
    except Exception:
        return False


# "Submit R21's output to Kaggle?", "Should we submit R5?" (the round asked about).
_ASKS_SUBMIT = re.compile(r"(?i)\bsubmit\b[^?]*?\bR(\d+)\b")
_YES_NO = {"yes", "no", "y", "n"}


def submission_round(question: str, options: list[str] | None = None) -> int | None:
    """The round a yes/no question asks to submit, or None for any other question."""
    m = _ASKS_SUBMIT.search(question or "")
    if m is None or any(o.strip().lower() not in _YES_NO for o in options or []):
        return None
    return int(m.group(1))


def ask_submission(
    artifacts_dir: Path,
    round_num: int,
    question: str,
    *,
    source: str = "",
    require_output: bool = True,
) -> Interaction:
    """Ask whether to submit one round; answering "yes" submits it (kaggle.submission)."""
    from researchforge.kaggle.submission import submittable

    if require_output and not submittable(artifacts_dir, round_num):
        raise ValueError(
            f"R{round_num} left nothing Kaggle takes (no kept submission file or "
            "notebook run), so it cannot be submitted; nothing was asked"
        )
    item = request_interaction(
        artifacts_dir,
        question,
        kind="submission_approval",
        options=["yes", "no"],
        default="no",
        answer_key=f"round:{round_num}",
        source=source or "submission offer",
    )
    slug = artifacts_dir.name.removeprefix(".overnight_")
    notify(
        artifacts_dir,
        "candidate",
        f"🎯 Submission candidate: {question}\n"
        f"Approve on the dashboard (🔔) to submit it, or: researchforge kaggle respond {slug} "
        f"{item.id} yes",
        priority="high",
    )
    return item


def offer_submission(
    artifacts_dir: Path, round_num: int, title: str, metric: float | None, target: str
) -> Interaction | None:
    """Ask the human whether to submit an improved round, as the submission policy says.

    aggressive   — after any improved round, even one that left nothing to submit
    balanced     — after an improved round that left something Kaggle takes (its
                   Kaggle notebook or a submission file), wherever it ran
    conservative — never proactively; the user asks for candidates themselves

    Nothing is submitted until the person answers "yes" (kaggle.submission does it then).
    """
    from researchforge.kaggle.submission import submittable

    policy = LoopControl.load(artifacts_dir).submission_policy
    if policy == "conservative":
        return None
    if policy == "balanced" and not submittable(artifacts_dir, round_num):
        return None
    scored = isinstance(metric, (int, float))
    shown = f"{metric:.4f}" if scored else "unscored"
    measured = (
        f"metric {shown}, measured on {target}"
        if scored
        else f"ran on {target} and wrote a submission but printed no score; "
        "only Kaggle can score it"
    )
    return ask_submission(
        artifacts_dir,
        round_num,
        f"Submit R{round_num} '{title}' ({measured}) to Kaggle?",
        require_output=policy == "balanced",
    )
