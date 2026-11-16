"""Human-in-the-loop commands for a running overnight loop.

  researchforge kaggle status <slug>                     where the loop is, in one screen
  researchforge kaggle pause <slug> [--reason]           stop starting new rounds
  researchforge kaggle resume <slug>                     start again
  researchforge kaggle respond <slug> [<id> <answer>]    answer a question the loop asked
  researchforge kaggle compute <slug> <target>           change compute for the next rounds
  researchforge kaggle add-hypothesis <slug> "<idea>"    inject an idea into the pool now
  researchforge kaggle settings <slug> [--notify-level ...] [--submission-policy ...]

None of these interrupt a round that is already running; they take effect at the next
prepare-round / execute-round.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Annotated, Any

import typer

from researchforge.kaggle import control
from researchforge.kaggle.cli import kaggle_app
from researchforge.kaggle.gpu_detector import COMPUTE_CHOICES
from researchforge.kaggle.hypothesis_pool import setup_direction as _setup_direction
from researchforge.kaggle.orchestrator import OvernightOrchestrator
from researchforge.kaggle.workspace import (
    atomic_write_json,
    has_setup,
    read_json,
    resolve_artifacts_dir,
)

_LOCAL_TARGETS = {"cpu"}

SlugArg = Annotated[str, typer.Argument(help="Kaggle competition slug or URL")]


def _artifacts(competition: str) -> Path:
    slug = OvernightOrchestrator._parse_slug(competition)
    d = resolve_artifacts_dir(slug, register=False)
    if not (d / "overnight_setup.json").exists():
        typer.secho(
            f"❌ No overnight setup for '{slug}' (looked in {d}). "
            f"Run 'researchforge kaggle overnight {slug}' first.",
            fg=typer.colors.RED,
            err=True,
        )
        raise typer.Exit(1)
    return d


def _repo_project(d: Path) -> dict[str, Any] | None:
    """The repo project sharing this overnight dir's root, if any (read-only)."""
    import sqlite3

    db = d.parent / ".researchforge" / "researchforge.db"
    if not db.is_file():
        return None
    try:
        with sqlite3.connect(f"file:{db}?mode=ro", uri=True, timeout=2.0) as conn:
            (count,) = conn.execute("SELECT COUNT(*) FROM experiments").fetchone()
    except sqlite3.Error:
        count = 0
    return {
        "root": str(d.parent),
        "experiments": int(count),
        "next_command": "researchforge autorun --dry-run",
    }


def build_status(d: Path) -> dict[str, Any]:
    """One-screen status, read from the same files the dashboard reads."""
    from researchforge.kaggle.dashboard_state import _best, _read_overnight

    slug = d.name.removeprefix(".overnight_")
    o = _read_overnight(d, slug, [])
    exps = o["experiments"]
    done = [e for e in exps if e.get("status") not in ("running", "stale")]
    best = _best(exps)
    pool = o["hypotheses"]
    blocked = control.blocked_ideas(d)
    untested = [h for h in pool if not h.get("tested")]
    ctl = control.LoopControl.load(d)
    inflight = [e for e in exps if e.get("status") == "running"]
    return {
        "slug": slug,
        "artifacts_dir": str(d),
        # A new chat or `kaggle overnight` resumes a set-up project instead of
        # repeating setup; this is the field they check.
        "set_up": has_setup(d),
        "next_command": f"researchforge kaggle prepare-round {slug}"
        if has_setup(d)
        else f"researchforge kaggle overnight {slug}",
        # Runs driven by repo autorun live in the repo project, not experiment_log.db.
        "repo_project": _repo_project(d),
        "paused": ctl.paused,
        "pause_reason": ctl.pause_reason,
        "compute": o["setup"]["gpu"] or "cpu",
        "baseline": o["setup"]["baseline_cv"],
        "metric": o["competition"]["metric"],
        "direction": o["competition"]["direction"],
        "rounds_completed": len(done),
        "running": [{"round": e.get("num"), "title": e.get("title")} for e in inflight],
        "best": {"round": best.get("num"), "metric": best.get("metric"), "title": best.get("title")}
        if best
        else None,
        "hypotheses": {
            "untested": len(untested),
            "blocked_by_questions": len([h for h in untested if h.get("idea") in blocked]),
            "total": len(pool),
        },
        "open_questions": [asdict(i) for i in control.open_interactions(d)],
        "settings": {
            "notification_level": ctl.notification_level,
            "submission_policy": ctl.submission_policy,
            "kaggle_gpu_daily_cap": ctl.kaggle_gpu_daily_cap,
            "kaggle_gpu_runs_today": control.kaggle_gpu_runs_today(d),
        },
        "merge_candidate": o["merge_candidate"],
        "next_sensor_cycle": (read_json(d / "monitor_state.json", {}) or {}).get("next_cycle_at"),
    }


@kaggle_app.command("status")
def status_cmd(
    competition: SlugArg,
    as_json: Annotated[bool, typer.Option("--json", help="Machine-readable output")] = False,
) -> None:
    """Where the loop is: paused?, running round, best result, pool, open questions."""
    st = build_status(_artifacts(competition))
    if as_json:
        typer.echo(json.dumps(st, indent=2, default=str))
        return
    typer.secho(f"\n{st['slug']}", bold=True)
    state = "⏸  PAUSED" + (f" — {st['pause_reason']}" if st["pause_reason"] else "")
    typer.echo(f"  State:      {state if st['paused'] else '▶  active'}")
    typer.echo(f"  Compute:    {st['compute']}")
    typer.echo(f"  Baseline:   {st['baseline']}  ({st['metric']}, {st['direction']})")
    best = st["best"]
    typer.echo(
        f"  Best:       R{best['round']} = {best['metric']}  {best['title'] or ''}"
        if best
        else "  Best:       no scored round yet"
    )
    typer.echo(f"  Rounds:     {st['rounds_completed']} completed")
    for r in st["running"]:
        typer.echo(f"  Running:    R{r['round']} {r['title'] or ''}")
    h = st["hypotheses"]
    typer.echo(
        f"  Ideas:      {h['untested']} untested of {h['total']}"
        + (
            f" ({h['blocked_by_questions']} waiting on your answer)"
            if h["blocked_by_questions"]
            else ""
        )
    )
    s = st["settings"]
    cap = s["kaggle_gpu_daily_cap"]
    typer.echo(
        f"  Settings:   notify={s['notification_level']}  submit={s['submission_policy']}  "
        f"kaggle-gpu today={s['kaggle_gpu_runs_today']}" + (f"/{cap}" if cap else "")
    )
    if st["merge_candidate"]:
        typer.echo("  🔀 Merge opportunity — see prepare-round output")
    if st["open_questions"]:
        typer.secho(f"\n  {len(st['open_questions'])} question(s) for you:", fg=typer.colors.YELLOW)
        for q in st["open_questions"]:
            opts = f"  [{' / '.join(q['options'])}]" if q["options"] else ""
            typer.echo(f"    {q['id']}: {q['question']}{opts}")
        typer.echo(f'  Answer: researchforge kaggle respond {st["slug"]} <id> "<answer>"')


@kaggle_app.command("pause")
def pause_cmd(
    competition: SlugArg,
    reason: Annotated[str, typer.Option("--reason", help="Why (shown on the dashboard)")] = "",
) -> None:
    """Stop starting new rounds. A round already running finishes; sensors keep running."""
    d = _artifacts(competition)
    control.pause(d, reason)
    typer.secho(
        "⏸  Paused. prepare-round and execute-round will not start new rounds.",
        fg=typer.colors.YELLOW,
    )
    typer.echo(f"   Resume with: researchforge kaggle resume {d.name.removeprefix('.overnight_')}")


@kaggle_app.command("resume")
def resume_cmd(competition: SlugArg) -> None:
    """Start new rounds again after a pause."""
    d = _artifacts(competition)
    was = control.LoopControl.load(d).paused
    control.resume(d)
    typer.secho("▶  Resumed." if was else "▶  The loop was not paused.", fg=typer.colors.GREEN)


@kaggle_app.command("ask")
def ask_cmd(
    competition: SlugArg,
    question: Annotated[str, typer.Argument(help="The question, in one sentence")],
    options: Annotated[
        str,
        typer.Option(
            "--options",
            help="Allowed answers, comma-separated (e.g. yes,no). Commas inside brackets or "
            "quotes stay in the answer; separate with | to keep every comma.",
        ),
    ] = "",
    default: Annotated[str, typer.Option("--default", help="Suggested answer")] = "",
    blocks: Annotated[
        list[str] | None,
        typer.Option("--blocks", help="An exact pool idea that waits for the answer (repeatable)"),
    ] = None,
    hold: Annotated[
        bool, typer.Option("--hold", help="Start no new round until it is answered")
    ] = False,
    action: Annotated[
        list[str] | None,
        typer.Option(
            "--action",
            help='Settings an answer changes when the person picks it: "<answer> => '
            'compute:kaggle, max_experiments:40" (repeatable, one per answer)',
        ),
    ] = None,
) -> None:
    """Ask the person a question without stopping the loop; it reaches their phone.

    Ideas named with --blocks wait for the answer; everything else keeps running.
    Answers come from the dashboard's bell, `kaggle respond`, or a chat. An answer
    given an --action changes those Run configs settings when the person picks it.
    """
    import os as _os

    from researchforge.kaggle.auto_run import detect_host

    d = _artifacts(competition)
    if _os.environ.get("RESEARCHFORGE_AUTO_RUN") == "1":
        source = f"auto-run ({_os.environ.get('RESEARCHFORGE_DRIVER') or 'agent'})"
    else:
        host = detect_host()[0]
        source = {"claude": "Claude Code chat", "cursor": "Cursor chat"}.get(host, "you")
    opts = control.split_options(options)
    if default and opts and default not in opts:
        typer.secho(f"❌ --default must be one of {opts}", fg=typer.colors.RED, err=True)
        raise typer.Exit(1)
    try:
        actions = dict(control.parse_action(a) for a in action or [])
        item = control.ask(
            d,
            question,
            options=opts,
            default=default,
            blocks=blocks or [],
            holds_loop=hold,
            source=source,
            actions=actions,
        )
    except ValueError as exc:
        typer.secho(f"❌ {exc}", fg=typer.colors.RED, err=True)
        raise typer.Exit(1) from None
    typer.secho(f"✅ Asked {item.id}: {item.question}", fg=typer.colors.GREEN)
    if item.kind == "submission_approval":
        typer.echo("   It is this round's submission question: answering yes submits it.")
    if item.blocks:
        typer.echo(f"   {len(item.blocks)} idea(s) wait for the answer; the rest keep running.")
    if item.holds_loop:
        typer.echo("   No new round starts until it is answered.")
    for ans, changes in item.actions.items():
        what = ", ".join(f"{k} → {v}" for k, v in changes.items())
        typer.echo(f"   Picking {ans!r} changes Run configs: {what}")
    slug = d.name.removeprefix(".overnight_")
    typer.echo(f'   Answer: researchforge kaggle respond {slug} {item.id} "<answer>"')


@kaggle_app.command("offer-submission")
def offer_submission_cmd(
    competition: SlugArg,
    round_num: Annotated[int, typer.Argument(help="The round to offer, e.g. 21")],
    note: Annotated[
        str, typer.Option("--note", help="What the person should know, e.g. its LB claim")
    ] = "",
) -> None:
    """Ask the person whether to submit one round; their "yes" submits it.

    For a round the loop did not offer by itself, such as a public notebook run
    unscored. Nothing is submitted until the person answers.
    """
    import os as _os

    from researchforge.kaggle.late_rounds import _round_title

    d = _artifacts(competition)
    title = _round_title(d, round_num)
    question = f"Submit R{round_num} '{title}'" + (f" ({note})" if note else "") + " to Kaggle?"
    source = (
        f"auto-run ({_os.environ.get('RESEARCHFORGE_DRIVER') or 'agent'})"
        if _os.environ.get("RESEARCHFORGE_AUTO_RUN") == "1"
        else "you"
    )
    try:
        item = control.ask_submission(d, round_num, question, source=source)
    except ValueError as exc:
        typer.secho(f"❌ {exc}", fg=typer.colors.RED, err=True)
        raise typer.Exit(1) from None
    slug = d.name.removeprefix(".overnight_")
    typer.secho(f"✅ Asked {item.id}: {item.question}", fg=typer.colors.GREEN)
    typer.echo(f"   Approve: researchforge kaggle respond {slug} {item.id} yes (or the 🔔)")


@kaggle_app.command("respond")
def respond_cmd(
    competition: SlugArg,
    interaction_id: Annotated[str, typer.Argument(help="Question id from `kaggle status`")] = "",
    answer: Annotated[str, typer.Argument(help="Your answer")] = "",
    dismiss: Annotated[
        bool, typer.Option("--dismiss", help="Close it without answering; its ideas run again")
    ] = False,
    comment: Annotated[
        str, typer.Option("--comment", help="Your own words with it; the next proposal reads them")
    ] = "",
) -> None:
    """Answer a question the loop asked. With no id, lists the open questions."""
    d = _artifacts(competition)
    if dismiss:
        try:
            control.dismiss(d, interaction_id, "dismissed in a terminal", comment)
        except (KeyError, ValueError) as exc:
            typer.secho(f"❌ {exc}", fg=typer.colors.RED, err=True)
            raise typer.Exit(1) from None
        typer.secho(f"✅ {interaction_id} dismissed.", fg=typer.colors.GREEN)
        return
    open_qs = control.open_interactions(d)
    if not interaction_id:
        if not open_qs:
            typer.echo("No open questions.")
            return
        for q in open_qs:
            opts = f"  [{' / '.join(q.options)}]" if q.options else ""
            dflt = f"  (default: {q.default})" if q.default else ""
            typer.echo(f"{q.id}: {q.question}{opts}{dflt}")
        return
    if not answer:
        typer.secho(
            '❌ Give the answer too: kaggle respond <slug> <id> "<answer>"',
            fg=typer.colors.RED,
            err=True,
        )
        raise typer.Exit(1)
    asked = next((q for q in open_qs if q.id == interaction_id), None)
    if asked is not None and asked.kind == "submission_approval":
        from researchforge.kaggle.submission import answer as answer_submission

        slug = d.name.removeprefix(".overnight_")
        try:
            line = answer_submission(d, slug, interaction_id, answer, comment)
        except (KeyError, ValueError) as exc:
            typer.secho(f"❌ {exc}", fg=typer.colors.RED, err=True)
            raise typer.Exit(1) from None
        typer.secho(f"✅ {line}", fg=typer.colors.GREEN)
        return
    try:
        item = control.respond(d, interaction_id, answer, comment)
    except (KeyError, ValueError) as exc:
        typer.secho(f"❌ {exc}", fg=typer.colors.RED, err=True)
        raise typer.Exit(1) from None
    typer.secho(f"✅ {item.id} answered: {item.answer}", fg=typer.colors.GREEN)
    for change in item.applied:
        typer.echo(f"   Run configs: {change}")
    if item.blocks:
        typer.echo(f"   {len(item.blocks)} idea(s) it was blocking are runnable again.")


@kaggle_app.command("compute")
def compute_cmd(
    competition: SlugArg,
    target: Annotated[str, typer.Argument(help=f"One of: {COMPUTE_CHOICES}")],
) -> None:
    """Switch compute for the next rounds.

    Refuses targets the data policy forbids, and ssh / cloud until they are configured.
    """
    from researchforge.kaggle.run_config import set_compute

    d = _artifacts(competition)
    try:
        before, backend = set_compute(d, target)
    except ValueError as e:
        typer.secho(f"❌ {e}", fg=typer.colors.RED, err=True)
        raise typer.Exit(1) from None
    typer.secho(f"✅ Compute: {before} → {backend} (from the next round)", fg=typer.colors.GREEN)


@kaggle_app.command("add-hypothesis")
def add_hypothesis_cmd(
    competition: SlugArg,
    idea: Annotated[str, typer.Argument(help="The idea, in one sentence")],
    rationale: Annotated[str, typer.Option("--rationale")] = "",
    priority: Annotated[int, typer.Option("--priority", help="1 = try first")] = 1,
) -> None:
    """Inject your own idea into the pool now; the next prepare-round sees it first."""
    d = _artifacts(competition)
    pool_path = d / "pending_hypotheses.json"
    pool: list[dict[str, Any]] = read_json(pool_path, []) or []
    if any(h.get("idea") == idea for h in pool):
        typer.echo("That idea is already in the pool.")
        return
    pool.insert(
        0,
        {
            "idea": idea,
            "rationale": rationale,
            "source": "user",
            "priority": priority,
            "tested": False,
        },
    )
    atomic_write_json(pool_path, pool)
    typer.secho(f"✅ Added to the pool ({len(pool)} ideas).", fg=typer.colors.GREEN)


@kaggle_app.command("draft")
def draft_cmd(
    competition: SlugArg,
    start: Annotated[
        bool, typer.Option("--start", help="Make a draft from gather-facts' output")
    ] = False,
    form_json: Annotated[
        str, typer.Option("--form-json", help="Answers to merge into the draft (JSON)")
    ] = "",
    run: Annotated[bool, typer.Option("--run", help="Run the draft's setup now")] = False,
    as_json: Annotated[bool, typer.Option("--json", help="Print the draft as JSON")] = False,
) -> None:
    """A setup in progress, shared with the dashboard's New competition page.

    The dashboard saves its answers here; /overnight reads them instead of asking again.
    A chat can also save answers (--start, --form-json) for the person to finish on the
    dashboard, or run the setup itself (--run) — the same commands either way.
    """
    import subprocess
    import sys

    from researchforge.kaggle import new_project as np

    slug = np.parse_slug(competition)
    d = resolve_artifacts_dir(slug, register=False)
    draft = np.load_draft(d)
    if start:
        facts = read_json(d / "facts.json", None)
        if not isinstance(facts, dict):
            typer.secho(f"❌ Run: researchforge kaggle gather-facts {slug}", fg="red", err=True)
            raise typer.Exit(1)
        form = (draft or {}).get("form") or np.default_form(facts)
        draft = np.save_draft(
            d,
            {"slug": slug, "root": str(d.parent), "via": "chat", "facts": facts,
             "form": form, "status": "answering"},
        )  # fmt: skip
        from researchforge.kaggle.workspace import register_overnight_dir

        register_overnight_dir(slug, d)
    if draft is None:
        typer.echo(f"No setup draft for {slug} (looked in {d}).")
        raise typer.Exit(1)
    if form_json:
        try:
            patch = json.loads(form_json)
        except ValueError as exc:
            typer.secho(f"❌ --form-json: {exc}", fg="red", err=True)
            raise typer.Exit(1) from None
        form = dict(draft.get("form") or {})
        for k, v in patch.items():  # one level deep: {"colab": {"gpu": "L4"}} keeps the plan
            form[k] = (
                {**form[k], **v} if isinstance(v, dict) and isinstance(form.get(k), dict) else v
            )
        draft = np.save_draft(d, {"form": form})
    form = draft.get("form") or {}
    try:
        cmds = np.setup_commands(slug, form, draft.get("facts") or {}) if form else []
        problem = ""
    except ValueError as exc:
        cmds, problem = [], str(exc)
    missing = np.missing_keys(form) if form else []
    if as_json:
        typer.echo(json.dumps({**draft, "commands": cmds, "problem": problem,
                               "missing_keys": missing}, indent=2))  # fmt: skip
    elif not run:
        typer.echo(f"Setup draft for {slug} — {draft.get('status')} (from {draft.get('via')})")
        typer.echo(f"   {d}")
        if problem:
            typer.secho(f"   ⚠️  {problem}", fg="yellow")
        for k in missing:
            typer.echo(f"   🔑 needs {k}: researchforge kaggle keys open {k}")
        for c in cmds:
            typer.echo("   researchforge kaggle " + " ".join(c))
        typer.echo("   Finish it on the dashboard (New competition) or with --run.")
    if not run:
        return
    if problem or missing:
        typer.secho(f"❌ {problem or 'set these keys first: ' + ', '.join(missing)}", fg="red")
        raise typer.Exit(1)
    np.save_draft(d, {"status": "setting_up"})
    for c in cmds:
        typer.secho("→ researchforge kaggle " + " ".join(c), bold=True)
        if subprocess.run([sys.executable, "-m", "researchforge", "kaggle", *c],
                          cwd=d.parent, check=False).returncode != 0:  # fmt: skip
            np.save_draft(d, {"status": "answering"})
            typer.secho("❌ That step failed; the draft is kept. Fix it and --run again.", fg="red")
            raise typer.Exit(1)
    if str(form.get("ntfy_topic") or "").strip():
        from researchforge.kaggle.run_config import set_field

        set_field(d, "ntfy_topic", str(form["ntfy_topic"]).strip(), via="cli")
    (d / np.DRAFT_FILE).unlink(missing_ok=True)
    typer.secho(f"✅ {slug} is set up.", fg="green")


@kaggle_app.command("steer")
def steer_cmd(
    competition: SlugArg,
    text: Annotated[str, typer.Argument(help="The direction, or the reply with --reply")] = "",
    standing: Annotated[
        bool, typer.Option("--standing", help="Keep it for every round until removed")
    ] = False,
    reply_to: Annotated[
        str, typer.Option("--reply", help="Answer this direction id (the proposer does this)")
    ] = "",
    remove: Annotated[str, typer.Option("--remove", help="Remove this direction id")] = "",
    as_json: Annotated[bool, typer.Option("--json", help="Print JSON")] = False,
) -> None:
    """Steer the research: add a direction, answer one, remove one — or list them.

    Without --standing a direction is for the next round only. The next prepare-round
    hands it to the proposer, who answers it once with --reply.
    """
    from researchforge.kaggle import steering

    d = _artifacts(competition)
    try:
        if reply_to:
            hit = steering.reply(d, reply_to, text)
            typer.secho(f"✅ Replied to {hit.id}.", fg=typer.colors.GREEN)
            return
        if remove:
            hit = steering.remove(d, remove)
            typer.secho(f"✅ Removed {hit.id}: {hit.text}", fg=typer.colors.GREEN)
            return
        if text.strip():
            hit = steering.add(d, text, "standing" if standing else "next", via="cli")
            when = "every round until removed" if standing else "the next round"
            typer.secho(f"✅ {hit.id} added for {when}.", fg=typer.colors.GREEN)
            return
    except (KeyError, ValueError) as exc:
        typer.secho(f"❌ {str(exc).strip(chr(39))}", fg=typer.colors.RED, err=True)
        raise typer.Exit(1) from None
    items = steering.load(d)
    if as_json:
        typer.echo(json.dumps([asdict(i) for i in items], indent=2))
        return
    if not items:
        typer.echo("No directions yet.")
    for i in items:
        typer.echo(f"{i.id} [{i.kind}, {i.status}] {i.text}")
        for r in i.replies:
            typer.echo(f"    ↳ {r['by']}: {r['text']}")


@kaggle_app.command("skip-idea")
def skip_idea_cmd(
    competition: SlugArg,
    idea: Annotated[str, typer.Argument(help="The pool idea, as prepare-round shows it")],
    reason: Annotated[str, typer.Option("--reason", help="Why it cannot run")] = "",
) -> None:
    """Take an idea out of the pool's ranking (it stays listed as skipped, with the reason)."""
    from researchforge.kaggle.hypothesis_pool import skip_idea

    d = _artifacts(competition)
    if not reason.strip():
        typer.secho("❌ Say why with --reason.", fg=typer.colors.RED, err=True)
        raise typer.Exit(1)
    hit = skip_idea(d, idea, reason.strip())
    if hit is None:
        typer.secho(f"❌ No untested pool idea matches: {idea!r}", fg=typer.colors.RED, err=True)
        raise typer.Exit(1)
    typer.secho(f"✅ Skipped: {hit}", fg=typer.colors.GREEN)


@kaggle_app.command("settings")
def settings_cmd(
    competition: SlugArg,
    notify_level: Annotated[
        str | None,
        typer.Option(
            "--notify-level", help="verbose = every result · balanced · minimal = candidates only"
        ),
    ] = None,
    submission_policy: Annotated[
        str | None,
        typer.Option(
            "--submission-policy",
            help=(
                "aggressive = any improved round · balanced = an improved round that left a "
                "notebook or file to submit · conservative = only when asked"
            ),
        ),
    ] = None,
    kaggle_gpu_daily_cap: Annotated[
        int | None,
        typer.Option(
            "--kaggle-gpu-daily-cap", help="Max Kaggle GPU experiments per day (0 = no cap)"
        ),
    ] = None,
    metric_direction: Annotated[
        str | None,
        typer.Option(
            "--metric-direction",
            help="maximize (higher is better) or minimize (lower is better), if setup got it wrong",
        ),
    ] = None,
    ntfy_topic: Annotated[
        str | None,
        typer.Option(
            "--ntfy-topic",
            help="This competition's ntfy topic for phone alerts ('' = use the machine-wide one)",
        ),
    ] = None,
) -> None:
    """Show or change notification level, submission policy, the Kaggle GPU cap, the
    phone-alert topic and which way the metric improves."""
    d = _artifacts(competition)
    if metric_direction is not None:
        _set_metric_direction(d, metric_direction)
    if ntfy_topic is not None:
        from researchforge.kaggle.run_config import set_ntfy_topic

        set_ntfy_topic(d, ntfy_topic)
    from researchforge.kaggle.run_config import set_control

    for field, value in (
        ("notification_level", notify_level),
        ("submission_policy", submission_policy),
        ("kaggle_gpu_daily_cap", kaggle_gpu_daily_cap),
    ):
        if value is None:
            continue
        try:
            set_control(d, field, value)
        except ValueError as exc:
            typer.secho(f"❌ {exc}", fg=typer.colors.RED, err=True)
            raise typer.Exit(1) from None
    ctl = control.LoopControl.load(d)
    typer.echo(
        f"notification_level={ctl.notification_level}  submission_policy={ctl.submission_policy}  "
        f"kaggle_gpu_daily_cap={ctl.kaggle_gpu_daily_cap or 'none'}  "
        f"metric_direction={_setup_direction(d)}"
    )
    topic, where = control.notify_topic(d)
    typer.echo(f"phone alerts: {f'ntfy topic {topic} ({where})' if topic else 'off (no topic)'}")


def _set_metric_direction(d: Path, direction: str) -> None:
    """Point every reader of the direction the same way (see run_config)."""
    from researchforge.kaggle.experiment_loop import load_rounds
    from researchforge.kaggle.run_config import set_metric_direction

    try:
        before, direction = set_metric_direction(d, direction)
    except ValueError as exc:
        typer.secho(f"❌ --{exc}", fg=typer.colors.RED, err=True)
        raise typer.Exit(1) from None
    typer.secho(f"✅ Metric direction: {before} → {direction}", fg=typer.colors.GREEN)
    done = len(load_rounds(d / "experiment_log.db"))
    if before != direction and done:
        typer.secho(
            f"⚠️  {done} finished round(s) were judged {before}. Their recorded scores and "
            "improved flags are unchanged; if a script reported a flipped score to work "
            "around the old direction, change it to print the real score.",
            fg=typer.colors.YELLOW,
        )


@kaggle_app.command("auto-run")
def auto_run_cmd(
    competition: SlugArg,
    on: Annotated[
        bool | None,
        typer.Option("--on/--off", help="Turn the auto_run_sensor on or off for this project"),
    ] = None,
    driver: Annotated[
        str | None,
        typer.Option(
            "--driver",
            help="claude (Claude Code), cursor (Cursor agent), byok (API key), or auto: "
            "match how this project runs (the default the first time)",
        ),
    ] = None,
    rounds: Annotated[int | None, typer.Option("--rounds", help="Rounds per launch (1-20)")] = None,
    daily_max: Annotated[
        int | None, typer.Option("--daily-max", help="Most rounds per day, all launches (1-100)")
    ] = None,
    max_minutes: Annotated[
        int | None, typer.Option("--max-minutes", help="Stop a launch after this long")
    ] = None,
    model: Annotated[
        str | None, typer.Option("--model", help="Model for the claude / cursor driver")
    ] = None,
    cli_path: Annotated[
        str | None, typer.Option("--cli-path", help="Path to the claude / cursor-agent binary")
    ] = None,
    stop: Annotated[
        bool, typer.Option("--stop", help="Stop the launch that is running now")
    ] = False,
    now: Annotated[
        bool, typer.Option("--now", help="Check the gates and start a launch now")
    ] = False,
    dry_run: Annotated[
        bool, typer.Option("--dry-run", help="Show the gates and the exact command; start nothing")
    ] = False,
) -> None:
    """Run experiments automatically when the pool has untested ideas.

    Each monitor cycle the auto_run_sensor starts up to --rounds rounds with the
    chosen driver, unless the loop is paused, a question is open, a round is
    running, the pool is empty or today's limit is reached. It never submits.
    Keys for unattended runs go in ~/.config/researchforge/auto_run.env (mode 600).
    """
    from researchforge.kaggle import auto_run as ar
    from researchforge.kaggle import sensors as _sensors

    d = _artifacts(competition)
    slug = OvernightOrchestrator._parse_slug(competition)
    cfg = ar.AutoRunConfig.load(d)
    suggested, why = ar.suggested_driver(d)
    if driver == "auto":
        driver = suggested
    if driver is not None:
        if driver not in ar.DRIVERS:
            typer.secho(
                f"❌ --driver must be one of {', '.join(ar.DRIVERS)}", fg=typer.colors.RED, err=True
            )
            raise typer.Exit(1)
        if driver != cfg.driver:
            cfg = ar.set_driver(d, driver)  # keeps the setup file in step
    for name, value in (
        ("rounds_per_launch", rounds),
        ("max_rounds_per_day", daily_max),
        ("max_minutes", max_minutes),
        ("model", model),
        ("cli_path", cli_path),
    ):
        if value is not None:
            setattr(cfg, name, value)
    changed = any(v is not None for v in (driver, rounds, daily_max, max_minutes, model, cli_path))
    changed = changed or (on is True and not (d / ar.CONFIG_FILE).exists())  # keep the detected one
    if changed:
        cfg.save(d)
        cfg = ar.AutoRunConfig.load(d)  # clamped
    if on is not None:
        _sensors.set_enabled(d, slug, ar.SENSOR, on)
    enabled = _sensors.is_enabled(_sensors.load_config(d, slug), ar.SENSOR)

    if stop:
        stopped = ar.stop_launch(d)
        if stopped is None:
            typer.echo("No launch is running.")
        else:
            typer.secho(
                f"✅ Asked the {stopped['driver']} launch (pid {stopped['pid']}) to stop; it "
                "records what it finished.",
                fg=typer.colors.GREEN,
            )
        return
    ready = ar.readiness(cfg)
    blockers = ar.gates(d, cfg)
    running = ar.current_launch(d)
    last = ar.load_state(d).get("last") or {}
    typer.echo(
        f"Auto-run: {'ON' if enabled else 'off'} · driver {cfg.driver} · {cfg.rounds_per_launch} "
        f"round(s) per launch · at most {cfg.max_rounds_per_day}/day · stop after "
        f"{cfg.max_minutes} min"
    )
    match = (
        "matches how this project runs" if cfg.driver == suggested else f"suggested: {suggested}"
    )
    typer.echo(f"  Why:     {match} — {why}")
    typer.echo(f"  Driver:  {'ready' if ready.ready else 'NOT READY'} — {ready.detail}")
    if ready.fix:
        typer.echo(f"           fix: {ready.fix}")
    if running:
        typer.echo(
            f"  Running: {running['driver']} since {running['started_at']} (pid {running['pid']})"
        )
    if last:
        err = f" — {last['error']}" if last.get("error") else ""
        typer.echo(
            f"  Last:    {last.get('status')} at {last.get('finished_at', '?')}, "
            f"{last.get('rounds_run', 0)} round(s){err}"
        )
    typer.echo(
        "  Gates:   " + ("all clear" if not blockers else "waiting — " + "; ".join(blockers))
    )
    if enabled and not now and not dry_run:
        typer.echo(
            "  Starts on the next monitor cycle (or now: "
            f"researchforge kaggle auto-run {slug} --now)."
        )
    if dry_run:
        cmd = ar.build_command(d, slug, cfg)
        shown = [a if len(a) < 120 else a[:117] + "..." for a in cmd]
        typer.echo("  Command: " + " ".join(f'"{a}"' if " " in a else a for a in shown))
        typer.echo(f"  In:      {d.parent}")
        return
    if now:
        if running:
            typer.echo("  A launch is already running.")
            return
        if blockers or not ready.ready:
            typer.secho(
                "❌ Not starting: " + "; ".join(blockers or [ready.detail]),
                fg=typer.colors.RED,
                err=True,
            )
            raise typer.Exit(1)
        rec = ar.launch(d, slug, cfg)
        typer.secho(
            f"✅ Started {cfg.driver} for up to {cfg.rounds_per_launch} round(s), "
            f"pid {rec['pid']}. Log: {d / ar.LOG_FILE}",
            fg=typer.colors.GREEN,
        )


@kaggle_app.command("auto-run-key")
def auto_run_key_cmd(
    name: Annotated[
        str, typer.Argument(help="CLAUDE_CODE_OAUTH_TOKEN, CURSOR_API_KEY or an AI API key")
    ],
) -> None:
    """Store a token or API key for unattended auto-run, typed at a hidden prompt.

    Run it yourself in a terminal. The value goes to ~/.config/researchforge/auto_run.env
    (mode 600) and is passed only to the auto-run driver. It is never printed.
    """
    import sys as _sys

    from researchforge.kaggle import auto_run as ar

    if name not in ar.STORABLE_KEYS:
        typer.secho(f"❌ Use one of: {', '.join(ar.STORABLE_KEYS)}", fg=typer.colors.RED, err=True)
        raise typer.Exit(1)
    if not _sys.stdin.isatty():
        typer.secho(
            "❌ Run this in your own terminal: the value is typed at a hidden prompt, "
            "never passed by another program.",
            fg=typer.colors.RED,
            err=True,
        )
        raise typer.Exit(1)
    value = typer.prompt(name, hide_input=True).strip()
    path = ar.store_key(name, value)
    typer.secho(f"✅ Saved {name} to {path} (mode 600).", fg=typer.colors.GREEN)


_DRIVER_NAMES = {"claude": "Claude Code", "cursor": "Cursor agent", "byok": "API key"}


@kaggle_app.command("driver")
def driver_cmd(
    competition: SlugArg,
    driver: Annotated[
        str | None,
        typer.Argument(help="claude, cursor or byok — omit to see the current one"),
    ] = None,
    model: Annotated[str | None, typer.Option("--model", help="Model for this driver")] = None,
    sensors: Annotated[
        bool | None,
        typer.Option(
            "--sensors/--no-sensors",
            help="Route the sensors' AI calls (notebook ideas, arXiv, re-synthesis) "
            "through this driver",
        ),
    ] = None,
) -> None:
    """Show or switch the project's driver: who runs rounds and sensors unattended.

    Switching takes effect at the next auto-run launch and sensor call. A launch
    already running finishes on its driver (stop it with `kaggle auto-run --stop`).
    """
    from researchforge.kaggle import auto_run as ar

    d = _artifacts(competition)
    cfg = ar.AutoRunConfig.load(d)
    if driver is not None or model is not None or sensors is not None:
        try:
            cfg = ar.set_driver(d, driver or cfg.driver, model=model, route_sensors=sensors)
        except ValueError as exc:
            typer.secho(f"❌ {exc}", fg=typer.colors.RED, err=True)
            raise typer.Exit(1) from None
        typer.secho(f"✅ Driver: {_DRIVER_NAMES[cfg.driver]}", fg=typer.colors.GREEN)
    ready = ar.readiness(cfg)
    running = ar.current_launch(d)
    typer.echo(
        f"Driver: {_DRIVER_NAMES[cfg.driver]}{' · model ' + cfg.model if cfg.model else ''} · "
        f"sensors {'use it' if cfg.route_sensors and cfg.driver != 'byok' else 'use an API key'}"
    )
    typer.echo(f"  {'✅ ready' if ready.ready else '⚠️  not ready'} — {ready.detail}")
    typer.echo("  Setup:")
    for step in ar.setup_steps(cfg.driver):
        typer.echo(f"    • {step}")
    if running and running.get("driver") != cfg.driver:
        typer.echo(
            f"  A {running['driver']} launch is still running; the next one uses "
            f"{_DRIVER_NAMES[cfg.driver]}."
        )


_BYOK_PROVIDERS = {
    "anthropic": "ANTHROPIC_API_KEY",
    "gemini": "GEMINI_API_KEY",
    "openai": "OPENAI_API_KEY",
}


@kaggle_app.command("driver-login")
def driver_login_cmd(
    driver: Annotated[str, typer.Argument(help="claude, cursor, byok or colab")],
    provider: Annotated[
        str | None,
        typer.Option("--provider", help="byok only: anthropic, gemini or openai"),
    ] = None,
) -> None:
    """Sign a driver in for unattended runs — one command, run in your own terminal.

    claude: runs Claude Code's `setup-token` (a browser opens once), then asks you
    to paste the token it prints. cursor: runs `cursor-agent login`. byok: asks for
    your API key. Tokens and keys are saved to ~/.config/researchforge/auto_run.env
    (mode 600) and never printed.
    """
    import subprocess as _sp
    import sys as _sys

    from researchforge.kaggle import auto_run as ar

    if driver not in (*ar.DRIVERS, "colab"):
        typer.secho(f"❌ Use one of: {', '.join(ar.DRIVERS)}, colab", fg=typer.colors.RED, err=True)
        raise typer.Exit(1)
    if not _sys.stdin.isatty():
        typer.secho(
            "❌ Run this in your own terminal: it signs you in and asks for a token at a "
            "hidden prompt.",
            fg=typer.colors.RED,
            err=True,
        )
        raise typer.Exit(1)

    if driver == "colab":
        from researchforge.kaggle import colab

        if not colab.find_cli():
            if not typer.confirm(
                "Google's Colab CLI is not installed. Install it now "
                "(uv tool install google-colab-cli)?"
            ):
                raise typer.Exit(1)
            ok, last = colab.install()
            if not ok:
                typer.secho(f"❌ Install failed: {last}", fg=typer.colors.RED)
                raise typer.Exit(1)
        typer.echo("Signing in to Colab — open the link it prints, approve, and paste the code.\n")
        _sp.run([colab.find_cli(), "whoami"], check=False)
        st = colab.status()
        if st["signed_in"]:
            typer.secho(
                f"✅ Colab: {st['detail']}. Rounds can run on Colab unattended.",
                fg=typer.colors.GREEN,
            )
            return
        typer.secho(f"⚠️  Still not signed in: {st['detail']}", fg=typer.colors.YELLOW)
        raise typer.Exit(1)

    if driver == "byok":
        name = provider or typer.prompt("Provider (anthropic, gemini, openai)", default="anthropic")
        key_name = _BYOK_PROVIDERS.get(name.strip().lower())
        if key_name is None:
            typer.secho("❌ Provider must be anthropic, gemini or openai", fg=typer.colors.RED)
            raise typer.Exit(1)
        ar.store_key(key_name, typer.prompt(f"{key_name}", hide_input=True).strip())
    else:
        binary = ar.find_binary(driver)
        if not binary:
            install = (
                "npm i -g @anthropic-ai/claude-code"
                if driver == "claude"
                else "curl https://cursor.com/install -fsS | bash"
            )
            typer.secho(f"❌ Not installed. Install it first: {install}", fg=typer.colors.RED)
            raise typer.Exit(1)
        if driver == "claude":
            typer.echo("Running Claude Code's setup-token — finish the sign-in in your browser.\n")
            if _sp.run([binary, "setup-token"], check=False).returncode != 0:
                typer.secho("❌ setup-token did not finish.", fg=typer.colors.RED)
                raise typer.Exit(1)
            token = typer.prompt(
                "\nPaste the token it printed (starts with sk-ant-)", hide_input=True
            ).strip()
            ar.store_key("CLAUDE_CODE_OAUTH_TOKEN", token)
        else:
            typer.echo("Running cursor-agent login — finish the sign-in in your browser.\n")
            if _sp.run([binary, "login"], check=False).returncode != 0:
                typer.secho("❌ cursor-agent login did not finish.", fg=typer.colors.RED)
                raise typer.Exit(1)

    ready = ar.readiness(ar.AutoRunConfig(driver=driver))
    if ready.ready:
        typer.secho(
            f"✅ {_DRIVER_NAMES[driver]} is ready for unattended runs.", fg=typer.colors.GREEN
        )
    else:
        typer.secho(f"⚠️  Still not ready: {ready.detail}. {ready.fix}", fg=typer.colors.YELLOW)
        raise typer.Exit(1)


@kaggle_app.command("pull-kernel")
def pull_kernel_cmd(
    competition: SlugArg,
    ref: Annotated[str, typer.Argument(help="Public notebook, owner/notebook-slug")],
    outputs: Annotated[
        bool,
        typer.Option("--outputs/--no-outputs", help="Also download the notebook's output files"),
    ] = True,
    datasets: Annotated[
        bool,
        typer.Option(
            "--datasets", help="Also download the datasets it attaches (if the data policy allows)"
        ),
    ] = False,
) -> None:
    """Download a public notebook into reference_kernels/<name>/ to read or port from.

    Gets the source and kernel-metadata.json, and by default its output files. The
    datasets it attaches are listed so a proposal can attach them to a Kaggle GPU
    run (`dataset_sources`); --datasets downloads them here when the data policy
    allows local copies.
    """
    import subprocess as _sp

    from researchforge.kaggle.api import KaggleAPIError, pull_kernel_source
    from researchforge.kaggle.compute_policy import ComputePolicy

    d = _artifacts(competition)
    if ref.count("/") != 1 or not all(ref.split("/")):
        typer.secho("❌ Give the notebook as owner/notebook-slug", fg=typer.colors.RED, err=True)
        raise typer.Exit(1)
    dest = d.parent / "reference_kernels" / ref.split("/")[1]
    try:
        pull_kernel_source(ref, dest)
    except KaggleAPIError as exc:
        typer.secho(f"❌ {exc}", fg=typer.colors.RED, err=True)
        raise typer.Exit(1) from None
    meta = read_json(dest / "kernel-metadata.json", {}) or {}
    files = sorted(p.name for p in dest.iterdir() if p.is_file())
    typer.secho(f"✅ {ref} → {dest}", fg=typer.colors.GREEN)
    typer.echo(f"   Files: {', '.join(files)}")
    if outputs:
        out_dir = dest / "output"
        out_dir.mkdir(exist_ok=True)
        res = _sp.run(
            ["kaggle", "kernels", "output", ref, "-p", str(out_dir)],
            capture_output=True,
            text=True,
            timeout=900,
            check=False,
        )
        got = sorted(p.name for p in out_dir.iterdir())
        if res.returncode == 0:
            typer.echo(
                f"   Outputs ({len(got)}): {', '.join(got[:10])}{' …' if len(got) > 10 else ''}"
            )
        else:
            typer.echo(f"   Outputs: not downloaded — {(res.stderr or res.stdout).strip()[:200]}")
    from researchforge.kaggle.api import kernel_inputs

    found = kernel_inputs(meta)
    attached = list(found.get("dataset_sources", []))
    for key, label in _INPUT_LABELS:
        if found.get(key):
            value = found[key]
            typer.echo(f"   {label}: {', '.join(value) if isinstance(value, list) else value}")
    if found:
        typer.echo(
            "   propose-from-kernel attaches all of these to the round; a proposal written by "
            "hand names them with the same keys in proposal.json."
        )
    if datasets and attached:
        policy = ComputePolicy.load(d)
        if policy is not None and not policy.data.can_download_locally:
            typer.echo(
                f"   Datasets not downloaded: the data policy keeps data at "
                f"'{policy.data.location}'. Use dataset_sources on a Kaggle run instead."
            )
        else:
            for ds in attached:
                target = dest / "input" / ds.split("/")[-1]
                res = _sp.run(
                    ["kaggle", "datasets", "download", ds, "-p", str(target), "--unzip"],
                    capture_output=True,
                    text=True,
                    timeout=1800,
                    check=False,
                )
                status = (
                    "downloaded" if res.returncode == 0 else f"failed: {res.stderr.strip()[:120]}"
                )
                typer.echo(f"   Dataset {ds}: {status}")


# A notebook's inputs, as proposal.json and kernel-metadata.json name them.
_INPUT_LABELS = (
    ("dataset_sources", "Datasets attached"),
    ("kernel_sources", "Notebook outputs attached"),
    ("model_sources", "Models attached"),
    ("docker_image", "Docker image"),
    ("machine_shape", "GPU type"),
)


@kaggle_app.command("propose-from-kernel")
def propose_from_kernel_cmd(
    competition: SlugArg,
    ref: Annotated[str, typer.Argument(help="Public notebook, owner/notebook-slug")],
    title: Annotated[str, typer.Option("--title", help="The round's title")],
    description: Annotated[str, typer.Option("--description")] = "",
    rationale: Annotated[str, typer.Option("--rationale")] = "",
    expected_impact: Annotated[str, typer.Option("--expected-impact")] = "",
    hypothesis: Annotated[
        str, typer.Option("--hypothesis", help="The exact pool idea this round tests")
    ] = "",
    parent: Annotated[
        str | None, typer.Option("--parent", help="Parent round, e.g. 3 or R3")
    ] = None,
    branch: Annotated[str, typer.Option("--branch", help="Branch tag")] = "model",
    append: Annotated[
        Path | None,
        typer.Option("--append", help="A .py file appended after the notebook (scoring code)"),
    ] = None,
    dataset: Annotated[
        list[str] | None,
        typer.Option("--dataset", help="Another dataset to attach, owner/slug (repeatable)"),
    ] = None,
) -> None:
    """Write proposal.json from a public notebook, so it runs as a round on the chosen compute.

    Pulls the notebook into reference_kernels/ if it is not there yet, turns its code
    cells into one script (shell and magic lines rewritten for plain Python),
    attaches its datasets, and checks that the script parses. Nothing runs here;
    execute-round runs it next.
    """
    from researchforge.kaggle.api import KaggleAPIError, pull_kernel_source
    from researchforge.kaggle.kernel_port import build_proposal

    d = _artifacts(competition)
    if ref.count("/") != 1 or not all(ref.split("/")):
        typer.secho("❌ Give the notebook as owner/notebook-slug", fg=typer.colors.RED, err=True)
        raise typer.Exit(1)
    if (d / "execute_round.lock").exists():
        typer.secho("❌ A round is running; propose after it ends.", fg=typer.colors.RED, err=True)
        raise typer.Exit(1)
    kernel_dir = d.parent / "reference_kernels" / ref.split("/")[1]
    if (
        not kernel_dir.is_dir()
        or not any(kernel_dir.glob("*.ipynb"))
        and not any(kernel_dir.glob("*.py"))
    ):
        try:
            pull_kernel_source(ref, kernel_dir)
        except KaggleAPIError as exc:
            typer.secho(f"❌ {exc}", fg=typer.colors.RED, err=True)
            raise typer.Exit(1) from None
    try:
        proposal, notes = build_proposal(
            kernel_dir,
            ref,
            title=title,
            description=description,
            rationale=rationale,
            expected_impact=expected_impact,
            hypothesis=hypothesis,
            parent=_round_number(parent) if parent else None,
            branch=branch,
            append=append.read_text(encoding="utf-8") if append else "",
            extra_datasets=dataset or [],
        )
    except (ValueError, OSError) as exc:
        typer.secho(f"❌ {exc}", fg=typer.colors.RED, err=True)
        raise typer.Exit(1) from None
    atomic_write_json(d / "proposal.json", proposal)
    typer.secho(
        f"✅ proposal.json written from {ref} ({len(proposal['code_changes']):,} chars)",
        fg=typer.colors.GREEN,
    )
    for key, label in _INPUT_LABELS:
        if proposal.get(key):
            value = proposal[key]
            typer.echo(f"   {label}: {', '.join(value) if isinstance(value, list) else value}")
    for note in notes:
        typer.echo(f"   Note: {note}")
    typer.echo(f"   Next: researchforge kaggle execute-round {competition}")


@kaggle_app.command("merge-history")
def merge_history_cmd(competition: SlugArg) -> None:
    """Fold the repo project beside this competition into its one history.

    Copies the repo project's experiments that ran into the Kaggle loop's log and
    its untested hypotheses into the pool. The repo project is only read. The
    monitor does the same each cycle; this does it now.
    """
    from researchforge.kaggle.history_import import import_repo_history

    d = _artifacts(competition)
    result = import_repo_history(d, force=True)
    if result is None:
        typer.echo("No repo project beside this competition; nothing to merge.")
        return
    added = result["ideas_added"]
    typer.secho(
        f"✅ {result['experiments']} repo experiment(s) are in this competition's history"
        + (f", {added} untested idea(s) added to the pool." if added else "."),
        fg=typer.colors.GREEN,
    )
    if result["not_run"]:
        typer.echo(f"   {result['not_run']} repo plan(s) that never ran stay in the repo project.")
    if result.get("metric_name"):
        typer.echo(
            f"   Their scores are the repo benchmark's {result['metric_name']}, not comparable "
            "with round scores: shown, never best or a parent."
        )


def _round_number(value: str) -> int:
    try:
        return int(value.lstrip("Rr"))
    except ValueError:
        typer.secho("❌ Give a round number, e.g. 7 or R7", fg=typer.colors.RED, err=True)
        raise typer.Exit(1) from None


@kaggle_app.command("archive")
def archive_cmd(
    competition: SlugArg,
    round_id: Annotated[str, typer.Argument(help="Round number, e.g. 7 or R7")],
    restore: Annotated[
        bool, typer.Option("--restore", help="Bring an archived round back")
    ] = False,
    reason: Annotated[str, typer.Option("--reason", help="Why, for the record")] = "",
) -> None:
    """Archive a round: it leaves the views, is never a parent and never "best". Reversible."""
    from researchforge.kaggle.experiment_loop import load_rounds, set_archived

    d = _artifacts(competition)
    num = _round_number(round_id)
    if num not in {
        r.round_num for r in load_rounds(d / "experiment_log.db", include_archived=True)
    }:
        typer.secho(f"❌ No round {num}", fg=typer.colors.RED, err=True)
        raise typer.Exit(1)
    set_archived(d, num, not restore, reason or "cli")
    typer.secho(f"✅ R{num} {'restored' if restore else 'archived'}.", fg=typer.colors.GREEN)


@kaggle_app.command("delete-round")
def delete_round_cmd(
    competition: SlugArg,
    round_id: Annotated[str, typer.Argument(help="Round number, e.g. 7 or R7")],
    yes: Annotated[bool, typer.Option("--yes", help="Skip the confirmation")] = False,
) -> None:
    """Delete a round from the log for good. Prefer `kaggle archive`, which can be undone."""
    import os as _os

    from researchforge.kaggle.experiment_loop import delete_round

    if _os.environ.get("RESEARCHFORGE_AUTO_RUN") == "1":
        typer.secho("❌ An unattended run may not delete history.", fg=typer.colors.RED, err=True)
        raise typer.Exit(1)
    d = _artifacts(competition)
    num = _round_number(round_id)
    if not yes and not typer.confirm(f"Delete R{num} for good? This cannot be undone"):
        raise typer.Exit(1)
    if not delete_round(d, num):
        typer.secho(f"❌ No round {num}", fg=typer.colors.RED, err=True)
        raise typer.Exit(1)
    typer.secho(f"✅ R{num} deleted.", fg=typer.colors.GREEN)


@kaggle_app.command("colab-setup")
def colab_setup_cmd(
    competition: SlugArg,
    plan: Annotated[
        str | None,
        typer.Option("--plan", help="free, pro, pro_plus, payg or enterprise"),
    ] = None,
    gpu: Annotated[str | None, typer.Option("--gpu", help="T4, L4, G4, A100 or H100")] = None,
    high_mem: Annotated[
        bool | None, typer.Option("--high-mem/--no-high-mem", help="High-RAM machine (Pro/Pro+)")
    ] = None,
    install: Annotated[
        bool, typer.Option("--install", help="Install Google's Colab CLI if it is missing")
    ] = False,
) -> None:
    """Set up Colab for unattended rounds: the plan, the GPU, and Google's Colab CLI.

    Sign in afterwards with `researchforge kaggle driver-login colab` in your own
    terminal, or from the dashboard's Run configs.
    """
    from researchforge.kaggle import colab

    d = _artifacts(competition)
    try:
        cfg = colab.set_config(d, plan=plan, gpu=gpu, high_mem=high_mem)
    except ValueError as exc:
        typer.secho(f"❌ {exc}", fg=typer.colors.RED, err=True)
        raise typer.Exit(1) from None
    if install and not colab.find_cli():
        ok, last = colab.install()
        typer.echo(f"   Colab CLI: {'installed' if ok else 'install failed: ' + last}")
    st = colab.status()
    typer.echo(
        f"Colab: {colab.PLAN_LABELS[cfg.plan]} · GPU {cfg.gpu or '—'}"
        f"{' · high-RAM' if cfg.high_mem else ''}"
    )
    typer.echo(f"  CLI: {st['detail']}" + (f" — {st['fix']}" if st["fix"] else ""))
    if cfg.plan == "enterprise":
        typer.echo("  Colab Enterprise is not supported yet; rounds need regular Colab.")
