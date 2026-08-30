"""`researchforge ship` sub-app."""

from __future__ import annotations

import json
from contextlib import closing
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Annotated
from uuid import uuid4

import typer

from researchforge.execution.worktrees import WorktreeError, WorktreeManager
from researchforge.shipping import push
from researchforge.shipping.branch import ShipBlockedError, prepare_ship, ship_branch
from researchforge.shipping.gh import GhClient, GhError
from researchforge.shipping.push import PushError
from researchforge.storage.db import open_project_db
from researchforge.utils.output import JsonOption, echo_model

if TYPE_CHECKING:
    from researchforge.domain.deliverable import Deliverable
    from researchforge.domain.experiment import Experiment

ship_app = typer.Typer(name="ship", no_args_is_help=True, help="Ship validated findings.")


def _resolve_target(
    remotes: list[tuple[str, str]], *, remote: str | None, url: str | None, yes: bool
) -> push.PushTarget:
    """Which GitHub repository this pull request goes to.

    Asked rather than inferred: `gh` resolves a base repository from remote
    names and prefers `upstream`, which silently targets the project you cloned
    from instead of your own fork.
    """
    if url is not None:
        return push.target_from_url(url)

    targets = push.github_targets(remotes)
    if remote is not None:
        for candidate in targets:
            if candidate.remote == remote:
                return candidate
        known = ", ".join(t.remote or t.url for t in targets) or "none"
        raise PushError(f"No GitHub remote named {remote!r} (configured: {known}).")
    if not targets:
        raise PushError(
            "No GitHub remote configured — add one, or pass --repo-url, before opening a PR."
        )
    if len(targets) == 1 or yes:
        return targets[0]

    typer.echo("Where should this pull request go?")
    for index, candidate in enumerate(targets, start=1):
        typer.echo(f"  {index}. {candidate.remote:<10} {candidate.nwo}")
    typer.echo(f"  {len(targets) + 1}. another repository (paste a GitHub URL)")
    choice = typer.prompt("Choose", default="1").strip()
    if choice == str(len(targets) + 1):
        return push.target_from_url(typer.prompt("GitHub URL"))
    try:
        return targets[int(choice) - 1]
    except (ValueError, IndexError):
        raise PushError(f"{choice!r} is not one of the options.") from None


def _replay_message_file(repo_root: Path, winner: Experiment, deliverable: Deliverable) -> Path:
    """The shipped commit's own message, reused for the replayed commit."""
    ship_dir = repo_root / ".researchforge" / "artifacts" / "ship" / winner.experiment_id
    ship_dir.mkdir(parents=True, exist_ok=True)
    message_file = ship_dir / "commit_message.txt"
    if not message_file.is_file():
        manager = WorktreeManager(repo_root)
        message_file.write_text(
            manager.commit_message(deliverable.commit_sha or ""), encoding="utf-8"
        )
    return message_file


def _describe_push(
    *,
    target: push.PushTarget,
    base_branch: str,
    branch: str,
    head_sha: str,
    replay: push.Replay | None,
    fork_mode: bool,
    winner_files: list[str],
    ahead: int | None,
) -> None:
    """State exactly what will be pushed, where, and how it was built."""
    if fork_mode:
        typer.echo(
            f"You do not have push access to {target.nwo} (an open-source repo?). "
            "Proceeding will: (1) create or reuse a PUBLIC fork under your GitHub "
            f"account, (2) push one commit ({head_sha[:12]}) to {branch!r} on that "
            f"fork, and (3) open a DRAFT pull request on {target.nwo} for human review."
        )
    else:
        typer.echo(
            f"About to push commit {head_sha[:12]} to {branch!r} on {target.nwo} "
            f"and open a DRAFT pull request against {base_branch}."
        )
    if replay is not None:
        typer.echo(
            f"  Contents:  {len(replay.changed_files)} file(s) — "
            f"{', '.join(replay.changed_files)} — replayed onto {base_branch}, "
            "so the diff is the change alone."
        )
        if not replay.clean:
            typer.echo(
                f"  WARNING:   {base_branch} holds a different version of "
                f"{', '.join(replay.diverged_files)} than the benchmark measured. "
                "The change is being applied to code it was not tested against."
            )
    else:
        typer.echo(
            f"  Contents:  the shipped commit on the frozen baseline, changing "
            f"{', '.join(winner_files)}."
        )
        if ahead is not None and ahead > 1:
            typer.echo(
                f"  WARNING:   {branch!r} carries {ahead} commits that are not on "
                f"{base_branch} — e.g. a locally committed benchmark. The PR will "
                "include ALL of them; review `git log` before proceeding."
            )


def _provenance_note(
    replay: push.Replay | None, deliverable: Deliverable, base_branch: str, nwo: str
) -> str | None:
    """How the pushed commit relates to the tree that was actually measured."""
    if replay is None:
        return None
    baseline = str(deliverable.details.get("baseline_commit", ""))[:12]
    note = (
        f"Measured on baseline commit `{baseline}` and replayed onto "
        f"`{base_branch}` of `{nwo}`, so this diff carries the change alone "
        "rather than the benchmark scaffolding it was measured with."
    )
    if not replay.clean:
        note += (
            f" Note that `{base_branch}` holds a different version of "
            f"{', '.join(f'`{p}`' for p in replay.diverged_files)} than the "
            "measurement used — re-run the benchmark before merging."
        )
    return note


@ship_app.command("branch")
def branch_command(
    experiment_id: Annotated[
        str | None,
        typer.Argument(help="Experiment to ship (defaults to the unique validated one)."),
    ] = None,
    branch: Annotated[
        str | None, typer.Option("--branch", help="Override the derived branch name.")
    ] = None,
    yes: Annotated[bool, typer.Option("--yes", help="Skip the confirmation prompt.")] = False,
    json_output: JsonOption = False,
) -> None:
    """Reconstruct the validated change as a clean local branch (never pushed)."""
    with closing(open_project_db()) as conn:
        if not yes:
            try:
                ship = prepare_ship(conn, experiment_id)
            except ShipBlockedError as exc:
                typer.echo(str(exc))
                raise typer.Exit(code=1) from None
            minutes = ship.prep.contract.spec.execution.timeout_minutes
            typer.echo(
                f"Shipping {ship.experiment.experiment_id} ({ship.experiment.title}) — "
                f"a pre-ship confirmation will re-run the full benchmark (~{minutes} min "
                "worst case), then a clean local branch is created. Nothing is pushed."
            )
            confirmation = typer.prompt("Type 'ship' to proceed")
            if confirmation.strip().lower() != "ship":
                typer.echo("Not shipped.")
                raise typer.Exit(code=1)

        try:
            result = ship_branch(conn, experiment_id, branch=branch)
        except ShipBlockedError as exc:
            typer.echo(str(exc))
            raise typer.Exit(code=1) from None

    from researchforge.analytics.service import record_event

    record_event("branch_created")
    if json_output:
        echo_model(result)
        return
    typer.echo(f"Branch:    {result.branch}")
    typer.echo(f"Commit:    {result.commit_sha[:12]} (parent {result.baseline_commit[:12]})")
    typer.echo(f"Pre-ship:  primary metric {result.preship_primary_value}")
    typer.echo(f"Changed:   {', '.join(result.changed_files)}")
    typer.echo(
        "Branch created locally. Nothing was pushed. Push with `researchforge ship pr` "
        "(draft, opt-in) or `git push` yourself."
    )
    typer.echo(f"Next: {result.next_action}")


@ship_app.command("pr")
def pr_command(
    experiment_id: Annotated[
        str | None,
        typer.Argument(help="Shipped experiment (defaults to the latest shipped branch)."),
    ] = None,
    base: Annotated[str | None, typer.Option("--base", help="PR base branch.")] = None,
    remote: Annotated[
        str | None,
        typer.Option("--remote", help="Push to this git remote instead of asking."),
    ] = None,
    repo_url: Annotated[
        str | None,
        typer.Option("--repo-url", help="Push to this GitHub URL instead of asking."),
    ] = None,
    as_measured: Annotated[
        bool,
        typer.Option(
            "--as-measured",
            help="Push the shipped commit on the frozen baseline instead of "
            "replaying the change onto the base branch.",
        ),
    ] = False,
    yes: Annotated[bool, typer.Option("--yes", help="Skip the push confirmation.")] = False,
    json_output: JsonOption = False,
) -> None:
    """Push the shipped change and open a DRAFT pull request (opt-in twice)."""
    from researchforge.domain.deliverable import Deliverable, DeliverableKind
    from researchforge.domain.experiment import BenchmarkStage
    from researchforge.execution.validation import summarize_validation
    from researchforge.shipping.pr_body import build_pr_body, build_pr_title
    from researchforge.storage.baseline_repository import get_latest_successful_baseline
    from researchforge.storage.contract_repository import get_active_contract
    from researchforge.storage.deliverable_repository import (
        get_branch_deliverable,
        insert_deliverable,
        list_deliverables,
    )
    from researchforge.storage.experiment_repository import (
        get_experiment,
        list_executions,
        list_experiments,
        list_runs,
    )
    from researchforge.storage.hypothesis_repository import get_hypothesis
    from researchforge.storage.paper_repository import list_papers
    from researchforge.storage.project_repository import get_project

    gh = GhClient()
    with closing(open_project_db()) as conn:
        project = get_project(conn)
        if project is None:
            typer.echo("No project found.")
            raise typer.Exit(code=1)
        repo_root = Path(project.repository.path) if project.repository.path else Path.cwd()

        deliverable = get_branch_deliverable(conn, experiment_id)
        if deliverable is None or deliverable.experiment_id is None:
            typer.echo("No shipped branch found — run `researchforge ship branch` first.")
            raise typer.Exit(code=1)
        branch = deliverable.location
        winner = get_experiment(conn, deliverable.experiment_id)
        assert winner is not None

        contract = get_active_contract(conn)
        if contract is None:
            typer.echo("No approved contract.")
            raise typer.Exit(code=1)
        if not contract.spec.shipping.allow_draft_pr:
            typer.echo(
                "Draft PRs are opt-in — set shipping.allow_draft_pr: true in "
                "researchforge.yaml and re-approve the contract."
            )
            raise typer.Exit(code=1)
        if not gh.available():
            typer.echo("The GitHub CLI (gh) is not installed: https://cli.github.com/")
            raise typer.Exit(code=1)
        if not gh.auth_ok(repo_root):
            typer.echo("gh is not authenticated — run `gh auth login` first.")
            raise typer.Exit(code=1)
        try:
            target = _resolve_target(
                gh.list_remotes(repo_root), remote=remote, url=repo_url, yes=yes
            )
        except (GhError, PushError) as exc:
            typer.echo(str(exc))
            raise typer.Exit(code=1) from None

        try:
            base_branch = base or gh.default_branch(repo_root, target.nwo)
        except GhError as exc:
            typer.echo(str(exc))
            raise typer.Exit(code=1) from None

        fork_mode = not gh.viewer_can_push(repo_root, target.nwo)
        head_sha = deliverable.commit_sha or ""
        replay = None
        if not as_measured:
            manager = WorktreeManager(repo_root)
            try:
                base_sha = manager.fetch(target.source, base_branch)
                replay = push.replay_onto(
                    manager,
                    source_sha=head_sha,
                    baseline_commit=str(deliverable.details.get("baseline_commit", "")),
                    changed_files=list(winner.changed_files),
                    base_sha=base_sha,
                    message_file=_replay_message_file(repo_root, winner, deliverable),
                )
            except (WorktreeError, PushError) as exc:
                typer.echo(f"Could not replay the change onto {base_branch}: {exc}")
                typer.echo("Retry with --as-measured to push the shipped commit instead.")
                raise typer.Exit(code=1) from None
            head_sha = replay.commit_sha

        if fork_mode and json_output:
            typer.echo(
                f"You do not have push access to {target.nwo}, so this would open a "
                "cross-repository pull request from a public fork — that needs an "
                "interactive confirmation. Re-run without --json."
            )
            raise typer.Exit(code=1)

        if not json_output:
            _describe_push(
                target=target,
                base_branch=base_branch,
                branch=branch,
                head_sha=head_sha,
                replay=replay,
                fork_mode=fork_mode,
                winner_files=list(winner.changed_files),
                ahead=None
                if replay is not None
                else gh.commits_ahead(repo_root, branch, f"{target.source}/{base_branch}"),
            )

        # A cross-repository pull request pushes to a repository you do not own
        # and files it on someone else's project: always typed, never --yes.
        if fork_mode:
            confirmation = typer.prompt("Type 'fork' to proceed")
            if confirmation.strip().lower() != "fork":
                typer.echo("Not forked, nothing pushed.")
                raise typer.Exit(code=1)
        elif not yes:
            confirmation = typer.prompt("Type 'push' to proceed")
            if confirmation.strip().lower() != "push":
                typer.echo("Not pushed.")
                raise typer.Exit(code=1)
        upstream = target.nwo

        hypothesis = get_hypothesis(conn, winner.hypothesis_id)
        assert hypothesis is not None
        baseline = get_latest_successful_baseline(conn)
        assert baseline is not None
        experiments = list_experiments(conn, winner.plan_id)
        runs = [r for r in list_runs(conn) if r.plan_id == winner.plan_id]
        executions = list_executions(conn, run_id=runs[-1].run_id) if runs else []
        validation_attempts = [
            e
            for e in executions
            if e.experiment_id == winner.experiment_id
            and e.benchmark_stage is BenchmarkStage.VALIDATION
        ]
        validation = (
            summarize_validation(
                winner,
                validation_attempts,
                baseline,
                contract.spec.objective.primary_metric.direction,
            )
            if validation_attempts
            else None
        )
        preship = validation_attempts[-1] if validation_attempts else None
        reports = list_deliverables(conn, kind=DeliverableKind.ENGINEERING_REPORT)
        report_path = reports[-1].location if reports else None

        body = build_pr_body(
            contract=contract,
            hypothesis=hypothesis,
            papers=list_papers(conn),
            experiments=experiments,
            executions=executions,
            winner=winner,
            baseline=baseline,
            validation=validation,
            preship=preship,
            report_path=report_path,
            provenance=_provenance_note(replay, deliverable, base_branch, target.nwo),
        )
        title = build_pr_title(hypothesis, winner, contract)
        body_dir = repo_root / ".researchforge" / "artifacts" / "ship" / winner.experiment_id
        body_dir.mkdir(parents=True, exist_ok=True)
        body_file = body_dir / "pr_body.md"
        body_file.write_text(body, encoding="utf-8")

        try:
            if fork_mode:
                gh.fork_and_add_remote(repo_root, upstream)
                gh.push_commit(repo_root, head_sha, branch, destination="fork")
                url = gh.create_draft_pr(
                    repo_root,
                    branch=branch,
                    title=title,
                    body_file=body_file,
                    base=base_branch,
                    repo=upstream,
                    head_owner=gh.viewer_login(repo_root),
                )
            else:
                gh.push_commit(repo_root, head_sha, branch, destination=target.source)
                url = gh.create_draft_pr(
                    repo_root,
                    branch=branch,
                    title=title,
                    body_file=body_file,
                    base=base_branch,
                    repo=target.nwo,
                )
        except GhError as exc:
            typer.echo(str(exc))
            raise typer.Exit(code=1) from None

        insert_deliverable(
            conn,
            project.id,
            Deliverable(
                deliverable_id=uuid4().hex,
                kind=DeliverableKind.DRAFT_PR,
                experiment_id=winner.experiment_id,
                location=url,
                commit_sha=head_sha,
                details={
                    "branch": branch,
                    "repository": target.nwo,
                    "base": base_branch,
                    "content": "as_measured" if replay is None else "replayed",
                },
                created_at=datetime.now(UTC),
            ),
        )

    from researchforge.analytics.service import record_event

    record_event("draft_pr_created")
    if json_output:
        typer.echo(
            json.dumps(
                {
                    "url": url,
                    "branch": branch,
                    "draft": True,
                    "repository": target.nwo,
                    "base": base_branch,
                    "commit": head_sha,
                    "content": "as_measured" if replay is None else "replayed",
                },
                indent=2,
            )
        )
    else:
        typer.echo(f"Draft PR opened: {url}")
