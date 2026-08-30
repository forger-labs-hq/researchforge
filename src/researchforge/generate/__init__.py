"""`researchforge generate` sub-app — AI-powered file generation."""

from __future__ import annotations

from contextlib import closing
from pathlib import Path
from typing import Annotated

import typer

from researchforge.utils.output import JsonOption

generate_app = typer.Typer(
    name="generate",
    no_args_is_help=True,
    help="Generate project files using a built-in AI provider.",
)


@generate_app.command("eval-script")
def eval_script_command(
    provider: Annotated[
        str | None,
        typer.Option(
            "--provider",
            "-p",
            help="AI provider: anthropic|google|openai. Auto-detected from env when omitted.",
        ),
    ] = None,
    model: Annotated[
        str | None,
        typer.Option("--model", "-m", help="Override model name."),
    ] = None,
    existing: Annotated[
        Path | None,
        typer.Option(
            "--existing",
            help=(
                "Point to an eval script you already have. ResearchForge will copy it to "
                "benchmarks/evaluate.py (or use it in place) instead of generating one. "
                "Example: --existing scripts/run_eval.py"
            ),
            exists=True,
            dir_okay=False,
        ),
    ] = None,
    output_dir: Annotated[
        Path | None,
        typer.Option(
            "--output-dir",
            help="Write files here instead of the cwd. Defaults to the project root.",
        ),
    ] = None,
    force: Annotated[bool, typer.Option("--force", help="Overwrite existing files.")] = False,
    json_output: JsonOption = False,
) -> None:
    """Generate benchmarks/evaluate.py and src/config.py using a built-in AI provider.

    If you already have an eval script, point to it with --existing:

      researchforge generate eval-script --existing path/to/my_eval.py

    If you have neither an eval script nor an API key configured, you can skip
    this step and write benchmarks/evaluate.py manually, then set
    `execution.full_command` in researchforge.yaml to point at it.

    Otherwise, set an API key and ResearchForge will generate both files:

      export ANTHROPIC_API_KEY=sk-ant-...   # or GEMINI_API_KEY / OPENAI_API_KEY
      researchforge generate eval-script

    Run this after `researchforge repo scan` and before `researchforge contract generate`.
    """
    import json as _json

    root = output_dir or Path.cwd()
    benchmarks_dir = root / "benchmarks"
    src_dir = root / "src"
    eval_path = benchmarks_dir / "evaluate.py"
    config_path = src_dir / "config.py"
    init_path = src_dir / "__init__.py"

    # ── existing script path ───────────────────────────────────────────────
    if existing is not None:
        benchmarks_dir.mkdir(parents=True, exist_ok=True)
        if eval_path.exists() and not force:
            typer.echo(
                "benchmarks/evaluate.py already exists. "
                "Use --force to overwrite, or skip this step.",
                err=True,
            )
            raise typer.Exit(code=1)
        import shutil

        shutil.copy2(existing, eval_path)
        if json_output:
            typer.echo(_json.dumps({"eval_script": str(eval_path), "source": str(existing)}))
        else:
            typer.echo(f"✓ Copied {existing} → {eval_path}")
            typer.echo("Review the file, set result_file in researchforge.yaml, then:")
            typer.echo("  researchforge contract generate")
            typer.echo("  researchforge contract approve")
        return

    # ── already exists check (no --existing, no --force) ──────────────────
    if eval_path.exists() and not force:
        typer.echo(
            f"benchmarks/evaluate.py already exists at {eval_path}\n"
            f"Options:\n"
            f"  --force           overwrite with AI-generated version\n"
            f"  --existing <path> copy your own script there instead\n"
            f"  (skip this step)  if the script is already correct, just run "
            f"`researchforge contract generate`",
        )
        raise typer.Exit(code=0)

    # ── AI generation ──────────────────────────────────────────────────────
    from researchforge.ai.eval_gen import generate_eval_files
    from researchforge.ai.service import resolve_provider
    from researchforge.storage.db import open_project_db
    from researchforge.storage.project_repository import get_project
    from researchforge.storage.scan_repository import get_latest_scan

    try:
        ai_provider = resolve_provider(provider_hint=provider, model_hint=model)
    except RuntimeError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=1) from None

    if not json_output:
        typer.echo(f"Generating eval script with {ai_provider.name}…")

    with closing(open_project_db()) as conn:
        project = get_project(conn)
        if project is None or project.objective is None:
            typer.echo("No project found. Run `researchforge project create` first.")
            raise typer.Exit(code=1)
        scan = get_latest_scan(conn)
        if scan is None:
            typer.echo("No repository scan. Run `researchforge repo scan .` first.")
            raise typer.Exit(code=1)

    try:
        eval_py, config_py = generate_eval_files(project, scan, ai_provider)
    except ValueError as exc:
        typer.echo(f"Generation failed: {exc}", err=True)
        raise typer.Exit(code=1) from None

    benchmarks_dir.mkdir(parents=True, exist_ok=True)
    src_dir.mkdir(parents=True, exist_ok=True)

    for fpath, content, label in [
        (eval_path, eval_py, "benchmarks/evaluate.py"),
        (config_path, config_py, "src/config.py"),
    ]:
        if fpath.exists() and not force:
            typer.echo(
                f"  {label} already exists — use --force to overwrite.",
                err=True,
            )
            raise typer.Exit(code=1)
        fpath.write_text(content, encoding="utf-8")

    if not init_path.exists():
        init_path.write_text("", encoding="utf-8")

    if json_output:
        typer.echo(_json.dumps({"eval_script": str(eval_path), "config": str(config_path)}))
        return

    typer.echo(f"✓ Wrote {eval_path}")
    typer.echo(f"✓ Wrote {config_path}")

    # ── Auto-commit the generated files ──────────────────────────────────────
    # ResearchForge runs experiments in git worktrees at the baseline commit.
    # If these files are not committed, the worktree won't have them and
    # `baseline run` will fail with "No such file or directory".
    _auto_commit_generated_files(root, [eval_path, config_path, init_path])

    typer.echo("Review the files, then:")
    typer.echo("  researchforge contract generate")
    typer.echo("  researchforge contract approve")
    typer.echo("  researchforge baseline run")


def _auto_commit_generated_files(repo_root: Path, files: list[Path]) -> None:
    """Commit the generated files to git so worktrees can find them.

    ResearchForge experiments run in detached git worktrees at the baseline
    commit. Files that are not in git simply do not exist in those worktrees,
    causing ``baseline run`` to fail with 'No such file or directory'.

    This function stages and commits the given files automatically.  The user
    is notified; the commit can be inspected with ``git log``.
    """
    import subprocess as _sp

    existing = [f for f in files if f.is_file()]
    if not existing:
        return

    try:
        # Check we're inside a git repo
        _sp.run(
            ["git", "rev-parse", "--git-dir"],
            cwd=repo_root,
            capture_output=True,
            check=True,
        )
    except (_sp.CalledProcessError, FileNotFoundError):
        typer.echo(
            "\nwarning: Not inside a git repository. "
            "Commit the generated files manually before running `researchforge baseline run`:\n"
            "  git add benchmarks/evaluate.py src/config.py src/__init__.py\n"
            "  git commit -m 'chore: add ResearchForge benchmark script'"
        )
        return

    # Stage files relative to repo root
    rel_paths = []
    for f in existing:
        try:
            rel_paths.append(str(f.relative_to(repo_root)))
        except ValueError:
            rel_paths.append(str(f))

    try:
        _sp.run(["git", "add"] + rel_paths, cwd=repo_root, capture_output=True, check=True)
    except _sp.CalledProcessError as exc:
        typer.echo(f"\nwarning: git add failed: {exc.stderr.decode(errors='replace').strip()}")
        return

    # Check if there's anything new to commit
    diff = _sp.run(
        ["git", "diff", "--cached", "--name-only"],
        cwd=repo_root,
        capture_output=True,
        text=True,
    )
    if not diff.stdout.strip():
        typer.echo("(Files already in git — no commit needed.)")
        return

    msg = "chore: add ResearchForge benchmark script and config"
    try:
        _sp.run(
            ["git", "commit", "-m", msg],
            cwd=repo_root,
            capture_output=True,
            check=True,
        )
        typer.echo(f"✓ Committed to git: {', '.join(rel_paths)}")
        typer.echo("  (ResearchForge worktrees use this commit as the baseline.)")
    except _sp.CalledProcessError as exc:
        stderr = exc.stderr.decode(errors="replace").strip()
        typer.echo(f"\nwarning: git commit failed: {stderr}")
        typer.echo(
            "Commit the files manually before `researchforge baseline run`:\n"
            f"  git commit -m '{msg}'"
        )


@generate_app.command("dockerfile")
def dockerfile_command(
    provider: Annotated[
        str | None,
        typer.Option(
            "--provider",
            "-p",
            help="AI provider: anthropic|google|openai. Omit to use a minimal heuristic template.",
        ),
    ] = None,
    model: Annotated[
        str | None,
        typer.Option("--model", "-m", help="Override model name."),
    ] = None,
    cuda: Annotated[
        bool,
        typer.Option("--cuda", help="Use an NVIDIA CUDA base image (GPU workloads)."),
    ] = False,
    force: Annotated[
        bool,
        typer.Option("--force", help="Overwrite an existing Dockerfile."),
    ] = False,
    output_dir: Annotated[
        Path | None,
        typer.Option("--output-dir", help="Write Dockerfile here (default: project root)."),
    ] = None,
    json_output: JsonOption = False,
) -> None:
    """Generate a Dockerfile for this project.

    Without an API key: produces a minimal working Dockerfile from the repo scan.
    With an API key: the AI tailors it to your specific dependencies.

    After generating, update researchforge.yaml to use Docker isolation:
      execution:
        mode: docker

    Docker is strongly recommended for public repos you cloned from GitHub.
    """
    from researchforge.ai.dockerfile_gen import (
        build_minimal_dockerfile,
        generate_dockerfile_with_ai,
    )
    from researchforge.storage.db import open_project_db
    from researchforge.storage.project_repository import get_project
    from researchforge.storage.scan_repository import get_latest_scan

    with closing(open_project_db()) as conn:
        project = get_project(conn)
        if project is None:
            typer.echo("No project found. Run `researchforge project create` first.")
            raise typer.Exit(code=1)
        scan = get_latest_scan(conn)
        if scan is None:
            typer.echo("No repository scan. Run `researchforge repo scan .` first.")
            raise typer.Exit(code=1)

    root = output_dir or Path.cwd()
    dockerfile_path = root / "Dockerfile"

    if dockerfile_path.exists() and not force:
        typer.echo(f"Dockerfile already exists at {dockerfile_path}")
        typer.echo("Use --force to overwrite, or use the existing Dockerfile.")
        raise typer.Exit(code=0)

    # Try AI first if provider configured; fall back to heuristic
    content: str | None = None
    used_ai = False

    if provider or any(
        [
            __import__("os").environ.get("ANTHROPIC_API_KEY"),
            __import__("os").environ.get("GEMINI_API_KEY"),
            __import__("os").environ.get("OPENAI_API_KEY"),
        ]
    ):
        try:
            from researchforge.ai.service import resolve_provider

            ai_provider = resolve_provider(provider_hint=provider, model_hint=model)
            if not json_output:
                typer.echo(f"Generating Dockerfile with {ai_provider.name}…")
            content = generate_dockerfile_with_ai(project, scan, ai_provider, cuda=cuda)
            used_ai = True
        except Exception as exc:  # noqa: BLE001
            if not json_output:
                typer.echo(f"AI generation failed ({exc}), using heuristic template…")

    if content is None:
        if not json_output:
            typer.echo("Generating minimal Dockerfile from repo scan…")
        content = build_minimal_dockerfile(scan)

    dockerfile_path.write_text(content, encoding="utf-8")

    if json_output:
        import json

        typer.echo(json.dumps({"dockerfile": str(dockerfile_path), "used_ai": used_ai}))
        return

    typer.echo(f"✓ Wrote {dockerfile_path}")
    typer.echo("")
    typer.echo("Next steps:")
    typer.echo("  1. Review Dockerfile (edit if needed)")
    typer.echo("  2. Update researchforge.yaml → set  execution.mode: docker")
    typer.echo("  3. researchforge contract generate --force  (regenerate with Docker mode)")
    typer.echo("  4. researchforge contract approve")
    typer.echo("  5. researchforge baseline run")
