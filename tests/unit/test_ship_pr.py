"""`ship pr` tests — FakeProcessRunner, no network or real gh ever."""

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

import researchforge.shipping.cli as shipping_cli
from researchforge.cli import app
from researchforge.shipping.gh import GhClient, ProcessResult


class FakeProcessRunner:
    """Scripted process runner recording every argv."""

    def __init__(
        self,
        pr_url: str = "https://github.com/acme/repo/pull/7",
        permission: str = "WRITE",
        commits_ahead: int = 1,
        login: str = "contributor",
        nwo: str = "acme/repo",
        remotes: tuple[tuple[str, str], ...] = (("origin", "https://github.com/acme/repo.git"),),
    ) -> None:
        self.calls: list[list[str]] = []
        self.pr_url = pr_url
        self.permission = permission
        self.commits_ahead = commits_ahead
        self.login = login
        self.nwo = nwo
        self.remotes = remotes
        self.fork_created = False

    def run(self, argv: list[str], *, cwd: Path, timeout_seconds: float = 60.0) -> ProcessResult:
        self.calls.append(argv)

        def out(text: str) -> ProcessResult:
            return ProcessResult(exit_code=0, stdout=f"{text}\n", stderr="")

        if argv[:3] == ["gh", "pr", "create"]:
            return out(self.pr_url)
        if argv[:3] == ["gh", "repo", "view"]:
            if "viewerPermission" in argv:
                return out(self.permission)
            if "nameWithOwner" in argv:
                return out(self.nwo)
            if "defaultBranchRef" in argv:
                return out("main")
        if argv[:3] == ["gh", "api", "user"]:
            return out(self.login)
        if argv[:3] == ["gh", "repo", "fork"]:
            self.fork_created = True
            return out("created fork")
        if argv[:3] == ["git", "remote", "--verbose"]:
            lines = [
                f"{name}\t{url} ({direction})"
                for name, url in self.remotes
                for direction in ("fetch", "push")
            ]
            return out("\n".join(lines))
        if argv[:3] == ["git", "rev-list", "--count"]:
            return out(str(self.commits_ahead))
        if argv[:3] == ["git", "remote", "get-url"] and argv[3] == "fork":
            code = 0 if self.fork_created else 1
            return ProcessResult(exit_code=code, stdout="", stderr="")
        return ProcessResult(exit_code=0, stdout="", stderr="")


@pytest.fixture
def fake_runner(monkeypatch: pytest.MonkeyPatch) -> FakeProcessRunner:
    runner = FakeProcessRunner()

    def _make_client() -> GhClient:
        client = GhClient(runner=runner)
        client.available = lambda: True  # type: ignore[method-assign]
        return client

    monkeypatch.setattr(shipping_cli, "GhClient", _make_client)
    return runner


def _git(repo: Path, *args: str) -> str:
    import subprocess

    return subprocess.run(
        ["git", "-C", str(repo), *args], capture_output=True, text=True, check=True
    ).stdout.strip()


@pytest.fixture
def shipped_project(
    cli_runner: CliRunner, validated_project: Path, isolated_project_dir: Path, tmp_path: Path
) -> Path:
    """Validated project with the winner shipped as a local branch and
    shipping.allow_draft_pr flipped on (contract v2, plan intact).

    `origin` is a local bare repository holding `main`, so the replay path
    fetches and pushes for real without touching the network. What the CLI
    believes about the *GitHub* identity of that remote comes from the faked
    `git remote --verbose`, which is independent of this URL.
    """
    assert cli_runner.invoke(app, ["ship", "branch", "--yes"]).exit_code == 0
    contract_file = validated_project / "researchforge.yaml"
    contract_file.write_text(
        contract_file.read_text(encoding="utf-8").replace(
            "allow_draft_pr: false", "allow_draft_pr: true"
        ),
        encoding="utf-8",
    )
    assert cli_runner.invoke(app, ["contract", "approve", "--yes"]).exit_code == 0

    origin = tmp_path / "origin.git"
    _git(tmp_path, "init", "--bare", "--initial-branch=main", str(origin))
    _git(validated_project, "remote", "add", "origin", str(origin))
    _git(validated_project, "push", "origin", "HEAD:refs/heads/main")
    return validated_project


class TestShipPr:
    def test_push_and_draft_pr(
        self,
        cli_runner: CliRunner,
        shipped_project: Path,
        isolated_project_dir: Path,
        fake_runner: FakeProcessRunner,
    ) -> None:
        result = cli_runner.invoke(app, ["ship", "pr", "--yes", "--json"])

        assert result.exit_code == 0, result.output
        payload = json.loads(result.output)
        assert payload["url"] == "https://github.com/acme/repo/pull/7"
        assert payload["draft"] is True
        assert payload["repository"] == "acme/repo"
        assert payload["base"] == "main"
        assert payload["content"] == "replayed"

        # Exactly one push, of exactly one commit, to the chosen remote, never --force.
        pushes = [c for c in fake_runner.calls if c[:2] == ["git", "push"]]
        assert len(pushes) == 1
        assert pushes[0][:3] == ["git", "push", "origin"]
        assert pushes[0][3].endswith(":refs/heads/researchforge/caching-improves-f1-cheaply")
        # The PR is always a draft.
        creates = [c for c in fake_runner.calls if c[:3] == ["gh", "pr", "create"]]
        assert len(creates) == 1
        assert "--draft" in creates[0]

        # Body written with the spec-required sections.
        body = (
            shipped_project / ".researchforge" / "artifacts" / "ship" / "exp-001" / "pr_body.md"
        ).read_text(encoding="utf-8")
        for heading in (
            "## Objective",
            "## Current baseline",
            "## Motivating papers",
            "## Experiments attempted",
            "## Rejected approaches",
            "## Validated metrics",
            "## Constraints",
            "## Changed files",
            "## Reproduction",
            "## Risks and limitations",
            "## ResearchForge report",
        ):
            assert heading in body
        assert "exp-002" in body  # rejected approach listed
        assert "no new tests were authored" in body

        # Deliverable recorded.
        from contextlib import closing

        from researchforge.storage.db import open_project_db

        with closing(open_project_db()) as conn:
            rows = conn.execute(
                "SELECT location FROM deliverables WHERE kind = 'draft_pr'"
            ).fetchall()
        assert [r["location"] for r in rows] == ["https://github.com/acme/repo/pull/7"]

    def test_opt_in_gate_blocks_without_contract_flag(
        self,
        cli_runner: CliRunner,
        validated_project: Path,
        isolated_project_dir: Path,
        fake_runner: FakeProcessRunner,
    ) -> None:
        assert cli_runner.invoke(app, ["ship", "branch", "--yes"]).exit_code == 0
        # allow_draft_pr is false in the funnel contract.
        result = cli_runner.invoke(app, ["ship", "pr", "--yes"])

        assert result.exit_code == 1
        assert "allow_draft_pr" in result.output
        assert fake_runner.calls == []  # zero process invocations

    def test_declined_confirmation_pushes_nothing(
        self,
        cli_runner: CliRunner,
        shipped_project: Path,
        isolated_project_dir: Path,
        fake_runner: FakeProcessRunner,
    ) -> None:
        result = cli_runner.invoke(app, ["ship", "pr"], input="no\n")

        assert result.exit_code == 1
        pushes = [c for c in fake_runner.calls if c[:2] == ["git", "push"]]
        assert pushes == []

    def test_without_shipped_branch(
        self,
        cli_runner: CliRunner,
        validated_project: Path,
        isolated_project_dir: Path,
        fake_runner: FakeProcessRunner,
    ) -> None:
        result = cli_runner.invoke(app, ["ship", "pr", "--yes"])

        assert result.exit_code == 1
        assert "ship branch" in result.output
        assert fake_runner.calls == []


class TestChoosingTheDestination:
    """`gh` prefers a remote named `upstream` when resolving a base repository,
    which silently targets the project you cloned from. The destination is
    asked for and pinned instead."""

    @pytest.fixture
    def two_remotes(self, monkeypatch: pytest.MonkeyPatch) -> FakeProcessRunner:
        runner = FakeProcessRunner(
            remotes=(
                ("origin", "https://github.com/me/mine.git"),
                ("upstream", "https://github.com/them/theirs.git"),
            )
        )

        def _make_client() -> GhClient:
            client = GhClient(runner=runner)
            client.available = lambda: True  # type: ignore[method-assign]
            return client

        monkeypatch.setattr(shipping_cli, "GhClient", _make_client)
        return runner

    def test_the_choice_is_offered_and_honoured(
        self,
        cli_runner: CliRunner,
        shipped_project: Path,
        isolated_project_dir: Path,
        two_remotes: FakeProcessRunner,
    ) -> None:
        result = cli_runner.invoke(app, ["ship", "pr"], input="1\npush\n")

        assert result.exit_code == 0, result.output
        assert "Where should this pull request go?" in result.output
        assert "them/theirs" in result.output  # both are listed
        creates = [c for c in two_remotes.calls if c[:3] == ["gh", "pr", "create"]]
        assert creates[0][creates[0].index("--repo") + 1] == "me/mine"

    def test_remote_flag_skips_the_prompt(
        self,
        cli_runner: CliRunner,
        shipped_project: Path,
        isolated_project_dir: Path,
        two_remotes: FakeProcessRunner,
    ) -> None:
        result = cli_runner.invoke(app, ["ship", "pr", "--remote", "origin", "--yes"])

        assert result.exit_code == 0, result.output
        assert "Where should this pull request go?" not in result.output

    def test_an_unknown_remote_is_refused(
        self,
        cli_runner: CliRunner,
        shipped_project: Path,
        isolated_project_dir: Path,
        two_remotes: FakeProcessRunner,
    ) -> None:
        result = cli_runner.invoke(app, ["ship", "pr", "--remote", "nope", "--yes"])

        assert result.exit_code == 1
        assert "No GitHub remote named 'nope'" in result.output
        assert all(c[:2] != ["git", "push"] for c in two_remotes.calls)


class TestForkAwareShipPr:
    @pytest.fixture
    def readonly_runner(self, monkeypatch: pytest.MonkeyPatch) -> FakeProcessRunner:
        """gh viewer has no push access to origin (the open-source case)."""
        runner = FakeProcessRunner(permission="READ")

        def _make_client() -> GhClient:
            client = GhClient(runner=runner)
            client.available = lambda: True  # type: ignore[method-assign]
            return client

        monkeypatch.setattr(shipping_cli, "GhClient", _make_client)
        return runner

    def test_fork_push_and_cross_repo_draft_pr(
        self,
        cli_runner: CliRunner,
        shipped_project: Path,
        isolated_project_dir: Path,
        readonly_runner: FakeProcessRunner,
    ) -> None:
        result = cli_runner.invoke(app, ["ship", "pr"], input="fork\n")

        assert result.exit_code == 0, result.output
        assert "do not have push access to acme/repo" in result.output
        assert "PUBLIC fork" in result.output

        forks = [c for c in readonly_runner.calls if c[:3] == ["gh", "repo", "fork"]]
        assert forks == [["gh", "repo", "fork", "acme/repo", "--remote", "--remote-name", "fork"]]

        pushes = [c for c in readonly_runner.calls if c[:2] == ["git", "push"]]
        assert len(pushes) == 1
        assert pushes[0][:3] == ["git", "push", "fork"]
        assert pushes[0][3].endswith(":refs/heads/researchforge/caching-improves-f1-cheaply")

        creates = [c for c in readonly_runner.calls if c[:3] == ["gh", "pr", "create"]]
        assert len(creates) == 1
        create = creates[0]
        assert "--draft" in create
        assert "--repo" in create and create[create.index("--repo") + 1] == "acme/repo"
        head = create[create.index("--head") + 1]
        assert head == "contributor:researchforge/caching-improves-f1-cheaply"

    def test_declined_fork_consent_does_nothing(
        self,
        cli_runner: CliRunner,
        shipped_project: Path,
        isolated_project_dir: Path,
        readonly_runner: FakeProcessRunner,
    ) -> None:
        result = cli_runner.invoke(app, ["ship", "pr"], input="no\n")

        assert result.exit_code == 1
        assert "Not forked, nothing pushed" in result.output
        assert all(c[:3] != ["gh", "repo", "fork"] for c in readonly_runner.calls)
        assert all(c[:2] != ["git", "push"] for c in readonly_runner.calls)

    def test_extra_commits_warning(
        self,
        cli_runner: CliRunner,
        shipped_project: Path,
        isolated_project_dir: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        runner = FakeProcessRunner(permission="READ", commits_ahead=3)

        def _make_client() -> GhClient:
            client = GhClient(runner=runner)
            client.available = lambda: True  # type: ignore[method-assign]
            return client

        monkeypatch.setattr(shipping_cli, "GhClient", _make_client)
        result = cli_runner.invoke(app, ["ship", "pr", "--as-measured"], input="fork\n")

        assert result.exit_code == 0, result.output
        assert "carries 3 commits" in result.output
        assert "review `git log`" in result.output

    def test_existing_fork_remote_reused(
        self,
        cli_runner: CliRunner,
        shipped_project: Path,
        isolated_project_dir: Path,
        readonly_runner: FakeProcessRunner,
    ) -> None:
        readonly_runner.fork_created = True  # `fork` remote already wired

        result = cli_runner.invoke(app, ["ship", "pr"], input="fork\n")

        assert result.exit_code == 0, result.output
        assert all(c[:3] != ["gh", "repo", "fork"] for c in readonly_runner.calls)

    def test_yes_cannot_authorize_a_cross_repo_pr(
        self,
        cli_runner: CliRunner,
        shipped_project: Path,
        isolated_project_dir: Path,
        readonly_runner: FakeProcessRunner,
    ) -> None:
        """--yes is for unattended runs against your own repository; pushing to
        someone else's project is always typed."""
        result = cli_runner.invoke(app, ["ship", "pr", "--yes"], input="\n")

        assert result.exit_code == 1
        assert "Type 'fork' to proceed" in result.output
        assert all(c[:2] != ["git", "push"] for c in readonly_runner.calls)
