"""Orchestrator for the full `/overnight` flow.

Coordinates:
  1. Credential validation
  2. Competition metadata reading (Kaggle page + AI extraction)
  3. GPU detection
  4. Human-in-the-loop questions
  5. Data download
  6. Baseline measurement
  7. Idea source collection (arXiv + kernels)
  8. Contract generation
  9. User approval
 10. Unattended experiment loop
 11. Morning report + notification
"""

from __future__ import annotations

import contextlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from researchforge.domain.contract import ContractSpec
from researchforge.kaggle.api import download_competition_data
from researchforge.kaggle.auto_run import detect_host
from researchforge.kaggle.competition_reader import CompetitionMetadata, read_competition_metadata
from researchforge.kaggle.contract_generator import generate_contract_for_kaggle
from researchforge.kaggle.credentials import (
    get_kaggle_credentials,
    setup_credentials_interactive,
    validate_credentials,
)
from researchforge.kaggle.gpu_detector import (
    COMPUTE_CHOICES,
    GPUBackend,
    GPUDetection,
    ask_user_gpu_preference,
    detect_gpu_from_competition,
    resolve_compute_target,
)
from researchforge.kaggle.human_questions import (
    UserAnswers,
    ask_questions_interactive,
    generate_setup_questions,
)
from researchforge.kaggle.idea_sources import IdeaCollection, fetch_sources
from researchforge.kaggle.notebook_baseline import (
    BaselineResult,
    find_best_public_kernel,
    run_baseline_on_kernel,
)

_DIRECTION_SOURCES = {
    "leaderboard": "from the leaderboard order",
    "ai": "read from the evaluation text",
    "keyword": "from the metric name",
    "guess": "a guess",
    "user": "set by you",
}


def _direction_label(meta: CompetitionMetadata) -> str:
    better = "lower is better" if meta.metric_direction == "minimize" else "higher is better"
    source = _DIRECTION_SOURCES.get(meta.metric_direction_source, "")
    return f"{better}, {source}" if source else better


class OvernightError(Exception):
    """Overnight flow error; message is user-facing."""


@dataclass
class OvernightSetup:
    """Configuration for one overnight run — fully serialisable to JSON."""

    competition_slug: str
    data_dir: Path
    artifacts_dir: Path
    competition_dir: Path | None = None
    selected_gpu: GPUBackend | None = None
    baseline_kernel_ref: str = ""
    baseline_result: BaselineResult | None = None
    contract: ContractSpec | None = None
    competition_metadata: CompetitionMetadata | None = None
    idea_sources: IdeaCollection | None = None
    user_answers: UserAnswers | None = None
    time_budget_hours: int = 8
    # AI provider configuration — governs proposals + resynthesize
    # "claude_code": Claude in the IDE drives the loop (no API key needed)
    # "anthropic" / "gemini" / "openai": headless mode (BYOK, no editor needed —
    #   the machine running the loop must still stay on)
    # "ollama": headless mode on a local model (no key, no token cost)
    ai_provider_mode: str = "claude_code"  # claude_code | anthropic | gemini | openai | ollama
    ai_model: str = ""  # e.g. claude-opus-4-5, gemini-2.5-pro, gpt-4o
    # The agent setup ran inside ("claude" | "cursor" | ""), so auto-run can offer it.
    agent_host: str = ""
    max_experiments: int = 20
    ntfy_topic: str = ""  # ntfy topic for push notifications (from NTFY_TOPIC env)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: Path | None = None) -> None:
        """Serialise to JSON."""
        if path is None:
            path = self.artifacts_dir / "overnight_setup.json"
        path.parent.mkdir(parents=True, exist_ok=True)

        data: dict[str, Any] = {
            "competition_slug": self.competition_slug,
            "data_dir": str(self.data_dir),
            "artifacts_dir": str(self.artifacts_dir),
            "competition_dir": str(self.competition_dir) if self.competition_dir else None,
            "selected_gpu": self.selected_gpu.value if self.selected_gpu else None,
            "baseline_kernel_ref": self.baseline_kernel_ref,
            "time_budget_hours": self.time_budget_hours,
            "max_experiments": self.max_experiments,
            "baseline_result": asdict(self.baseline_result) if self.baseline_result else None,
            "contract": self.contract.model_dump() if self.contract else None,
            "competition_metadata": asdict(self.competition_metadata)
            if self.competition_metadata
            else None,
            "idea_sources": (
                {
                    "competition_slug": self.idea_sources.competition_slug,
                    "sources": [
                        {**asdict(s), "source_type": s.source_type.value}
                        for s in self.idea_sources.sources
                    ],
                }
                if self.idea_sources
                else None
            ),
            "user_answers": asdict(self.user_answers) if self.user_answers else None,
            "ai_provider_mode": self.ai_provider_mode,
            "ai_model": self.ai_model,
            "agent_host": self.agent_host,
            "ntfy_topic": self.ntfy_topic,
        }
        with path.open("w") as fh:
            json.dump(data, fh, indent=2)

    @classmethod
    def load(cls, path: Path) -> OvernightSetup:
        """Deserialise from JSON."""
        with path.open() as fh:
            data = json.load(fh)

        baseline_result = None
        if data.get("baseline_result"):
            baseline_result = BaselineResult(**data["baseline_result"])

        contract = None
        if data.get("contract"):
            contract = ContractSpec(**data["contract"])

        competition_metadata = None
        if data.get("competition_metadata"):
            competition_metadata = CompetitionMetadata(**data["competition_metadata"])

        idea_sources = None
        if data.get("idea_sources"):
            from researchforge.kaggle.idea_sources import IdeaSource, IdeaSourceType

            raw_sources = data["idea_sources"].get("sources", [])
            sources = [
                IdeaSource(
                    source_type=IdeaSourceType(s["source_type"]),
                    title=s["title"],
                    url=s["url"],
                    ideas=s.get("ideas", []),
                    relevance_score=s.get("relevance_score", 0.0),
                    raw_text=s.get("raw_text", ""),
                )
                for s in raw_sources
            ]
            idea_sources = IdeaCollection(
                competition_slug=data["idea_sources"]["competition_slug"],
                sources=sources,
            )

        user_answers = None
        if data.get("user_answers"):
            user_answers = UserAnswers(
                answers=data["user_answers"].get("answers", {}),
                asked_at=data["user_answers"].get("asked_at", ""),
            )

        return cls(
            competition_slug=data["competition_slug"],
            data_dir=Path(data["data_dir"]),
            artifacts_dir=Path(data["artifacts_dir"]),
            competition_dir=Path(data["competition_dir"]) if data.get("competition_dir") else None,
            selected_gpu=GPUBackend(data["selected_gpu"]) if data.get("selected_gpu") else None,
            baseline_kernel_ref=data.get("baseline_kernel_ref", ""),
            baseline_result=baseline_result,
            contract=contract,
            competition_metadata=competition_metadata,
            idea_sources=idea_sources,
            user_answers=user_answers,
            time_budget_hours=data.get("time_budget_hours", 8),
            max_experiments=data.get("max_experiments", 20),
            ai_provider_mode=data.get("ai_provider_mode", "claude_code"),
            ai_model=data.get("ai_model", ""),
            agent_host=data.get("agent_host", ""),
            ntfy_topic=data.get("ntfy_topic", ""),
        )


# ---------------------------------------------------------------------------
# Scheduler setup helper
# ---------------------------------------------------------------------------


def _offer_scheduler_install(slug: str, artifacts_dir: Path) -> None:
    """Ask whether to install the persistent launchd kernel watcher.

    Skipped automatically in non-interactive mode (no TTY) so it never blocks
    Claude Code or CI runs. In interactive mode it asks once and remembers the
    answer in scheduler_config.json so it isn't asked again on re-runs.
    """
    import sys

    if not sys.stdin.isatty():
        return  # non-interactive — skip silently

    # Already installed?
    label = f"com.researchforge.watcher.{slug}"
    from pathlib import Path as _P

    plist = _P.home() / "Library" / "LaunchAgents" / f"{label}.plist"
    if plist.exists():
        return  # already installed — don't ask again

    # Already declined? (stored in scheduler_config.json)
    declined_marker = artifacts_dir / ".scheduler_install_declined"
    if declined_marker.exists():
        return

    print("\n" + "─" * 60)
    print("📡  Background kernel watcher")
    print(
        "   Install a system service that checks for better public kernels\n"
        "   every 6 hours and sends you a push notification — even when\n"
        "   Claude Code is closed."
    )
    print("─" * 60)

    try:
        answer = input("   Install now? [Y/n] ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        return

    if answer in ("", "y", "yes"):
        hours_input = ""
        with contextlib.suppress(EOFError, KeyboardInterrupt):
            hours_input = input("   Poll interval in hours [6]: ").strip()
        try:
            hours = float(hours_input) if hours_input else 6.0
        except ValueError:
            hours = 6.0

        import subprocess
        import sys as _sys

        result = subprocess.run(
            [
                _sys.executable,
                "-m",
                "researchforge",
                "kaggle",
                "scheduler-install",
                slug,
                "--interval-hours",
                str(hours),
            ],
            check=False,
        )
        if result.returncode != 0:
            print("   ⚠️  Install failed. Run manually:")
            print(f"   researchforge kaggle scheduler-install {slug} --interval-hours {int(hours)}")
    else:
        # Mark as declined so we don't ask again
        declined_marker.touch()
        print(f"   Skipped. Run later: researchforge kaggle scheduler-install {slug}")


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


class OvernightOrchestrator:
    """Main coordinator for Kaggle Overnight."""

    def __init__(self, competition_url: str) -> None:
        self.slug = self._parse_slug(competition_url)
        from researchforge.kaggle.workspace import resolve_artifacts_dir

        self.artifacts_dir = resolve_artifacts_dir(self.slug)
        self.data_dir = self.artifacts_dir / "data"

    @staticmethod
    def _parse_slug(url_or_slug: str) -> str:
        if url_or_slug.startswith("http"):
            return url_or_slug.rstrip("/").split("/")[-1]
        return url_or_slug

    def forced_compute(self, gpu_preference: str | None) -> GPUBackend | None:
        """The backend ``--gpu`` names, checked before setup spends any time.

        An unknown name, or a remote target that has not been configured, stops
        setup: silently running somewhere the user did not choose is worse.
        """
        if not gpu_preference or gpu_preference.strip().lower() in ("auto", ""):
            return None
        backend = resolve_compute_target(gpu_preference)
        if backend is None:
            raise OvernightError(
                f"Unknown compute target '{gpu_preference}'. Use one of: {COMPUTE_CHOICES}."
            )
        if backend == GPUBackend.SSH_SERVER:
            from researchforge.kaggle.ssh_config import SSHConfig

            if SSHConfig.load(self.artifacts_dir) is None:
                raise OvernightError(
                    "--gpu ssh needs a server first:\n"
                    f"  researchforge kaggle ssh-setup {self.slug} --host user@server "
                    "--key ~/.ssh/id_ed25519"
                )
        if backend == GPUBackend.CLOUD_GPU:
            from researchforge.kaggle.cloud_gpu_config import CloudGPUConfig

            if CloudGPUConfig.load(self.artifacts_dir) is None:
                raise OvernightError(
                    "--gpu cloud needs a provider first:\n"
                    f"  researchforge kaggle cloud-gpu-setup {self.slug} --provider runpod "
                    "--list-gpus"
                )
        return backend

    def _prefer_facts_recommendation(self, detection: GPUDetection) -> None:
        """Recommend what gather-facts recommended, so both steps agree.

        gather-facts weighs disk space, a local GPU and Kaggle quota; the
        competition-page detector only knows what the page mentions.
        """
        from researchforge.kaggle.facts_gatherer import CompetitionFacts

        facts = CompetitionFacts.load(self.artifacts_dir)
        backend = resolve_compute_target(facts.recommended_compute) if facts else None
        if backend is not None:
            detection.recommended = backend

    def setup(
        self,
        gpu_preference: str | None = None,
        time_budget_hours: int = 8,
        max_experiments: int = 20,
        target_improvement_pct: float = 0.03,
    ) -> OvernightSetup:
        """Execute full setup phase and return an OvernightSetup."""
        print(f"\n🚀 ResearchForge Kaggle Overnight: {self.slug}")
        print("=" * 70)
        forced_gpu = self.forced_compute(gpu_preference)

        # ── 1. Credentials ────────────────────────────────────────────────────
        print("\n1️⃣  Checking Kaggle credentials...")
        if not validate_credentials():
            ok = setup_credentials_interactive()
            if not ok:
                raise OvernightError(
                    "Kaggle credentials are required to continue.\n"
                    "Run `researchforge kaggle overnight` again after setting up your token."
                )
        creds = get_kaggle_credentials()
        print(f"✅ Credentials valid (user: {creds['username']})")

        # ── 2. AI provider (optional — used for richer analysis) ─────────────
        from researchforge.ai.providers import get_provider
        from researchforge.kaggle.token_log import metered

        ai_provider = metered(get_provider(), self.artifacts_dir, "setup")
        if ai_provider:
            print(f"\n2️⃣  AI provider: {ai_provider.name}")
        else:
            print("\n2️⃣  No AI provider configured (set ANTHROPIC_API_KEY etc. for richer analysis)")

        # ── 3. Competition metadata ───────────────────────────────────────────
        print("\n3️⃣  Reading competition metadata...")
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)
        meta = read_competition_metadata(self.slug, self.artifacts_dir, ai_provider=ai_provider)
        print(f"✅ {meta.title or self.slug}")
        print(
            f"   Task: {meta.task_type}  |  Metric: {meta.metric_name} ({_direction_label(meta)})"
        )
        if meta.metric_direction_source == "guess":
            print(
                "   ⚠️  Nothing said which way this metric improves, so that is a guess. Fix with\n"
                f"      researchforge kaggle settings {self.slug} "
                "--metric-direction <maximize|minimize>"
            )

        # ── 4. GPU detection ──────────────────────────────────────────────────
        print("\n4️⃣  Detecting GPU options...")
        detection = detect_gpu_from_competition(self.slug)
        print(f"✅ {detection.explanation}")

        if forced_gpu is not None:
            selected_gpu = forced_gpu
        else:
            self._prefer_facts_recommendation(detection)
            selected_gpu = ask_user_gpu_preference(detection)
        print(f"✅ Using: {selected_gpu.value.replace('_', ' ').title()}")

        # ── 5. HIL questions ──────────────────────────────────────────────────
        cached_answers = UserAnswers.load(self.artifacts_dir)
        if cached_answers.answers:
            print("\n5️⃣  Using pre-supplied domain answers from user_answers.json")
            user_answers = cached_answers
        else:
            questions = generate_setup_questions(meta, ai_provider)
            user_answers = ask_questions_interactive(questions) if questions else UserAnswers()
        user_answers.save(self.artifacts_dir)

        # ── 5b. AI provider mode ───────────────────────────────────────────────
        # Determine how experiments will be proposed:
        #   claude_code → Claude in the IDE drives the loop (no key needed)
        #   anthropic / gemini / openai → headless mode (BYOK, no editor needed)
        #   ollama → headless mode on a local model (no key)
        # Either way the loop runs on this machine, so it must stay on.
        import os

        if ai_provider is not None:
            # Key already set in env — autonomous mode available
            provider_name = ai_provider.name.split("/")[0]  # e.g. "anthropic"
            ai_provider_mode = provider_name
            ai_model_used = ai_provider.name.partition("/")[2]  # keeps "hf.co/org/model"
            print(
                f"\n5b. AI provider: {ai_provider.name} → headless mode (no editor needed"
                f"{', local model, no API key' if provider_name == 'ollama' else ''})\n"
                "   The loop runs on this machine, so keep it on — or run it on an\n"
                "   always-on server over SSH."
            )
        else:
            ai_provider_mode = "claude_code"
            ai_model_used = ""
            print(
                "\n5b. AI provider: Claude Code (you are the AI — no extra key needed)\n"
                "   Keep the editor session and this machine on. Set ANTHROPIC_API_KEY /\n"
                "   GEMINI_API_KEY / OPENAI_API_KEY, or RESEARCHFORGE_LLM=ollama/<model> for a\n"
                "   local model, to run headless without the editor\n"
                "   (on an always-on server if you want your laptop free)."
            )

        # ── 6. Data download (CPU only — Kaggle GPU data lives on Kaggle servers) ──
        print("\n6️⃣  Data setup...")
        self.data_dir.mkdir(parents=True, exist_ok=True)
        if selected_gpu == GPUBackend.CPU:
            ok = download_competition_data(self.slug, self.data_dir, quiet=True)
            print(
                "✅ Data downloaded"
                if ok
                else "⚠️  Data download failed (check competition rules / disk space)"
            )
        elif selected_gpu == GPUBackend.SSH_SERVER:
            from researchforge.kaggle.ssh_config import SSHConfig

            ssh_cfg = SSHConfig.load(self.artifacts_dir)
            if ssh_cfg and ssh_cfg.rsync_data:
                ok = download_competition_data(self.slug, self.data_dir, quiet=True)
                print(
                    "✅ Data downloaded locally (will rsync to SSH server per experiment)"
                    if ok
                    else "⚠️  Data download failed"
                )
            else:
                print(
                    "✅ SSH mode — set --rsync-data in ssh-setup to copy data to server, or ensure "
                    "data is pre-loaded there"
                )
        elif selected_gpu == GPUBackend.CLOUD_GPU:
            from researchforge.kaggle.cloud_gpu_config import CloudGPUConfig

            cloud_cfg = CloudGPUConfig.load(self.artifacts_dir)
            if cloud_cfg and cloud_cfg.rsync_data:
                ok = download_competition_data(self.slug, self.data_dir, quiet=True)
                print(
                    "✅ Data downloaded locally (will rsync to cloud instance per experiment)"
                    if ok
                    else "⚠️  Data download failed"
                )
            else:
                print(
                    "✅ Cloud GPU mode — data will be downloaded on the instance (set --rsync-data "
                    "in cloud-gpu-setup to copy local data)"
                )
        else:
            print("✅ Kaggle GPU mode — data stays on Kaggle servers (no local download)")

        # ── 7. Baseline kernel ────────────────────────────────────────────────
        print("\n7️⃣  Finding best public notebook...")
        kernel = find_best_public_kernel(self.slug)
        if kernel is None:
            raise OvernightError(f"No public kernels found for {self.slug}")
        print(f"✅ Found: {kernel.title} by {kernel.author}")

        print("\n8️⃣  Running baseline kernel to measure CV score...")
        baseline_result = run_baseline_on_kernel(
            kernel_ref=kernel.ref,
            competition_slug=self.slug,
            gpu_backend=selected_gpu.value,
            data_dir=self.data_dir,
            artifacts_dir=self.artifacts_dir,
        )
        if baseline_result.cv_score is None:
            print(f"⚠️  Baseline measurement incomplete: {baseline_result.status}")
        else:
            print(f"✅ Baseline CV score: {baseline_result.cv_score:.4f}")

        # ── 8. Idea sources ───────────────────────────────────────────────────
        print("\n9️⃣  Gathering ideas (arXiv + kernels)...")
        idea_col = fetch_sources(meta, self.artifacts_dir, ai_provider=ai_provider)
        n_ideas = len(idea_col.all_ideas)
        print(f"✅ Collected {n_ideas} ideas from {len(idea_col.sources)} sources")

        # ── 9. Contract generation ────────────────────────────────────────────
        print("\n🔟  Generating experiment contract...")
        contract = generate_contract_for_kaggle(
            metadata=meta,
            baseline_cv_score=baseline_result.cv_score or 0.0,
            artifacts_dir=self.artifacts_dir,
            target_improvement_pct=target_improvement_pct,
        )
        print(
            f"✅ Contract: {contract.objective.primary_metric.name} "
            f"({contract.objective.primary_metric.direction})"
        )
        baseline_display = (
            f"{baseline_result.cv_score:.4f}" if baseline_result.cv_score is not None else "N/A"
        )
        print(
            f"   Baseline: {baseline_display}  Target: "
            f"{contract.objective.primary_metric.target_value:.4f}"
        )

        # ── 10. Approval screen ───────────────────────────────────────────────
        print("\n" + "─" * 60)
        print("✋  Ready for approval:")
        print(f"  Competition : {self.slug}")
        print(f"  GPU         : {selected_gpu.value.replace('_', ' ').title()}")
        print(f"  Baseline    : {kernel.title}")
        if baseline_result.cv_score is not None:
            score_label = (
                "LB score (from title)"
                if baseline_result.status == "success_lb_score"
                else "CV score"
            )
            print(f"  {score_label:<12}: {baseline_result.cv_score:.4f}")
        else:
            print("  Baseline    : score not available — will be set after first experiment run")
        print(f"  Metric      : {contract.objective.primary_metric.name}")
        if baseline_result.cv_score is not None:
            print(
                f"  Target      : {contract.objective.primary_metric.target_value:.4f} "
                f"(+{target_improvement_pct * 100:.0f}%)"
            )
        print(f"  Budget      : {time_budget_hours}h / {max_experiments} experiments")
        print(f"  Ideas ready : {n_ideas}")
        print("─" * 60)

        # ── 11. Background watcher (optional) ────────────────────────────────
        # Ask once whether to install the persistent launchd scheduler.
        # Skipped in non-interactive mode (Claude Code drives the loop).
        _offer_scheduler_install(self.slug, self.artifacts_dir)

        setup = OvernightSetup(
            competition_slug=self.slug,
            data_dir=self.data_dir,
            artifacts_dir=self.artifacts_dir,
            selected_gpu=selected_gpu,
            baseline_kernel_ref=kernel.ref,
            baseline_result=baseline_result,
            contract=contract,
            competition_metadata=meta,
            idea_sources=idea_col,
            user_answers=user_answers,
            time_budget_hours=time_budget_hours,
            max_experiments=max_experiments,
            ai_provider_mode=ai_provider_mode,
            ai_model=ai_model_used,
            agent_host=detect_host()[0],
            # A topic the person gave as a setup answer is where they expect alerts.
            ntfy_topic=os.environ.get("NTFY_TOPIC", "")
            or str((user_answers.answers if user_answers else {}).get("ntfy_topic") or "").strip(),
        )
        setup.save()

        # Register this competition in the global overnight registry so the
        # dashboard can find it regardless of which directory Claude runs from.
        try:
            from researchforge.kaggle.workspace import register_overnight_dir

            register_overnight_dir(self.slug, self.artifacts_dir)
        except Exception:
            pass

        # If NTFY_TOPIC is set in env, persist it to the global config so
        # future Claude Code subprocesses (which inherit no env) can find it.
        if os.environ.get("NTFY_TOPIC", ""):
            try:
                from researchforge.notifications.service import save_notify_config

                save_notify_config(topic=setup.ntfy_topic)
            except Exception:
                pass

        return setup

    def run_full_flow(
        self,
        gpu_preference: str | None = None,
        time_budget_hours: int = 8,
        max_experiments: int = 20,
    ) -> dict[str, Any]:
        """Single entry point: auto-detect the right mode and run end-to-end.

        Decision logic:
          - Code competition OR data > available free disk → ideation mode
          - Otherwise → full execution mode (setup + experiment loop)

        Returns a dict describing what happened and where outputs are.
        """
        import shutil

        from researchforge.ai.providers import get_provider
        from researchforge.kaggle.competition_reader import read_competition_metadata
        from researchforge.kaggle.credentials import (
            setup_credentials_interactive,
            validate_credentials,
        )

        # Silently determine mode before dispatching so setup() can own all printing.
        if not validate_credentials():
            ok = setup_credentials_interactive()
            if not ok:
                raise OvernightError("Kaggle credentials required.")

        from researchforge.kaggle.token_log import metered

        ai_provider = metered(get_provider(), self.artifacts_dir, "setup")
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)
        meta = read_competition_metadata(self.slug, self.artifacts_dir, ai_provider=ai_provider)

        free_gb = shutil.disk_usage(self.artifacts_dir).free / 1e9
        data_too_large = meta.estimated_data_gb > 0 and meta.estimated_data_gb > free_gb * 0.8

        if meta.is_code_competition:
            mode_reason = "code competition (notebook submission required)"
        elif data_too_large:
            mode_reason = (
                f"data ({meta.estimated_data_gb:.0f} GB) exceeds free disk ({free_gb:.0f} GB)"
            )
        else:
            mode_reason = None  # execution mode

        if mode_reason:
            return self._run_ideation_flow(meta, ai_provider, mode_reason)
        else:
            return self._run_execution_flow(
                meta, ai_provider, gpu_preference, time_budget_hours, max_experiments
            )

    def _run_ideation_flow(
        self, meta: Any, ai_provider: Any, mode_reason: str = ""
    ) -> dict[str, Any]:
        """Research-only path: kernels + arXiv → prioritised plan with code."""
        from researchforge.kaggle.idea_sources import fetch_sources
        from researchforge.kaggle.ideation import run_ideation
        from researchforge.kaggle.notebook_baseline import (
            _extract_lb_score,
            find_best_public_kernel,
        )

        print(f"\n🚀 ResearchForge Kaggle Overnight: {self.slug}")
        print("=" * 60)
        if mode_reason:
            print(f"\n⚡ Ideation mode: {mode_reason}")
            print("   Researching competition → generating prioritised plan...")

        print("\n4️⃣  Finding best public kernel...")
        kernel = find_best_public_kernel(self.slug)
        kernel_title = kernel.title if kernel else ""
        if kernel:
            print(f"✅ {kernel_title} by {kernel.author}")

        baseline_score = _extract_lb_score(kernel_title) if kernel_title else None
        baseline_source = "lb_score" if baseline_score else "not_available"
        if baseline_score:
            print(f"✅ Baseline LB score: {baseline_score:.4f}")

        print("\n5️⃣  Gathering ideas (kernels + arXiv)...")
        idea_col = fetch_sources(meta, self.artifacts_dir, ai_provider=ai_provider)
        print(f"✅ {len(idea_col.all_ideas)} ideas from {len(idea_col.sources)} sources")

        print("\n6️⃣  Generating prioritised improvement plan...")
        plan = run_ideation(
            slug=self.slug,
            artifacts_dir=self.artifacts_dir,
            ai_provider=ai_provider,
            metadata=meta,
            idea_col=idea_col,
            baseline_score=baseline_score,
            baseline_source=baseline_source,
            kernel_title=kernel_title,
        )

        md_path = self.artifacts_dir / "ideation_plan.md"
        print(f"\n{'=' * 60}")
        print(f"✅ Done — {len(plan.ideas)} ideas ready")
        if plan.what_we_learned:
            print("\n💡 What we learned:")
            for insight in plan.what_we_learned:
                print(f"   • {insight}")
        print("\n📋 Top ideas:")
        for item in plan.ideas[:5]:
            tag = f" [{item.branch_tag}]" if item.branch_tag else ""
            print(f"   {item.rank}. {item.title}{tag} — {item.expected_impact}")
        print(f"\n   Full plan: {md_path}")

        return {
            "mode": "ideation",
            "competition": self.slug,
            "ideas_count": len(plan.ideas),
            "plan_path": str(md_path),
            "baseline_score": baseline_score,
        }

    def _run_execution_flow(
        self,
        meta: Any,
        ai_provider: Any,
        gpu_preference: Any,
        time_budget_hours: Any,
        max_experiments: Any,
    ) -> dict[str, Any]:
        """Full execution path: setup then prepare-round / execute-round loop."""
        self.setup(
            gpu_preference=gpu_preference,
            time_budget_hours=time_budget_hours,
            max_experiments=max_experiments,
        )

        slug = self.slug

        # Auto-start dashboard daemon so it's always available at localhost:7842
        try:
            from researchforge.kaggle.dashboard import is_running, start_daemon

            running, dash_port = is_running()
            if not running:
                dash_port = start_daemon(cwd=self.artifacts_dir.parent)
        except Exception:
            dash_port = 7842

        print(f"\n{'=' * 60}")
        print("✅ Setup complete. Ready for the experiment loop.")
        print()
        print(f"   📊 Dashboard → http://localhost:{dash_port}")
        print()
        print("   Run these commands in a loop:")
        print()
        print(f"   researchforge kaggle prepare-round {slug}")
        print(f"   # → read context, write .overnight_{slug}/proposal.json")
        print(f"   researchforge kaggle execute-round {slug}")
        print()
        print("   # Repeat until done, then:")
        print(f"   researchforge kaggle finish {slug}")
        if ai_provider is not None:
            print()
            print("   Or let the API model run it unattended (no Claude Code needed):")
            print(f"   researchforge kaggle run {slug}")
        print(f"{'=' * 60}")

        return {
            "mode": "experiment_loop",
            "competition": slug,
            "setup_path": str(self.artifacts_dir / "overnight_setup.json"),
            "next_command": f"researchforge kaggle prepare-round {slug}",
            "dashboard_url": f"http://localhost:{dash_port}",
        }
