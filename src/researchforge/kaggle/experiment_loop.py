"""Overnight experiment loop: propose patches, run screening, full benchmark, repeat.

Supports:
  - DAG experiment structure (parent_round_num + branch_tag)
  - Mid-loop human-in-the-loop pauses (via PatchProposal.needs_user_input)
  - Full/screening funnel with configurable time allocation
  - SQLite persistence of every round
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path

from researchforge.domain.contract import MetricDirection
from researchforge.kaggle.benchmark_runner import BenchmarkRunner
from researchforge.kaggle.orchestrator import OvernightSetup
from researchforge.kaggle.patch_proposer import PatchProposer

# ---------------------------------------------------------------------------
# Round-level data models
# ---------------------------------------------------------------------------


@dataclass
class ScreeningResult:
    """Fast validation on a subset of data."""

    passed: bool
    metric_value: float | None
    duration_seconds: float
    error_message: str | None = None


@dataclass
class BenchmarkResult:
    """Full evaluation result."""

    passed: bool
    metric_value: float
    metric_improved: bool
    duration_seconds: float
    error_message: str | None = None
    # What the run produced, for a later submission: kernel_ref / kernel_version for a
    # Kaggle notebook run, submission_file for the file it wrote (see kaggle.submission).
    outputs: dict[str, object] | None = None


@dataclass
class ExperimentRound:
    """Single experiment: patch + screening + full benchmark."""

    round_num: int
    patch_description: str
    patch_code: str
    # DAG fields
    parent_round_num: int | None = None
    branch_tag: str = ""
    source_credit: str = ""
    # The pool idea this round tested ("" if it came from elsewhere)
    hypothesis: str = ""
    # Results
    screening_result: ScreeningResult | None = None
    full_result: BenchmarkResult | None = None
    rejected_reason: str | None = None
    started_at: datetime = field(default_factory=datetime.now)
    completed_at: datetime | None = None

    @property
    def passed_screening(self) -> bool:
        return self.screening_result is not None and self.screening_result.passed

    @property
    def passed_full_benchmark(self) -> bool:
        return self.full_result is not None and self.full_result.passed

    @property
    def improved_over_baseline(self) -> bool:
        return self.full_result is not None and self.full_result.metric_improved

    @property
    def outcome(self) -> str:
        """The pool's word for how this round went: improved | no_improvement | failed."""
        if self.improved_over_baseline:
            return "improved"
        return "no_improvement" if self.passed_full_benchmark else "failed"


LOCK_FILE = "execute_round.lock"


def _pid_alive(pid: object) -> bool:
    import os

    try:
        os.kill(int(str(pid)), 0)
    except (ProcessLookupError, ValueError, TypeError):
        return False
    except PermissionError:
        return True
    return True


def recover_interrupted_round(artifacts_dir: Path) -> int | None:
    """Release a lock whose ``execute-round`` process is gone; returns its round number.

    The process can die without running its cleanup (a shell's time limit, a
    crash, a closed laptop lid). Its lock would then block every later round.
    The round it was running is recorded as interrupted, so history shows what
    happened and the next round takes the next number.
    """
    from researchforge.kaggle.workspace import atomic_write_json, read_json

    lock = artifacts_dir / LOCK_FILE
    if not lock.exists():
        return None
    info = read_json(lock, {}) or {}
    if not isinstance(info, dict) or _pid_alive(info.get("pid")):
        return None
    round_num = info.get("round_num")
    now = datetime.now().isoformat()
    if isinstance(round_num, int):
        db = artifacts_dir / "experiment_log.db"
        with sqlite3.connect(db) as conn:
            known = conn.execute(
                "SELECT 1 FROM experiments WHERE round_num = ?", (round_num,)
            ).fetchone()
            if not known:
                conn.execute(
                    "INSERT INTO experiments (round_num, parent_round_num, branch_tag, "
                    "patch_description, patch_code, rejected_reason, started_at, completed_at) "
                    "VALUES (?,?,?,?,?,?,?,?)",
                    (
                        round_num,
                        info.get("parent_round_num"),
                        info.get("branch_tag") or "",
                        info.get("title") or f"Round {round_num}",
                        "",
                        f"interrupted: process {info.get('pid')} ended during the "
                        f"{info.get('stage') or 'round'} stage",
                        info.get("started_at") or now,
                        now,
                    ),
                )
    inflight = read_json(artifacts_dir / "inflight_rounds.json", {}) or {}
    if isinstance(inflight, dict) and str(round_num) in inflight:
        inflight.pop(str(round_num))
        atomic_write_json(artifacts_dir / "inflight_rounds.json", inflight)
    lock.unlink(missing_ok=True)
    return round_num if isinstance(round_num, int) else -1


ARCHIVE_FILE = "archived_rounds.json"


def archived_rounds(artifacts_dir: Path) -> set[int]:
    """Rounds the person archived: hidden, never a parent, never "best"."""
    from researchforge.kaggle.workspace import read_json

    data = read_json(artifacts_dir / ARCHIVE_FILE, {}) or {}
    out: set[int] = set()
    for k in data if isinstance(data, dict) else []:
        try:
            out.add(int(k))
        except ValueError:
            continue
    return out


def set_archived(artifacts_dir: Path, round_num: int, archived: bool, reason: str = "") -> None:
    """Archive (or restore) one round. Reversible: the row stays in the log."""
    from researchforge.kaggle.workspace import atomic_write_json, read_json

    path = artifacts_dir / ARCHIVE_FILE
    data = read_json(path, {}) or {}
    data = data if isinstance(data, dict) else {}
    if archived:
        data[str(round_num)] = {
            "at": datetime.now().isoformat(timespec="seconds"),
            "reason": reason,
        }
    else:
        data.pop(str(round_num), None)
    atomic_write_json(path, data)


def delete_round(artifacts_dir: Path, round_num: int) -> bool:
    """Remove one round from the log for good (and its archive / driver notes)."""
    from researchforge.kaggle.workspace import atomic_write_json, read_json

    db = artifacts_dir / "experiment_log.db"
    if not db.exists():
        return False
    with sqlite3.connect(db) as conn:
        gone = conn.execute("DELETE FROM experiments WHERE round_num = ?", (round_num,)).rowcount
    for name in (ARCHIVE_FILE, "round_drivers.json"):
        data = read_json(artifacts_dir / name, {}) or {}
        if isinstance(data, dict) and str(round_num) in data:
            data.pop(str(round_num))
            atomic_write_json(artifacts_dir / name, data)
    return bool(gone)


def load_rounds(db_path: Path, *, include_archived: bool = False) -> list[ExperimentRound]:
    """Every round already in ``experiment_log.db``, oldest first ([] if none).

    Whoever drives the loop next — Claude, `kaggle run`, a new chat — starts from
    this history instead of from nothing. Archived rounds are left out unless asked.
    """
    if not db_path.exists():
        return []
    with sqlite3.connect(db_path) as conn:
        try:
            rows = conn.execute(
                "SELECT round_num, patch_description, patch_code, parent_round_num, branch_tag, "
                "source_credit, screening_passed, screening_metric, full_passed, full_metric, "
                "full_improved, rejected_reason FROM experiments ORDER BY round_num"
            ).fetchall()
        except sqlite3.OperationalError:  # no experiments table yet
            return []
    rounds: list[ExperimentRound] = []
    hidden = set() if include_archived else archived_rounds(db_path.parent)
    for row in rows:
        if row[0] in hidden:
            continue
        r = ExperimentRound(
            round_num=row[0],
            patch_description=row[1] or "",
            patch_code=row[2] or "",
            parent_round_num=row[3],
            branch_tag=row[4] or "",
            source_credit=row[5] or "",
            rejected_reason=row[11],
        )
        if row[6] is not None:
            r.screening_result = ScreeningResult(
                passed=bool(row[6]), metric_value=row[7], duration_seconds=0
            )
        if row[8] is not None:
            r.full_result = BenchmarkResult(
                passed=bool(row[8]),
                metric_value=row[9] or 0.0,
                metric_improved=bool(row[10]),
                duration_seconds=0,
            )
        rounds.append(r)
    return rounds


# ---------------------------------------------------------------------------
# Run-level summary
# ---------------------------------------------------------------------------


def _failure_reason(r: ExperimentRound) -> str | None:
    """Why a round failed, so the history shows a reason and not only a 0.0."""
    for res in (r.full_result, r.screening_result):
        if res is not None and not res.passed and res.error_message:
            return res.error_message[:500]
    return None


@dataclass
class RunSummary:
    setup: OvernightSetup
    rounds: list[ExperimentRound]
    started_at: datetime
    ended_at: datetime
    total_duration_seconds: float
    budget_remaining_seconds: float
    experiments_remaining: int

    @property
    def total_rounds_attempted(self) -> int:
        return len(self.rounds)

    @property
    def rounds_passed_screening(self) -> int:
        return sum(1 for r in self.rounds if r.passed_screening)

    @property
    def rounds_passed_full_benchmark(self) -> int:
        return sum(1 for r in self.rounds if r.passed_full_benchmark)

    @property
    def rounds_improved(self) -> int:
        return sum(1 for r in self.rounds if r.improved_over_baseline)

    @property
    def best_round(self) -> ExperimentRound | None:
        full_results = [r for r in self.rounds if r.passed_full_benchmark]
        if not full_results:
            return None
        contract = self.setup.contract
        maximize = contract is None or (
            contract.objective.primary_metric.direction == MetricDirection.MAXIMIZE
        )

        def value(r: ExperimentRound) -> float:
            return r.full_result.metric_value if r.full_result else 0.0

        return max(full_results, key=value) if maximize else min(full_results, key=value)


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------


class OvernightExperimentLoop:
    """Runs unattended overnight experiments."""

    def __init__(self, setup: OvernightSetup) -> None:
        self.setup = setup
        self.rounds: list[ExperimentRound] = []
        self.started_at = datetime.now()
        self.db_path = setup.artifacts_dir / "experiment_log.db"
        self.proposer = PatchProposer(setup)
        self.benchmark_runner = BenchmarkRunner(setup)
        self._init_database()
        # A restarted run knows what earlier runs and Claude-driven rounds tried.
        self.proposer.previous_rounds.extend(load_rounds(self.db_path))

    # ------------------------------------------------------------------
    # Database
    # ------------------------------------------------------------------

    def _init_database(self) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS experiments (
                    round_num        INTEGER PRIMARY KEY,
                    parent_round_num INTEGER,
                    branch_tag       TEXT,
                    source_credit    TEXT,
                    patch_description TEXT NOT NULL,
                    patch_code       TEXT NOT NULL,
                    screening_passed INTEGER,
                    screening_metric REAL,
                    screening_duration REAL,
                    full_passed      INTEGER,
                    full_metric      REAL,
                    full_improved    INTEGER,
                    full_duration    REAL,
                    rejected_reason  TEXT,
                    started_at       TEXT,
                    completed_at     TEXT
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS run_metadata (
                    key   TEXT PRIMARY KEY,
                    value TEXT
                )
            """)

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def run_overnight(self, time_budget_hours: float, max_experiments: int) -> RunSummary:
        """Run until time or experiment budget is exhausted."""
        budget_seconds = time_budget_hours * 3600
        deadline = self.started_at + timedelta(seconds=budget_seconds)
        # Continue after rounds already in the log (Claude-driven or an earlier
        # run) instead of overwriting them.
        first = self._next_round_num()
        self.round_driver = "byok"  # this loop proposes with an API key

        for round_num in range(first, first + max_experiments):
            now = datetime.now()
            if now >= deadline:
                break
            time_remaining = (deadline - now).total_seconds()

            result = self._run_experiment_round(round_num, time_remaining)
            self.rounds.append(result)
            self._persist_round(result)
            if result.hypothesis:
                from researchforge.kaggle.hypothesis_pool import mark_tested

                mark_tested(
                    self.setup.artifacts_dir,
                    {"hypothesis": result.hypothesis},
                    result.round_num,
                    result.outcome,
                )
            # Teach the proposer what happened
            self.proposer.record_round_result(result)

        ended_at = datetime.now()
        total = (ended_at - self.started_at).total_seconds()
        return RunSummary(
            setup=self.setup,
            rounds=self.rounds,
            started_at=self.started_at,
            ended_at=ended_at,
            total_duration_seconds=total,
            budget_remaining_seconds=max(0, budget_seconds - total),
            experiments_remaining=max(0, max_experiments - len(self.rounds)),
        )

    def _next_round_num(self) -> int:
        with sqlite3.connect(self.db_path) as conn:
            row = conn.execute("SELECT MAX(round_num) FROM experiments").fetchone()
        return int(row[0] or 0) + 1

    # ------------------------------------------------------------------
    # Single round
    # ------------------------------------------------------------------

    def _run_experiment_round(
        self, round_num: int, time_remaining_seconds: float
    ) -> ExperimentRound:
        """Propose → validate → screen → full benchmark."""
        from researchforge.kaggle.patch_proposer import PatchProposal

        # 1. Propose
        try:
            proposal: PatchProposal = self.proposer.propose_patch(round_num)
        except RuntimeError as exc:
            return ExperimentRound(
                round_num=round_num,
                patch_description="<proposal failed>",
                patch_code="",
                rejected_reason=str(exc),
                completed_at=datetime.now(),
            )

        # 2. Mid-loop HIL pause (if AI requested user input)
        if proposal.needs_user_input:
            from researchforge.kaggle.human_questions import UserAnswers, ask_mid_loop_questions

            updated_answers = ask_mid_loop_questions(
                proposal.needs_user_input,
                self.setup.user_answers or UserAnswers(),
                self.setup.artifacts_dir,
            )
            self.setup.user_answers = updated_answers
            self.setup.save()

        rnd = ExperimentRound(
            round_num=round_num,
            patch_description=proposal.title,
            patch_code=proposal.code_changes,
            parent_round_num=proposal.parent_round_num,
            branch_tag=proposal.branch_tag,
            source_credit=proposal.source_credit,
            hypothesis=proposal.hypothesis,
        )

        # 3. Constraint check
        if self._violates_constraints(proposal.code_changes):
            rnd.rejected_reason = "Patch violates path/security constraints"
            rnd.completed_at = datetime.now()
            return rnd

        # 4. Screening (up to 30% of remaining time, capped at 5 min)
        screen_budget = min(300.0, time_remaining_seconds * 0.3)
        rnd.screening_result = self.benchmark_runner.run_screening(
            proposal.code_changes, screen_budget
        )

        # 5. Full benchmark if screening passed (up to 70% remaining, capped at the
        # project's round time limit)
        if rnd.passed_screening:
            from researchforge.kaggle.run_config import round_time_limit_seconds

            limit = round_time_limit_seconds(self.setup.artifacts_dir)
            full_budget = min(limit, time_remaining_seconds * 0.7)
            rnd.full_result = self.benchmark_runner.run_full_benchmark(
                proposal.code_changes, full_budget
            )

        rnd.completed_at = datetime.now()
        return rnd

    # ------------------------------------------------------------------
    # Constraint guard
    # ------------------------------------------------------------------

    def _violates_constraints(self, code: str) -> bool:
        """Return True if the patch tries to access restricted paths/commands."""
        # TODO: integrate with researchforge.execution.path_guard
        return False

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _persist_round(self, r: ExperimentRound) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT INTO experiments (
                    round_num, parent_round_num, branch_tag, source_credit,
                    patch_description, patch_code,
                    screening_passed, screening_metric, screening_duration,
                    full_passed, full_metric, full_improved, full_duration,
                    rejected_reason, started_at, completed_at
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                """,
                (
                    r.round_num,
                    r.parent_round_num,
                    r.branch_tag or "",
                    r.source_credit or "",
                    r.patch_description,
                    r.patch_code,
                    int(r.screening_result.passed) if r.screening_result else None,
                    r.screening_result.metric_value if r.screening_result else None,
                    r.screening_result.duration_seconds if r.screening_result else None,
                    int(r.full_result.passed) if r.full_result else None,
                    r.full_result.metric_value if r.full_result else None,
                    int(r.full_result.metric_improved) if r.full_result else None,
                    r.full_result.duration_seconds if r.full_result else None,
                    r.rejected_reason or _failure_reason(r),
                    r.started_at.isoformat(),
                    r.completed_at.isoformat() if r.completed_at else None,
                ),
            )
        from researchforge.kaggle.auto_run import record_round_driver

        if r.full_result is not None and r.full_result.outputs:
            from researchforge.kaggle.submission import record_round_outputs

            record_round_outputs(self.db_path.parent, r.round_num, dict(r.full_result.outputs))

        # "byok" from run_overnight; otherwise auto-run's driver or the agent in use.
        record_round_driver(self.db_path.parent, r.round_num, getattr(self, "round_driver", ""))

    # ------------------------------------------------------------------
    # Morning report
    # ------------------------------------------------------------------

    def save_summary(self, summary: RunSummary) -> None:
        baseline_cv = (
            summary.setup.baseline_result.cv_score if summary.setup.baseline_result else None
        )
        data = {
            "metadata": {
                "started_at": summary.started_at.isoformat(),
                "ended_at": summary.ended_at.isoformat(),
                "total_duration_seconds": summary.total_duration_seconds,
                "competition": summary.setup.competition_slug,
                "gpu_backend": summary.setup.selected_gpu.value
                if summary.setup.selected_gpu
                else "cpu",
                "baseline_cv": baseline_cv,
            },
            "statistics": {
                "total_rounds_attempted": summary.total_rounds_attempted,
                "rounds_passed_screening": summary.rounds_passed_screening,
                "rounds_passed_full_benchmark": summary.rounds_passed_full_benchmark,
                "rounds_improved_over_baseline": summary.rounds_improved,
            },
            "experiment_dag": [
                {
                    "round_num": r.round_num,
                    "parent_round_num": r.parent_round_num,
                    "branch_tag": r.branch_tag,
                    "source_credit": r.source_credit,
                    "patch_description": r.patch_description,
                    "screening_passed": r.passed_screening,
                    "full_passed": r.passed_full_benchmark,
                    "improved": r.improved_over_baseline,
                    "metric": r.full_result.metric_value if r.full_result else None,
                    "rejected_reason": r.rejected_reason,
                }
                for r in summary.rounds
            ],
            "best_result": None,
        }

        best = summary.best_round
        if best is not None and best.full_result is not None:
            improvement_pct = 0.0
            if baseline_cv and baseline_cv != 0:
                improvement_pct = (
                    (best.full_result.metric_value - baseline_cv) / abs(baseline_cv)
                ) * 100
            data["best_result"] = {
                "round_num": best.round_num,
                "branch_tag": best.branch_tag,
                "parent_round_num": best.parent_round_num,
                "source_credit": best.source_credit,
                "patch_description": best.patch_description,
                "metric_value": best.full_result.metric_value,
                "improvement_percent": improvement_pct,
            }

        summary_path = self.setup.artifacts_dir / "overnight_summary.json"
        with summary_path.open("w") as fh:
            json.dump(data, fh, indent=2)
