"""Autonomous research loop engine.

Implements `researchforge autorun`: plans every pending hypothesis, runs each
plan with a per-plan stall, re-synthesizes new hypotheses from the results, and
repeats until the global stall, the target metric, the round cap, or the
wall-clock budget stops it.

The engine owns loop control only.  Everything consequential is delegated to
the existing enforcement boundaries: plan import validates AI output, the
executor isolates and measures, and the research log accumulates what was
learned.  Nothing here can approve a contract or bypass a path guard.
"""

from __future__ import annotations

import copy
import sqlite3
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

from researchforge.autorun.frontier import (
    Attempt,
    available_hypotheses,
    lineage_hypotheses,
    rank_hypotheses,
    round_breadth,
)
from researchforge.autorun.state import (
    AutorunState,
    RoundRecord,
    load_state,
    new_state,
    save_state,
    summarize_settings,
)
from researchforge.config.paths import research_log_path
from researchforge.domain.contract import MetricDirection
from researchforge.domain.experiment import (
    Experiment,
    ExperimentStatus,
    PlanApproval,
    PlanStatus,
    advance,
)
from researchforge.domain.hypothesis import Hypothesis
from researchforge.experiments.graph import ancestor_order
from researchforge.experiments.measurements import latest_full_execution, measured_values
from researchforge.experiments.selection import (
    BASELINE_NODE,
    MergeProposal,
    NodeStats,
    gain_over_baseline,
    propose_merge,
    selection_order,
)
from researchforge.research.research_log import (
    build_initial_log,
    build_resynth_context,
    results_instructions,
    update_log_after_round,
)
from researchforge.storage.baseline_repository import get_latest_baseline
from researchforge.storage.contract_repository import get_active_contract
from researchforge.storage.experiment_repository import (
    list_executions,
    list_experiments,
    list_plans,
    stored_parents,
    update_experiment,
    update_plan_status,
)
from researchforge.storage.hypothesis_repository import list_hypotheses
from researchforge.storage.project_repository import get_project
from researchforge.utils.progress import LiveProgress

PROGRESS_SEPARATOR = "─" * 60

ProgressFn = Callable[[str], None]


@dataclass
class PlanPreview:
    """What a human is shown before an autorun plan is allowed to execute."""

    plan_id: str
    hypothesis_id: str
    experiments: list[Experiment]
    worst_case_minutes: int


ApprovalGate = Callable[[PlanPreview], bool]


@dataclass
class RoundSummary:
    round_num: int
    hypotheses_planned: list[str]
    plan_ids: list[str]
    experiments_run: int
    promising: list[str]
    rejected: list[str]
    failed: list[str]
    best_experiment_id: str | None
    best_metric_value: float | None
    improved_over_previous: bool
    duration_seconds: float


@dataclass
class AutorunResult:
    rounds: list[RoundSummary] = field(default_factory=list)
    global_stall_count: int = 0
    best_experiment_id: str | None = None
    best_metric_value: float | None = None
    baseline_value: float | None = None
    metric_name: str = ""
    direction: MetricDirection = MetricDirection.MAXIMIZE
    target_value: float | None = None
    stop_reason: str = ""
    total_experiments: int = 0
    total_duration_seconds: float = 0.0
    objective_achieved: bool = False
    resumed_from_round: int = 0

    @property
    def improved(self) -> bool:
        return self.best_experiment_id is not None


@dataclass
class AutorunConfig:
    stall: int = 2                     # per-plan stall (experiments without improvement)
    global_stall: int = 3              # rounds with no improvement → stop
    max_rounds: int | None = None      # hard cap on rounds
    max_hours: float | None = None     # wall-clock safety limit
    target_value: float | None = None  # stop when the metric reaches this
    compound: bool = True              # new experiments build on a chosen graph node
    explore: float = 0.0               # UCB1 constant; 0 = always expand the best
    merge: bool = False                # propose combining independent winners
    observe: bool = False              # read each run's logs into an observation
    resynthesize: bool = True          # generate new hypotheses each round
    yes: bool = False                  # skip the round-1 approval gate
    provider: str | None = None        # AI provider for synthesis + planning
    model: str | None = None           # AI model override

    def as_settings(self) -> dict[str, str]:
        return summarize_settings(
            {
                "stall": self.stall,
                "global_stall": self.global_stall,
                "max_rounds": self.max_rounds,
                "max_hours": self.max_hours,
                "target_value": self.target_value,
                "compound": self.compound,
                "explore": self.explore,
                "merge": self.merge,
                "observe": self.observe,
                "resynthesize": self.resynthesize,
                "provider": self.provider,
                "model": self.model,
            }
        )


# ---------------------------------------------------------------------------
# Metric comparison — direction-aware everywhere, never hardcoded to maximize
# ---------------------------------------------------------------------------


def is_better(candidate: float, incumbent: float, direction: MetricDirection) -> bool:
    """Whether `candidate` beats `incumbent` for this metric's direction."""
    if direction is MetricDirection.MAXIMIZE:
        return candidate > incumbent
    return candidate < incumbent


def reaches_target(value: float, target: float, direction: MetricDirection) -> bool:
    """Whether `value` satisfies the objective's target for this direction."""
    if direction is MetricDirection.MAXIMIZE:
        return value >= target
    return value <= target


def target_progress_pct(
    value: float, baseline: float, target: float, direction: MetricDirection
) -> float:
    """How far the metric has travelled from baseline toward the target (0–100)."""
    span = target - baseline if direction is MetricDirection.MAXIMIZE else baseline - target
    if span <= 0:
        return 100.0
    travelled = value - baseline if direction is MetricDirection.MAXIMIZE else baseline - value
    return max(0.0, min(100.0, travelled / span * 100))


MEASURED_STATUSES = (
    ExperimentStatus.PROMISING,
    ExperimentStatus.VALIDATED,
    ExperimentStatus.IMPLEMENTATION_READY,
)

# A rejected experiment was measured too — it violated a constraint. It stays in
# the graph as a node the search may branch from, carrying no gain.
BRANCHABLE_STATUSES = (*MEASURED_STATUSES, ExperimentStatus.REJECTED)


def get_pending_hypotheses(conn: sqlite3.Connection) -> list[Hypothesis]:
    """Hypotheses that have no plan yet and were not rejected in review.

    This is the manual planner's question — "what have I not looked at at all" —
    and it is deliberately not the autorun loop's, which asks what is still
    worth trying *at a given node*. See `researchforge.autorun.frontier`.
    """
    planned = {plan.hypothesis_id for plan in list_plans(conn)}
    return [
        h
        for h in list_hypotheses(conn)
        if h.hypothesis_id not in planned and h.is_plannable
    ]


def collect_attempts(
    conn: sqlite3.Connection, baseline_value: float, direction: MetricDirection
) -> tuple[list[Attempt], dict[str, str]]:
    """Every hypothesis-at-a-node the graph already contains, and what it did.

    Also returns which hypothesis each node came from, which is what makes the
    lineage rule answerable. A merge counts as an attempt at each of its
    parents: it was written against all of them.
    """
    values = measured_values(list_executions(conn))
    experiments = list_experiments(conn)
    hypothesis_of = {e.experiment_id: e.hypothesis_id for e in experiments}

    attempts: list[Attempt] = []
    for experiment in experiments:
        value = values.get(experiment.experiment_id)
        gain = (
            gain_over_baseline(value, baseline_value, direction) if value is not None else None
        )
        for parent in experiment.parent_experiment_ids or [BASELINE_NODE]:
            reference = baseline_value if parent == BASELINE_NODE else values.get(parent)
            attempts.append(
                Attempt(
                    hypothesis_id=experiment.hypothesis_id,
                    node_id=parent,
                    gain=gain,
                    no_op=value is not None and reference is not None and value == reference,
                )
            )
    return attempts, hypothesis_of


def get_current_best(
    conn: sqlite3.Connection, baseline_value: float, direction: MetricDirection
) -> tuple[Experiment | None, float]:
    """The best measured experiment across every run, and its metric value.

    Falls back to the baseline when nothing has beaten it yet.
    """
    values = measured_values(list_executions(conn))
    best_experiment: Experiment | None = None
    best_value = baseline_value
    for experiment in list_experiments(conn):
        if experiment.status not in MEASURED_STATUSES:
            continue
        value = values.get(experiment.experiment_id)
        if value is None:
            continue
        if is_better(value, best_value, direction):
            best_value = value
            best_experiment = experiment
    return best_experiment, best_value


# ---------------------------------------------------------------------------
# The experiment graph, as the node selector sees it
# ---------------------------------------------------------------------------


@dataclass
class GraphView:
    """The stored graph reduced to what selection and merging need."""

    nodes: list[NodeStats] = field(default_factory=list)
    ancestors_of: dict[str, set[str]] = field(default_factory=dict)
    already_merged: set[frozenset[str]] = field(default_factory=set)


def build_graph_view(
    conn: sqlite3.Connection, baseline_value: float, direction: MetricDirection
) -> GraphView:
    """Read the experiment graph and score every node the loop may branch from.

    A node's visit count is how many children were spawned from it, so the
    baseline's visits are the experiments that were written against it directly.
    """
    values = measured_values(list_executions(conn))
    experiments = list_experiments(conn)
    branchable = [e for e in experiments if e.status in BRANCHABLE_STATUSES]

    visits: dict[str, int] = {BASELINE_NODE: 0}
    for experiment in experiments:
        parents = experiment.parent_experiment_ids or [BASELINE_NODE]
        for parent in parents:
            visits[parent] = visits.get(parent, 0) + 1

    view = GraphView()
    view.nodes.append(NodeStats(BASELINE_NODE, gain=0.0, visits=visits[BASELINE_NODE]))
    for experiment in branchable:
        value = values.get(experiment.experiment_id)
        gain = (
            gain_over_baseline(value, baseline_value, direction) if value is not None else 0.0
        )
        view.nodes.append(
            NodeStats(
                experiment.experiment_id,
                gain=gain,
                visits=visits.get(experiment.experiment_id, 0),
            )
        )

    parents_of = stored_parents(conn)
    for experiment in branchable:
        view.ancestors_of[experiment.experiment_id] = {
            *ancestor_order(experiment.experiment_id, parents_of),
            BASELINE_NODE,
        }
    view.already_merged = {
        frozenset(e.parent_experiment_ids) for e in experiments if e.is_merge
    }
    return view


def choose_next_move(
    conn: sqlite3.Connection,
    view: GraphView,
    config: AutorunConfig,
    loop: _LoopContext,
    remaining_minutes: float | None,
    *,
    retreat: bool = False,
) -> tuple[NodeStats | None, list[Hypothesis]]:
    """The node to expand next and the hypotheses to try there.

    Nodes are considered best-first and the first one with anything left to try
    wins, because the strongest node may be fully explored and the round should
    move down the order rather than declare the search finished.

    Moving down that order has a floor, though. Once something has improved on
    the baseline, a node that improved nothing is a step backwards: trying an
    idea there measures it without the gains already banked, which is an
    ablation, not progress. So those nodes are held back until `retreat` — the
    caller's way of saying it has already asked for new ideas and come up
    empty, and would rather run an ablation than stop.

    `--explore` opts out of that floor: a caller who asked to spend budget on
    under-explored branches has asked for exactly what is being held back.
    """
    attempts, hypothesis_of = collect_attempts(conn, loop.baseline_value, loop.direction)
    by_id = {h.hypothesis_id: h for h in list_hypotheses(conn) if h.is_plannable}
    candidates = sorted(by_id)

    order = (
        selection_order(view.nodes, config.explore)
        if config.compound
        else [NodeStats(BASELINE_NODE)]
    )
    if not retreat and config.explore <= 0 and any(node.gain > 0 for node in order):
        order = [node for node in order if node.gain > 0]

    for node in order:
        lineage = lineage_hypotheses(node.node_id, view.ancestors_of, hypothesis_of)
        open_here = available_hypotheses(candidates, node.node_id, attempts, lineage)
        if not open_here:
            continue
        ranked = rank_hypotheses(open_here, attempts)
        take = round_breadth(
            len(ranked),
            remaining_minutes,
            float(worst_case_minutes(conn, _variants_per_plan(conn))),
        )
        return node, [by_id[h] for h in ranked[:take]]
    return None, []


def _variants_per_plan(conn: sqlite3.Connection) -> int:
    """What one hypothesis costs, at worst: a plan filled to the contract's cap."""
    contract = get_active_contract(conn)
    return contract.spec.execution.max_experiments if contract is not None else 1


def _move_summary(node: NodeStats | None, pending: list[Hypothesis], loop: _LoopContext) -> str:
    """The round's move, in the terms a reader is asking about: where, and what."""
    where = "the baseline"
    if node is not None and node.node_id != BASELINE_NODE:
        where = f"{node.node_id} ({loop.metric_name} {node.gain:+.4f} vs baseline)"
    trying = ", ".join(h.hypothesis_id for h in pending)
    return f"Expanding {where} · trying {trying}"


# ---------------------------------------------------------------------------
# Planning
# ---------------------------------------------------------------------------


def force_plan_parent(plan_yaml: str, parent_experiment_id: str) -> str:
    """Set `parent:` on every plan entry that does not already declare one.

    Compound mode is enforced here rather than trusted to the prompt: the
    engine decides the lineage, the AI only writes the change.
    """
    try:
        document = yaml.safe_load(plan_yaml)
    except yaml.YAMLError:
        return plan_yaml  # import will report the malformed document
    if not isinstance(document, dict):
        return plan_yaml
    entries = document.get("experiments")
    if not isinstance(entries, list):
        return plan_yaml
    for entry in entries:
        if isinstance(entry, dict) and not entry.get("parent"):
            entry["parent"] = parent_experiment_id
    return yaml.safe_dump(document, sort_keys=False)


def _compound_instruction(context: Any, parent_experiment_id: str) -> Any:
    """A copy of the planning context telling the AI it is building on a parent."""
    instructed = copy.deepcopy(context)
    instructed.instructions.append(
        f"COMPOUND MODE: every experiment in this plan builds on {parent_experiment_id} "
        "— not on the baseline. ResearchForge sets "
        f"`parent: {parent_experiment_id}` itself. The REPOSITORY section already "
        f"shows the files as {parent_experiment_id} leaves them, so rewrite those "
        "files as they stand: do not re-apply that experiment's change, and do not "
        "expect the baseline's values."
    )
    return instructed


MERGE_PLAN_TEMPLATE = """\
hypothesis_id: {hypothesis_id}
approach_summary: >-
  Combine two independent improvements that were measured separately.
experiments:
  - key: {key}
    title: Combine {left} and {right}
    change_summary: >-
      Apply both {left} and {right} together to measure whether their
      improvements compound or interfere.
    parents: [{left}, {right}]
"""

AUTHORED_MERGE_PLAN_TEMPLATE = """\
hypothesis_id: {hypothesis_id}
approach_summary: >-
  Combine two improvements whose diffs overlap, re-authored as a single patch.
experiments:
  - key: {key}
    title: Combine {left} and {right}
    change_summary: >-
      Apply the changes from both {left} and {right} in one diff. Their patches
      edit the same lines, so they cannot be stacked and the combination was
      written as a single change against the baseline.
    parents: [{left}, {right}]
    patch_file: {patches_dir}/{key}.patch
    patch_includes_parents: true
"""


def merge_key(left: str, right: str) -> str:
    """A plan key naming the pair, within the importer's key pattern."""
    return f"merge-{left}-{right}".replace("exp-", "")


def _write_merge_plan(document: str) -> Path:
    from researchforge.config.paths import experiments_dir

    staging = experiments_dir()
    staging.mkdir(parents=True, exist_ok=True)
    plan_path = staging / "plan.yaml"
    plan_path.write_text(document, encoding="utf-8")
    return plan_path


def author_merged_patch(
    conn: sqlite3.Connection,
    left: Experiment,
    right: Experiment,
    provider_hint: str | None,
    model_hint: str | None,
) -> str:
    """Ask the AI for one patch containing both branches' changes.

    Raises MergeNotPossibleError when no usable diff can be produced — that is
    a normal outcome for two changes that genuinely contradict each other.
    """
    from researchforge.ai.merge_gen import MergeBranch, MergeNotPossibleError, generate_merged_patch
    from researchforge.ai.service import resolve_provider

    contract = get_active_contract(conn)
    if contract is None:
        raise MergeNotPossibleError("no approved contract")

    try:
        ai_provider = resolve_provider(provider_hint=provider_hint, model_hint=model_hint)
    except RuntimeError as exc:
        raise MergeNotPossibleError(str(exc)) from exc

    return generate_merged_patch(
        MergeBranch(left.experiment_id, left.title, left.change_summary, left.patch_text),
        MergeBranch(right.experiment_id, right.title, right.change_summary, right.patch_text),
        contract.spec.permissions.editable_paths,
        contract.spec.permissions.protected_paths,
        ai_provider,
    )


def _plan_authored_merge(
    conn: sqlite3.Connection,
    left: Experiment,
    right: Experiment,
    provider_hint: str | None,
    model_hint: str | None,
    on_progress: ProgressFn | None,
) -> str | None:
    """Fall back to an AI-authored combined patch. Returns a plan id, or None."""
    from researchforge.ai.merge_gen import MergeNotPossibleError
    from researchforge.config.paths import experiments_dir
    from researchforge.experiments.context_export import PATCHES_DIR_NAME
    from researchforge.experiments.importers import import_experiment_plan

    _report(on_progress, "  Diffs overlap — asking the AI for one combined patch…")
    try:
        patch_text = author_merged_patch(conn, left, right, provider_hint, model_hint)
    except MergeNotPossibleError as exc:
        _report(on_progress, f"  ✗ No combined patch: {exc}")
        return None
    except Exception as exc:  # noqa: BLE001 — a failed merge must not end the loop
        _report(on_progress, f"  ✗ Merge authoring failed: {exc}")
        return None

    key = merge_key(left.experiment_id, right.experiment_id)
    patches = experiments_dir() / PATCHES_DIR_NAME
    patches.mkdir(parents=True, exist_ok=True)
    (patches / f"{key}.patch").write_text(patch_text, encoding="utf-8")

    plan_path = _write_merge_plan(
        AUTHORED_MERGE_PLAN_TEMPLATE.format(
            hypothesis_id=left.hypothesis_id,
            key=key,
            left=left.experiment_id,
            right=right.experiment_id,
            patches_dir=PATCHES_DIR_NAME,
        )
    )
    result, plan = import_experiment_plan(conn, plan_path)
    if result.ok and plan is not None:
        _report(on_progress, "  ✓ Combined patch accepted")
        return plan.plan_id

    _report(on_progress, f"  ✗ Combined patch rejected: {'; '.join(result.errors)}")
    return None


def plan_merge(
    conn: sqlite3.Connection,
    proposal: MergeProposal,
    provider_hint: str | None = None,
    model_hint: str | None = None,
    on_progress: ProgressFn | None = None,
) -> str | None:
    """Import a plan combining two winners. Returns its plan id, or None.

    Composition is tried first and needs no AI: a pure merge contributes no
    diff of its own, so the plan is written directly and the importer decides
    whether the two branches actually compose.  Only when they conflict — both
    branches edited the same lines — is the AI asked to author the combination
    as a single patch instead.
    """
    from researchforge.experiments.importers import MERGE_CONFLICT_PREFIX, import_experiment_plan
    from researchforge.storage.experiment_repository import get_experiment

    left_id, right_id = proposal.parents
    left = get_experiment(conn, left_id)
    right = get_experiment(conn, right_id)
    if left is None or right is None:
        return None

    _report(on_progress, f"Proposing merge of {left_id} + {right_id}…")
    plan_path = _write_merge_plan(
        MERGE_PLAN_TEMPLATE.format(
            hypothesis_id=left.hypothesis_id,
            key=merge_key(left_id, right_id),
            left=left_id,
            right=right_id,
        )
    )
    result, plan = import_experiment_plan(conn, plan_path)
    if result.ok and plan is not None:
        return plan.plan_id

    if any(MERGE_CONFLICT_PREFIX in error for error in result.errors):
        return _plan_authored_merge(conn, left, right, provider_hint, model_hint, on_progress)

    _report(
        on_progress,
        f"  ✗ {left_id} + {right_id} cannot be combined: {'; '.join(result.errors)}",
    )
    return None


def plan_all_hypotheses(
    conn: sqlite3.Connection,
    hypotheses: list[Hypothesis],
    provider_hint: str | None,
    model_hint: str | None,
    parent_experiment_id: str | None,
    on_progress: ProgressFn | None = None,
) -> list[str]:
    """Plan each hypothesis with the AI. Returns the plan ids that imported.

    `parent_experiment_id` is the graph node the caller chose to expand; None
    means write the diffs against the baseline.
    """
    from researchforge.ai.plan_gen import (
        describe_plan,
        generate_experiment_plan,
        planning_phase_label,
        write_patch_files,
    )
    from researchforge.ai.service import resolve_provider
    from researchforge.experiments.context_export import (
        build_experiment_context,
        write_experiment_context,
    )
    from researchforge.experiments.importers import import_experiment_plan

    try:
        ai_provider = resolve_provider(provider_hint=provider_hint, model_hint=model_hint)
    except RuntimeError as exc:
        raise RuntimeError(f"Cannot plan experiments: {exc}") from exc

    compound_parent = parent_experiment_id
    plan_ids: list[str] = []
    for hypothesis in hypotheses:
        _report(on_progress, f"Planning {hypothesis.hypothesis_id}: {hypothesis.title[:50]}…")
        try:
            # The provider call is the long silence in a round: a minute or two
            # with nothing to show for it. Keep a clock on screen throughout.
            with LiveProgress(
                f"{hypothesis.hypothesis_id} · {ai_provider.name}",
                enabled=on_progress is not None,
            ) as live:
                live.phase(
                    "Reading the editable files as "
                    + (f"{compound_parent} leaves them…" if compound_parent else "the baseline…")
                )
                # Built with the parent, so the files the model rewrites are
                # the ones its patch will actually be applied to.
                context = build_experiment_context(
                    conn, hypothesis.hypothesis_id, parent_experiment_id=compound_parent
                )
                write_experiment_context(context)
                if compound_parent is not None:
                    context = _compound_instruction(context, compound_parent)

                live.phase(planning_phase_label(context, ai_provider.name))
                plan_yaml, patches = generate_experiment_plan(context, ai_provider)

                live.phase("Writing plan.yaml and patches…")
                if compound_parent is not None:
                    plan_yaml = force_plan_parent(plan_yaml, compound_parent)

                plan_path = Path(context.expected_artifacts.plan_path)
                patches_dir = Path(context.expected_artifacts.patches_dir)
                plan_path.parent.mkdir(parents=True, exist_ok=True)
                plan_path.write_text(plan_yaml, encoding="utf-8")
                write_patch_files(patches_dir, patches)

            _report(on_progress, f"  {describe_plan(plan_yaml, patches)}")

            result, plan = import_experiment_plan(conn, plan_path)
            if result.ok and plan is not None:
                plan_ids.append(plan.plan_id)
                _report(on_progress, f"  ✓ {plan.plan_id} imported")
            else:
                _report(
                    on_progress,
                    f"  ✗ {hypothesis.hypothesis_id} plan rejected: {'; '.join(result.errors)}",
                )
        except Exception as exc:  # noqa: BLE001 — one bad hypothesis must not end the loop
            _report(on_progress, f"  ✗ {hypothesis.hypothesis_id} planning failed: {exc}")

    return plan_ids


# ---------------------------------------------------------------------------
# Running
# ---------------------------------------------------------------------------


def worst_case_minutes(conn: sqlite3.Connection, experiment_count: int) -> int:
    contract = get_active_contract(conn)
    if contract is None:
        return experiment_count * 20
    stages = 2 if contract.spec.execution.screening_command else 1
    return experiment_count * contract.spec.execution.timeout_minutes * stages


def approve_plan(
    conn: sqlite3.Connection, plan_id: str, experiments: list[Experiment], typed: bool
) -> None:
    """Record approval for a plan's runnable experiments."""
    approval = PlanApproval(
        approved_at=datetime.now(UTC),
        method="typed" if typed else "flag",
        experiment_ids=[e.experiment_id for e in experiments],
        estimated_max_minutes=worst_case_minutes(conn, len(experiments)),
    )
    update_plan_status(conn, plan_id, PlanStatus.APPROVED, approval)
    for experiment in experiments:
        update_experiment(
            conn,
            experiment.model_copy(
                update={"status": advance(experiment.status, ExperimentStatus.APPROVED)}
            ),
        )


def run_single_plan(
    conn: sqlite3.Connection,
    plan_id: str,
    stall: int | None,
    on_progress: ProgressFn | None = None,
    gate: ApprovalGate | None = None,
) -> tuple[list[Experiment], list[Experiment], list[Experiment]]:
    """Run one plan. Returns its (promising, rejected, failed) experiments.

    When `gate` is supplied the plan only runs if the gate returns True — that
    is the human approval for the first autonomous batch.
    """
    from researchforge.execution.experiments import (
        ExperimentBlockedError,
        execute_run,
        start_run,
    )
    from researchforge.storage.experiment_repository import get_plan

    plan = get_plan(conn, plan_id)
    if plan is None:
        return [], [], []

    runnable = [e for e in list_experiments(conn, plan_id) if e.status is ExperimentStatus.PLANNED]
    if not runnable:
        return [], [], []

    if gate is not None:
        preview = PlanPreview(
            plan_id=plan_id,
            hypothesis_id=plan.hypothesis_id,
            experiments=runnable,
            worst_case_minutes=worst_case_minutes(conn, len(runnable)),
        )
        if not gate(preview):
            _report(on_progress, f"  {plan_id} not approved — autorun stopped.")
            raise AutorunDeclined(plan_id)

    approve_plan(conn, plan_id, runnable, typed=gate is not None)
    _report(on_progress, f"Running {len(runnable)} experiment(s) for {plan_id}…")

    try:
        prep, run = start_run(conn, plan_id)
        # Each experiment is a full install-and-benchmark cycle, so this is the
        # longest stretch of the round by far. Show the stage it is on.
        with LiveProgress(
            f"{plan_id} · {len(runnable)} experiment(s)",
            enabled=on_progress is not None,
        ) as live:
            execute_run(
                conn,
                prep,
                run,
                stall_override=stall,
                on_phase=live.phase,
                on_result=live.note,
            )
    except ExperimentBlockedError as exc:
        _report(on_progress, f"  ✗ Run blocked: {exc}")
        return [], [], []

    results = list_experiments(conn, plan_id)
    promising = [e for e in results if e.status in MEASURED_STATUSES]
    rejected = [e for e in results if e.status is ExperimentStatus.REJECTED]
    failed = [
        e
        for e in results
        if e.status in (ExperimentStatus.FAILED_SETUP, ExperimentStatus.FAILED_EXECUTION)
    ]
    return promising, rejected, failed


class AutorunDeclined(RuntimeError):
    """The human declined the plan at the approval gate."""

    def __init__(self, plan_id: str) -> None:
        super().__init__(f"{plan_id} was not approved")
        self.plan_id = plan_id


# ---------------------------------------------------------------------------
# Observations
# ---------------------------------------------------------------------------


def observe_experiments(
    conn: sqlite3.Connection,
    experiments: list[Experiment],
    baseline_value: float,
    metric_name: str,
    provider_hint: str | None,
    model_hint: str | None,
    on_progress: ProgressFn | None = None,
) -> dict[str, str]:
    """Read each run's benchmark output back into a one-paragraph observation.

    Returns the observations by experiment id and stores each one on its
    experiment.  Observations are commentary: a failure to produce one is
    reported and skipped, never fatal, and nothing here can change a metric,
    a constraint result or a status.
    """
    from researchforge.ai.observation_gen import (
        ObservationRequest,
        generate_observation,
        read_log_tail,
    )
    from researchforge.ai.service import resolve_provider

    if not experiments:
        return {}

    try:
        ai_provider = resolve_provider(provider_hint=provider_hint, model_hint=model_hint)
    except RuntimeError as exc:
        _report(on_progress, f"  ✗ Cannot write observations: {exc}")
        return {}

    executions = list_executions(conn)
    values = measured_values(executions)
    observations: dict[str, str] = {}

    for experiment in experiments:
        execution = latest_full_execution(executions, experiment.experiment_id)
        if execution is None:
            continue
        request = ObservationRequest(
            experiment_id=experiment.experiment_id,
            title=experiment.title,
            change_summary=experiment.change_summary,
            metric_name=metric_name,
            baseline_value=baseline_value,
            measured_value=values.get(experiment.experiment_id),
            status=experiment.status.value,
            log_tail=read_log_tail(Path(execution.artifacts.stdout_path)),
        )
        try:
            observation = generate_observation(request, ai_provider)
        except Exception as exc:  # noqa: BLE001 — commentary must not end the loop
            _report(on_progress, f"  ✗ {experiment.experiment_id} observation failed: {exc}")
            continue
        if observation is None:
            continue
        observations[experiment.experiment_id] = observation
        update_experiment(conn, experiment.model_copy(update={"observation": observation}))
        _report(on_progress, f"  {experiment.experiment_id}: {observation}")

    return observations


# ---------------------------------------------------------------------------
# Re-synthesis
# ---------------------------------------------------------------------------


def _log_instruction(bundle: Any, log_content: str) -> Any:
    """A copy of the synthesis context carrying the accumulated research log."""
    contextualized = copy.deepcopy(bundle)
    contextualized.instructions = [
        *contextualized.instructions,
        *results_instructions(log_content),
    ]
    return contextualized


def resynthesize_hypotheses(
    conn: sqlite3.Connection,
    provider_hint: str | None,
    model_hint: str | None,
    log_content: str,
    on_progress: ProgressFn | None = None,
) -> list[str]:
    """Generate new hypotheses grounded in the results so far. Returns new ids."""
    from researchforge.ai.service import resolve_provider
    from researchforge.ai.synthesis import synthesize as ai_synthesize
    from researchforge.ai.synthesis import write_artifacts
    from researchforge.config.settings import load_settings
    from researchforge.research.context_export import build_context
    from researchforge.research.importers import (
        import_additional_hypotheses,
        import_landscape,
    )
    from researchforge.storage.scan_repository import get_latest_scan

    _report(on_progress, "Re-synthesizing hypotheses from results…")

    try:
        ai_provider = resolve_provider(provider_hint=provider_hint, model_hint=model_hint)
    except RuntimeError as exc:
        _report(on_progress, f"  ✗ Cannot re-synthesize: {exc}")
        return []

    project = get_project(conn)
    scan = get_latest_scan(conn)
    if project is None or scan is None:
        return []

    settings = load_settings()
    bundle = build_context(conn, project, scan, settings)
    if not bundle.papers:
        _report(on_progress, "  No papers stored — nothing to re-synthesize from.")
        return []

    if log_content:
        bundle = _log_instruction(bundle, log_content)

    try:
        landscape_yaml, hypotheses_yaml = ai_synthesize(bundle, ai_provider)
    except ValueError as exc:
        _report(on_progress, f"  ✗ Synthesis failed: {exc}")
        return []

    landscape_path = Path(bundle.expected_artifacts.landscape_path)
    hypotheses_path = Path(bundle.expected_artifacts.hypotheses_path)
    write_artifacts(landscape_yaml, hypotheses_yaml, landscape_path, hypotheses_path)

    import_landscape(conn, landscape_path, project.id)
    additions = import_additional_hypotheses(conn, hypotheses_path, project.id, settings)
    if not additions.ok:
        _report(
            on_progress,
            f"  ✗ Hypothesis import rejected: {'; '.join(additions.result.errors)}",
        )
        return []

    if additions.restated:
        _report(
            on_progress,
            f"  {len(additions.restated)} candidate(s) restated an idea already "
            f"on record ({', '.join(sorted(set(additions.restated)))}) — skipped.",
        )
    _report(on_progress, f"  ✓ {len(additions.added)} new hypothesis(es)")
    return additions.added


# ---------------------------------------------------------------------------
# The loop
# ---------------------------------------------------------------------------


def _report(on_progress: ProgressFn | None, message: str) -> None:
    if on_progress is not None:
        on_progress(message)


@dataclass
class _LoopContext:
    """Everything the loop needs that does not change between rounds."""

    direction: MetricDirection
    metric_name: str
    baseline_value: float
    log: Path


def _prepare(conn: sqlite3.Connection) -> tuple[_LoopContext, str]:
    """Validate the project is loop-ready and initialize the research log."""
    project = get_project(conn)
    if project is None or not project.objective:
        raise RuntimeError("No project found. Run `researchforge project create` first.")

    baseline = get_latest_baseline(conn)
    if baseline is None or baseline.metrics is None:
        raise RuntimeError("No frozen baseline. Run `researchforge baseline run` first.")

    contract = get_active_contract(conn)
    if contract is None:
        raise RuntimeError("No approved contract. Run `researchforge contract approve` first.")

    log = research_log_path()
    if not log.is_file():
        log.parent.mkdir(parents=True, exist_ok=True)
        log.write_text(build_initial_log(project.objective, baseline), encoding="utf-8")

    metric = contract.spec.objective.primary_metric
    return (
        _LoopContext(
            direction=metric.direction,
            metric_name=metric.name,
            baseline_value=baseline.metrics.primary_metric.value,
            log=log,
        ),
        project.objective,
    )


def _restore(result: AutorunResult, state: AutorunState) -> float:
    """Seed a result from persisted state. Returns the elapsed-time offset."""
    result.global_stall_count = state.global_stall_count
    result.best_experiment_id = state.best_experiment_id
    result.best_metric_value = state.best_metric_value
    result.total_experiments = state.total_experiments
    result.resumed_from_round = state.rounds_completed
    result.rounds = [
        RoundSummary(
            round_num=record.round_num,
            hypotheses_planned=record.hypotheses_planned,
            plan_ids=record.plan_ids,
            experiments_run=record.experiments_run,
            promising=record.promising,
            rejected=record.rejected,
            failed=record.failed,
            best_experiment_id=record.best_experiment_id,
            best_metric_value=record.best_metric_value,
            improved_over_previous=record.improved,
            duration_seconds=record.duration_seconds,
        )
        for record in state.rounds
    ]
    return state.elapsed_seconds


def _persist(state: AutorunState, result: AutorunResult, elapsed: float) -> AutorunState:
    updated = state.model_copy(
        update={
            "rounds": [
                RoundRecord(
                    round_num=summary.round_num,
                    hypotheses_planned=summary.hypotheses_planned,
                    plan_ids=summary.plan_ids,
                    experiments_run=summary.experiments_run,
                    promising=summary.promising,
                    rejected=summary.rejected,
                    failed=summary.failed,
                    best_experiment_id=summary.best_experiment_id,
                    best_metric_value=summary.best_metric_value,
                    improved=summary.improved_over_previous,
                    duration_seconds=summary.duration_seconds,
                    completed_at=datetime.now(UTC),
                )
                for summary in result.rounds
            ],
            "rounds_completed": len(result.rounds),
            "global_stall_count": result.global_stall_count,
            "best_experiment_id": result.best_experiment_id,
            "best_metric_value": result.best_metric_value,
            "total_experiments": result.total_experiments,
            "objective_achieved": result.objective_achieved,
            "stop_reason": result.stop_reason,
            "elapsed_seconds": elapsed,
            "status": "stopped" if result.stop_reason else "running",
        }
    )
    save_state(updated)
    return updated


def run_autorun(
    conn: sqlite3.Connection,
    config: AutorunConfig,
    on_progress: ProgressFn | None = None,
    gate: ApprovalGate | None = None,
    resume: bool = False,
) -> AutorunResult:
    """Execute the autonomous research loop.

    `gate` is consulted before the first plan of a fresh loop runs — the human
    approval for autonomous execution.  Subsequent plans run unattended.
    """
    loop, _objective = _prepare(conn)
    result = AutorunResult(
        baseline_value=loop.baseline_value,
        metric_name=loop.metric_name,
        direction=loop.direction,
        target_value=config.target_value,
    )

    elapsed_offset = 0.0
    state = new_state(config.as_settings())
    if resume:
        previous = load_state()
        if previous is None:
            raise RuntimeError(
                "No autorun state to resume. Start a loop with `researchforge autorun`."
            )
        elapsed_offset = _restore(result, previous)
        state = previous.model_copy(update={"status": "running"})
        _report(
            on_progress,
            f"Resuming after round {previous.rounds_completed} "
            f"(stall {previous.global_stall_count}/{config.global_stall}, "
            f"{elapsed_offset / 3600:.1f}h already spent).",
        )

    started = time.monotonic()

    def elapsed_seconds() -> float:
        return elapsed_offset + (time.monotonic() - started)

    def budget_exhausted() -> bool:
        return config.max_hours is not None and elapsed_seconds() >= config.max_hours * 3600

    best_experiment, best_value = get_current_best(conn, loop.baseline_value, loop.direction)
    result.best_experiment_id = best_experiment.experiment_id if best_experiment else None
    result.best_metric_value = best_value

    round_num = result.resumed_from_round

    while True:
        round_num += 1

        if config.max_rounds is not None and round_num > config.max_rounds:
            result.stop_reason = f"max rounds ({config.max_rounds}) reached"
            break
        if result.global_stall_count >= config.global_stall:
            result.stop_reason = (
                f"global stall: {config.global_stall} consecutive rounds with no improvement"
            )
            break
        if budget_exhausted():
            result.stop_reason = f"time limit ({config.max_hours}h) reached"
            break

        _report(on_progress, f"\n{PROGRESS_SEPARATOR}")
        _report(
            on_progress,
            f"Round {round_num}"
            + (f"/{config.max_rounds}" if config.max_rounds else "")
            + f" · best {loop.metric_name} = {best_value:.4f}"
            + (f" ({best_experiment.experiment_id})" if best_experiment else " (baseline)")
            + f" · stall {result.global_stall_count}/{config.global_stall}",
        )
        if config.target_value is not None:
            pct = target_progress_pct(
                best_value, loop.baseline_value, config.target_value, loop.direction
            )
            _report(
                on_progress,
                f"  Target: {best_value:.4f} / {config.target_value} ({pct:.0f}%)",
            )

        # Which node the round expands. With --explore 0 this is the current
        # best; above 0 the search can pivot to an under-explored branch.
        view = build_graph_view(conn, loop.baseline_value, loop.direction)

        merge_plan_ids: list[str] = []
        if config.merge:
            proposal = propose_merge(view.nodes, view.ancestors_of, view.already_merged)
            if proposal is not None:
                merged = plan_merge(
                    conn, proposal, config.provider, config.model, on_progress
                )
                if merged is not None:
                    merge_plan_ids.append(merged)

        remaining = (
            None
            if config.max_hours is None
            else max(0.0, config.max_hours * 60 - elapsed_seconds() / 60)
        )
        node, pending = choose_next_move(conn, view, config, loop, remaining)

        # Nothing left anywhere in the graph is what re-synthesis is for. That
        # is a fact about the frontier, not about which round this happens to
        # be: a second invocation starts at round one with the whole graph
        # already explored.
        asked_for_more = False
        if not pending and not merge_plan_ids and config.resynthesize:
            asked_for_more = True
            new_ids = resynthesize_hypotheses(
                conn,
                config.provider,
                config.model,
                build_resynth_context(loop.log),
                on_progress,
            )
            if new_ids:
                node, pending = choose_next_move(conn, view, config, loop, remaining)

        # Out of new ideas and out of moves on the winning branches. Old ideas
        # on the branches that never won are all that is left, so take them
        # rather than stop with the clock still running.
        if not pending and not merge_plan_ids:
            node, pending = choose_next_move(
                conn, view, config, loop, remaining, retreat=True
            )

        if not pending and not merge_plan_ids:
            result.stop_reason = (
                "no new hypotheses could be generated"
                if asked_for_more
                else "no hypotheses left to try"
            )
            break

        expand_from = node.node_id if node is not None and node.node_id != BASELINE_NODE else None
        if pending:
            _report(on_progress, f"  {_move_summary(node, pending, loop)}")

        round_started = time.monotonic()
        plan_ids = merge_plan_ids + plan_all_hypotheses(
            conn,
            pending,
            config.provider,
            config.model,
            expand_from,
            on_progress,
        )
        if not plan_ids:
            result.global_stall_count += 1
            _report(on_progress, "  No plan survived validation this round.")
            state = _persist(state, result, elapsed_seconds())
            continue

        promising: list[Experiment] = []
        rejected: list[Experiment] = []
        failed: list[Experiment] = []
        first_plan_gate = gate if (round_num == 1 and not config.yes) else None

        for index, plan_id in enumerate(plan_ids):
            if budget_exhausted():
                _report(
                    on_progress,
                    f"  Time limit reached — {len(plan_ids) - index} plan(s) left unrun.",
                )
                break
            try:
                plan_promising, plan_rejected, plan_failed = run_single_plan(
                    conn,
                    plan_id,
                    config.stall,
                    on_progress,
                    gate=first_plan_gate if index == 0 else None,
                )
            except AutorunDeclined:
                result.stop_reason = "plan not approved"
                _persist(state, result, elapsed_seconds())
                result.total_duration_seconds = elapsed_seconds()
                return result
            promising.extend(plan_promising)
            rejected.extend(plan_rejected)
            failed.extend(plan_failed)

        previous_best = best_value
        best_experiment, best_value = get_current_best(conn, loop.baseline_value, loop.direction)
        improved = is_better(best_value, previous_best, loop.direction)

        if improved:
            result.global_stall_count = 0
            result.best_experiment_id = (
                best_experiment.experiment_id if best_experiment else None
            )
            result.best_metric_value = best_value
            _report(
                on_progress,
                f"  ✓ New best: {loop.metric_name} = {best_value:.4f} "
                f"({best_experiment.experiment_id if best_experiment else 'baseline'})",
            )
        else:
            result.global_stall_count += 1
            _report(
                on_progress,
                f"  ✗ No improvement (stall "
                f"{result.global_stall_count}/{config.global_stall})",
            )

        round_experiments = [*promising, *rejected, *failed]
        observations: dict[str, str] = {}
        if config.observe and round_experiments:
            _report(on_progress, "  Reading run output…")
            observations = observe_experiments(
                conn,
                round_experiments,
                loop.baseline_value,
                loop.metric_name,
                config.provider,
                config.model,
                on_progress,
            )

        update_log_after_round(
            loop.log,
            round_num,
            round_experiments,
            loop.baseline_value,
            loop.metric_name,
            best_experiment,
            best_value,
            loop.direction,
            observations,
        )

        result.rounds.append(
            RoundSummary(
                round_num=round_num,
                hypotheses_planned=[h.hypothesis_id for h in pending],
                plan_ids=plan_ids,
                experiments_run=len(round_experiments),
                promising=[e.experiment_id for e in promising],
                rejected=[e.experiment_id for e in rejected],
                failed=[e.experiment_id for e in failed],
                best_experiment_id=best_experiment.experiment_id if best_experiment else None,
                best_metric_value=best_value,
                improved_over_previous=improved,
                duration_seconds=time.monotonic() - round_started,
            )
        )
        result.total_experiments += len(round_experiments)

        if config.target_value is not None and reaches_target(
            best_value, config.target_value, loop.direction
        ):
            result.objective_achieved = True
            comparison = "≥" if loop.direction is MetricDirection.MAXIMIZE else "≤"
            result.stop_reason = (
                f"objective achieved: {loop.metric_name} = {best_value:.4f} "
                f"{comparison} {config.target_value}"
            )
            state = _persist(state, result, elapsed_seconds())
            break

        state = _persist(state, result, elapsed_seconds())

    result.total_duration_seconds = elapsed_seconds()
    if not result.stop_reason:
        result.stop_reason = "completed all rounds"
    _persist(state, result, result.total_duration_seconds)
    return result
