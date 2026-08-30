"""Validation and import of Claude-authored synthesis artifacts.

This is the enforcement boundary of the Claude<->CLI handshake: whatever
the synthesis wrote, nothing is persisted unless every layer passes —
parse, schema, referential integrity, uniqueness, cross-field rules.
Errors are field-level and actionable so the author can fix and retry;
imports are transactional and idempotent.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass, field
from pathlib import Path

from pydantic import ValidationError

from researchforge.config.settings import ResearchSettings
from researchforge.domain.hypothesis import Hypothesis
from researchforge.domain.landscape import ResearchLandscape
from researchforge.domain.project import ProjectStatus
from researchforge.project.service import touch_project_status
from researchforge.research.context_export import HypothesesArtifact
from researchforge.research.hypothesis_dedup import find_duplicate
from researchforge.storage.hypothesis_repository import (
    list_hypotheses,
    next_hypothesis_ids,
    replace_hypotheses,
)
from researchforge.storage.paper_repository import get_paper, list_papers, paper_ids, upsert_paper
from researchforge.storage.synthesis_repository import replace_landscape
from researchforge.utils.artifact_io import ArtifactLoadError, load_artifact

_NOVELTY_PHRASES = (
    "is novel",
    "first ever",
    "first-ever",
    "no prior work",
    "never been studied",
    "guaranteed novelty",
    "completely new",
    "unprecedented",
)


@dataclass
class ImportResult:
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors


def _format_validation_error(exc: ValidationError) -> list[str]:
    messages = []
    for error in exc.errors():
        location = ".".join(str(part) for part in error["loc"])
        messages.append(f"{location or '<root>'}: {error['msg']}")
    return messages


def _lint_novelty_language(text: str, where: str, warnings: list[str]) -> None:
    lowered = text.lower()
    for phrase in _NOVELTY_PHRASES:
        if phrase in lowered:
            warnings.append(
                f"{where}: contains novelty-claim language ({phrase!r}); "
                "prefer gap language such as 'underexplored'."
            )


def _check_paper_refs(referenced: set[str], known: set[str], where: str, errors: list[str]) -> None:
    unknown = sorted(referenced - known)
    if unknown:
        errors.append(
            f"{where}: unknown paper id(s) {', '.join(unknown)} — "
            "run `researchforge papers list` to see valid ids."
        )


def import_landscape(conn: sqlite3.Connection, path: Path, project_id: str) -> ImportResult:
    result = ImportResult()

    try:
        raw = load_artifact(path)
    except ArtifactLoadError as exc:
        result.errors.append(str(exc))
        return result

    try:
        landscape = ResearchLandscape.model_validate(raw)
    except ValidationError as exc:
        result.errors.extend(_format_validation_error(exc))
        return result

    known = paper_ids(conn)

    direction_ids = [d.direction_id for d in landscape.directions]
    if len(direction_ids) != len(set(direction_ids)):
        result.errors.append("directions: duplicate direction_id values.")
    for direction in landscape.directions:
        _check_paper_refs(
            set(direction.paper_ids), known, f"directions.{direction.direction_id}", result.errors
        )
        for text_field, content in (
            ("description", direction.description),
            ("established_findings", " ".join(direction.established_findings)),
            ("underexplored_aspects", " ".join(direction.underexplored_aspects)),
        ):
            _lint_novelty_language(
                content, f"directions.{direction.direction_id}.{text_field}", result.warnings
            )

    annotation_ids = [a.paper_id for a in landscape.paper_annotations]
    if len(annotation_ids) != len(set(annotation_ids)):
        result.errors.append("paper_annotations: duplicate paper_id values.")
    _check_paper_refs(set(annotation_ids), known, "paper_annotations", result.errors)

    evidence_ids = [e.evidence_id for e in landscape.evidence]
    if len(evidence_ids) != len(set(evidence_ids)):
        result.errors.append("evidence: duplicate evidence_id values.")
    _check_paper_refs({e.paper_id for e in landscape.evidence}, known, "evidence", result.errors)

    _lint_novelty_language(landscape.summary, "summary", result.warnings)

    if not result.ok:
        return result

    replace_landscape(conn, project_id, landscape, source_file=str(path))
    _merge_annotations_onto_papers(conn, project_id, landscape)
    return result


def _merge_annotations_onto_papers(
    conn: sqlite3.Connection, project_id: str, landscape: ResearchLandscape
) -> None:
    for annotation in landscape.paper_annotations:
        paper = get_paper(conn, annotation.paper_id)
        if paper is None:  # pragma: no cover — guarded by validation above
            continue
        updated = paper.model_copy(
            update={
                "evidence_strength": annotation.evidence_strength,
                "method_summary": annotation.method_summary,
                "reported_findings": annotation.reported_findings,
                "limitations": annotation.limitations,
                "repository_relevance": annotation.repository_relevance,
            }
        )
        upsert_paper(conn, project_id, updated)


def _load_hypotheses(path: Path, result: ImportResult) -> list[Hypothesis] | None:
    """Parse and schema-check an artifact; None means the errors say why."""
    try:
        raw = load_artifact(path)
    except ArtifactLoadError as exc:
        result.errors.append(str(exc))
        return None

    try:
        artifact = HypothesesArtifact.model_validate(raw)
    except ValidationError as exc:
        result.errors.extend(_format_validation_error(exc))
        return None
    return artifact.hypotheses


def _validate_hypotheses(
    conn: sqlite3.Connection,
    hypotheses: list[Hypothesis],
    settings: ResearchSettings,
    result: ImportResult,
) -> None:
    """Referential integrity, cross-field rules and grounding lints."""
    known = paper_ids(conn)

    ids = [h.hypothesis_id for h in hypotheses]
    if len(ids) != len(set(ids)):
        result.errors.append("hypotheses: duplicate hypothesis_id values.")

    for hypothesis in hypotheses:
        where = f"hypotheses.{hypothesis.hypothesis_id}"
        _check_paper_refs(
            set(hypothesis.supporting_paper_ids) | set(hypothesis.contradicting_paper_ids),
            known,
            where,
            result.errors,
        )
        overlap = set(hypothesis.supporting_paper_ids) & set(hypothesis.contradicting_paper_ids)
        if overlap:
            result.errors.append(
                f"{where}: paper(s) {', '.join(sorted(overlap))} appear in both "
                "supporting and contradicting lists."
            )
        for text_field, content in (
            ("claim", hypothesis.claim),
            ("rationale", hypothesis.rationale),
        ):
            _lint_novelty_language(content, f"{where}.{text_field}", result.warnings)
        if not hypothesis.supporting_paper_ids:
            result.warnings.append(
                f"{where}: no supporting evidence — will be labeled UNSUPPORTED."
            )

    count = len(hypotheses)
    if count < settings.hypothesis_min or count > settings.hypothesis_max:
        result.warnings.append(
            f"hypotheses: {count} provided; expected between "
            f"{settings.hypothesis_min} and {settings.hypothesis_max}."
        )


def import_hypotheses(
    conn: sqlite3.Connection,
    path: Path,
    project_id: str,
    settings: ResearchSettings,
) -> ImportResult:
    """Import a synthesis artifact as the project's hypothesis set, replacing it."""
    result = ImportResult()
    hypotheses = _load_hypotheses(path, result)
    if hypotheses is None:
        return result

    _validate_hypotheses(conn, hypotheses, settings, result)
    if not result.ok:
        return result

    replace_hypotheses(conn, project_id, hypotheses)
    recompute_paper_backlinks(conn, project_id, hypotheses)
    touch_project_status(conn, ProjectStatus.SYNTHESIZED)
    return result


@dataclass
class HypothesisAdditions:
    """What an additive hypothesis import changed."""

    result: ImportResult = field(default_factory=ImportResult)
    added: list[str] = field(default_factory=list)
    """Ids of the hypotheses that were stored, after renumbering."""

    restated: list[str] = field(default_factory=list)
    """Ids of stored hypotheses a dropped candidate was restating."""

    @property
    def ok(self) -> bool:
        return self.result.ok


def import_additional_hypotheses(
    conn: sqlite3.Connection,
    path: Path,
    project_id: str,
    settings: ResearchSettings,
) -> HypothesisAdditions:
    """Add a re-synthesis artifact to the stored set rather than replacing it.

    Everything already stored is kept: a later round must not delete the
    hypotheses whose plans and experiments are on record.  Candidates that
    restate a stored hypothesis are dropped, and the survivors are renumbered
    onto fresh ids so they cannot collide with ids already in use.
    """
    additions = HypothesisAdditions()
    hypotheses = _load_hypotheses(path, additions.result)
    if hypotheses is None:
        return additions

    _validate_hypotheses(conn, hypotheses, settings, additions.result)
    if not additions.ok:
        return additions

    stored = list_hypotheses(conn)
    fresh: list[Hypothesis] = []
    for candidate in hypotheses:
        duplicate = find_duplicate(candidate, stored)
        if duplicate is None:
            fresh.append(candidate)
            continue
        additions.restated.append(duplicate.hypothesis_id)
        additions.result.warnings.append(
            f"hypotheses.{candidate.hypothesis_id}: restates "
            f"{duplicate.hypothesis_id} ({duplicate.title!r}) — skipped."
        )

    if not fresh:
        additions.result.warnings.append(
            "hypotheses: every candidate restated one already on record; nothing was added."
        )
        return additions

    renumbered = [
        candidate.model_copy(update={"hypothesis_id": new_id})
        for candidate, new_id in zip(fresh, next_hypothesis_ids(conn, len(fresh)), strict=True)
    ]
    combined = [*stored, *renumbered]
    replace_hypotheses(conn, project_id, combined)
    recompute_paper_backlinks(conn, project_id, combined)
    touch_project_status(conn, ProjectStatus.SYNTHESIZED)
    additions.added = [h.hypothesis_id for h in renumbered]
    return additions


def recompute_paper_backlinks(
    conn: sqlite3.Connection, project_id: str, hypotheses: list[Hypothesis]
) -> None:
    """Rewrite supports/contradicts on every paper from the hypotheses (idempotent)."""
    supports: dict[str, list[str]] = {}
    contradicts: dict[str, list[str]] = {}
    for hypothesis in hypotheses:
        for pid in hypothesis.supporting_paper_ids:
            supports.setdefault(pid, []).append(hypothesis.hypothesis_id)
        for pid in hypothesis.contradicting_paper_ids:
            contradicts.setdefault(pid, []).append(hypothesis.hypothesis_id)

    for paper in list_papers(conn):
        updated = paper.model_copy(
            update={
                "supports_hypotheses": sorted(supports.get(paper.paper_id, [])),
                "contradicts_hypotheses": sorted(contradicts.get(paper.paper_id, [])),
            }
        )
        upsert_paper(conn, project_id, updated)
