"""Background intelligence gathering — runs parallel to experiments.

Three scanners run while experiments execute:
  1. arXiv scanner: new papers on the research topic → inject as hypotheses
  2. Metric confidence tagger: marks experiments as local-reliable or local-blind
  3. Next-batch preparer: while exp-N runs, pre-computes context for exp-N+1

All scanners:
  - Run in background threads, never block the experiment loop
  - Never abort if they fail — errors are logged and swallowed
  - Write results to .researchforge/intelligence/ for the loop to pick up
"""

from __future__ import annotations

import json
import re
import threading
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from researchforge.ai.providers import AiProvider


# ---------------------------------------------------------------------------
# arXiv background scanner
# ---------------------------------------------------------------------------


@dataclass
class ArxivIdea:
    """A technique extracted from an arXiv paper."""

    arxiv_id: str
    title: str
    idea: str  # extracted technique (not the paper text)
    relevance: str  # why this is relevant to the project
    url: str
    injected_at: str = field(default_factory=lambda: datetime.now().isoformat())


def run_arxiv_scanner(
    query: str,
    intelligence_dir: Path,
    ai_provider: AiProvider | None = None,
    max_results: int = 10,
    interval_hours: float = 24.0,
) -> None:
    """Background daemon: search arXiv daily and inject new paper techniques.

    Runs indefinitely. Call in a background thread.
    Writes new ideas to intelligence_dir/arxiv_ideas.json.
    """
    while True:
        try:
            _run_arxiv_cycle(query, intelligence_dir, ai_provider, max_results)
        except Exception as exc:
            _log(intelligence_dir, f"arxiv scanner error: {exc}")
        time.sleep(interval_hours * 3600)


def start_arxiv_scanner_thread(
    query: str,
    intelligence_dir: Path,
    ai_provider: AiProvider | None = None,
) -> threading.Thread:
    """Start the arXiv scanner as a background daemon thread."""
    t = threading.Thread(
        target=run_arxiv_scanner,
        args=(query, intelligence_dir, ai_provider),
        daemon=True,
        name="rf-arxiv-scanner",
    )
    t.start()
    return t


def _run_arxiv_cycle(
    query: str,
    intelligence_dir: Path,
    ai_provider: AiProvider | None,
    max_results: int,
) -> None:
    """One arXiv search cycle."""
    from researchforge.research.arxiv_client import ArxivClient

    intelligence_dir.mkdir(parents=True, exist_ok=True)
    seen_path = intelligence_dir / "arxiv_seen.json"
    seen_ids: set[str] = set()
    if seen_path.exists():
        try:
            seen_ids = set(json.loads(seen_path.read_text()))
        except Exception:
            seen_ids = set()

    client = ArxivClient()
    entries = client.search(query, max_results=max_results, page_size=min(max_results, 25))

    new_entries = [e for e in entries if e.arxiv_id not in seen_ids]
    if not new_entries:
        _log(intelligence_dir, f"arxiv: no new papers for query '{query[:50]}'")
        return

    # Extract technique ideas from new papers
    ideas: list[ArxivIdea] = []
    for entry in new_entries[:5]:  # limit AI calls per cycle
        idea_text = _extract_idea_from_paper(entry, ai_provider)
        if idea_text:
            ideas.append(
                ArxivIdea(
                    arxiv_id=entry.arxiv_id,
                    title=entry.title[:100],
                    idea=idea_text,
                    relevance=f"arXiv {entry.arxiv_id}: {entry.title[:60]}",
                    url=f"https://arxiv.org/abs/{entry.arxiv_id}",
                )
            )

    if not ideas:
        seen_ids.update(e.arxiv_id for e in new_entries)
        seen_path.write_text(json.dumps(list(seen_ids)))
        return

    # Append to intelligence pool
    pool_path = intelligence_dir / "arxiv_ideas.json"
    existing: list[dict[str, Any]] = []
    if pool_path.exists():
        try:
            existing = json.loads(pool_path.read_text())
        except Exception:
            existing = []

    existing_ids = {i.get("arxiv_id") for i in existing}
    new_ideas = [asdict(i) for i in ideas if i.arxiv_id not in existing_ids]
    if new_ideas:
        pool_path.write_text(json.dumps(existing + new_ideas, indent=2))
        _log(intelligence_dir, f"arxiv: +{len(new_ideas)} ideas from {len(new_entries)} new papers")

    seen_ids.update(e.arxiv_id for e in new_entries)
    seen_path.write_text(json.dumps(list(seen_ids)))


_IDEA_EXTRACTION_SYSTEM = """\
Given an arXiv paper title and abstract, extract ONE concrete, actionable
technique idea that could be applied to improve an ML experiment.
Return ONLY the technique as a single sentence (max 100 chars). No prose.
If the paper is not relevant to ML improvement, return exactly: SKIP
"""


def _extract_idea_from_paper(entry: Any, ai_provider: AiProvider | None) -> str | None:
    """Extract one technique from a paper. Returns None if not applicable."""
    if ai_provider is None:
        # Heuristic fallback: use title as idea
        return f"Technique from: {entry.title[:80]}"

    try:
        abstract = getattr(entry, "summary", "")[:400]
        prompt = f"Title: {entry.title}\n\nAbstract: {abstract}"
        raw = ai_provider.generate(_IDEA_EXTRACTION_SYSTEM, prompt, max_tokens=64)
        raw = raw.strip()
        if raw.upper() == "SKIP" or not raw:
            return None
        return raw[:100]
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Metric confidence tagger
# ---------------------------------------------------------------------------


class MetricConfidence:
    HIGH = "high"  # local benchmark reliably reflects real evaluation
    MEDIUM = "medium"  # local is a noisy proxy
    LOW = "low"  # local has some signal but may miss real gains
    BLIND = "blind"  # local cannot measure this change at all


# Patterns that indicate a change is invisible to the local benchmark
_BLIND_PATTERNS = [
    r"coconut",
    r"external.?data",
    r"fallback",
    r"out.?of.?vocab",
    r"missing.?compound",
    r"unknown.?molecule",
    r"novel.?compound",
    r"new.?entries",
]

# Patterns that indicate the change is directly testable locally
_HIGH_CONFIDENCE_PATTERNS = [
    r"channel.?1",
    r"rerank",
    r"weight",
    r"score",
    r"threshold",
    r"normalize",
    r"feature",
    r"model",
    r"embedding",
]


def tag_metric_confidence(
    hypothesis_text: str,
    changed_files: list[str],
    known_blind_modules: list[str] | None = None,
) -> str:
    """Return MetricConfidence for an experiment.

    Args:
        hypothesis_text: The hypothesis title/description
        changed_files: Files the experiment modifies
        known_blind_modules: Module names known to be invisible to local benchmark
    """
    text_lower = (hypothesis_text or "").lower()
    files_lower = " ".join(changed_files or []).lower()
    blind_modules = set(known_blind_modules or [])

    # Check for blind patterns
    for pattern in _BLIND_PATTERNS:
        if re.search(pattern, text_lower) or re.search(pattern, files_lower):
            return MetricConfidence.BLIND

    # Check if any changed file is in a known blind module
    for f in changed_files:
        for blind_mod in blind_modules:
            if blind_mod in f:
                return MetricConfidence.BLIND

    # Check for high-confidence patterns
    for pattern in _HIGH_CONFIDENCE_PATTERNS:
        if re.search(pattern, text_lower):
            return MetricConfidence.HIGH

    return MetricConfidence.MEDIUM


def confidence_note(confidence: str) -> str:
    """Human-readable note about what the confidence level means."""
    return {
        MetricConfidence.HIGH: "Local benchmark is reliable for this change.",
        MetricConfidence.MEDIUM: "Local benchmark is a noisy proxy — worth trying.",
        MetricConfidence.LOW: "Local benchmark has limited signal for this change.",
        MetricConfidence.BLIND: (
            "Local benchmark CANNOT measure this change. "
            "It modifies behaviour only visible on molecules/samples "
            "not in the local validation set. "
            "Consider a real Kaggle submission to validate."
        ),
    }.get(confidence, "Unknown confidence level.")


# ---------------------------------------------------------------------------
# Next-batch preparation
# ---------------------------------------------------------------------------


@dataclass
class NextBatchStatus:
    """Status of the next-batch pre-computation."""

    ready: bool
    hypothesis_ids: list[str]
    prepared_at: str
    note: str = ""


def prepare_next_batch_background(
    conn: Any,
    current_round_hypotheses: list[str],
    intelligence_dir: Path,
) -> None:
    """Pre-compute the next batch context in a background thread.

    While experiment N runs, this identifies what experiment N+1 will be
    and pre-fetches its context. This eliminates the gap between experiments.
    """

    def _work() -> None:
        try:
            from researchforge.autorun.engine import get_pending_hypotheses

            pending = get_pending_hypotheses(conn)
            # Exclude hypotheses already in the current round
            next_batch = [h for h in pending if h.hypothesis_id not in current_round_hypotheses]
            if not next_batch:
                return

            status = NextBatchStatus(
                ready=True,
                hypothesis_ids=[h.hypothesis_id for h in next_batch[:3]],
                prepared_at=datetime.now().isoformat(),
                note=f"{len(next_batch)} hypotheses ready for next round",
            )
            out = intelligence_dir / "next_batch_status.json"
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps(asdict(status), indent=2))
        except Exception:
            pass

    t = threading.Thread(target=_work, daemon=True, name="rf-next-batch-prep")
    t.start()


def get_next_batch_status(intelligence_dir: Path) -> NextBatchStatus | None:
    """Read the pre-computed next batch status."""
    path = intelligence_dir / "next_batch_status.json"
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text())
        return NextBatchStatus(**data)
    except Exception:
        return None


def inject_arxiv_ideas_into_hypothesis_pool(
    intelligence_dir: Path,
    project_dir: Path | None = None,
) -> int:
    """Move new arXiv ideas from intelligence pool into the project's hypothesis pool.

    Returns count of ideas injected.
    """
    arxiv_pool = intelligence_dir / "arxiv_ideas.json"
    if not arxiv_pool.exists():
        return 0

    try:
        ideas = json.loads(arxiv_pool.read_text())
    except Exception:
        return 0

    uninjected = [i for i in ideas if not i.get("injected", False)]
    if not uninjected:
        return 0

    # Write to pending_hypotheses.json in project dir (for Kaggle) or RF hypotheses
    base = project_dir or Path.cwd()
    # Try Kaggle-style first
    for candidate_dir in [
        base,
        base / ".researchforge",
    ]:
        pool_path = candidate_dir / "pending_hypotheses.json"
        if pool_path.parent.exists():
            existing: list[dict[str, Any]] = []
            if pool_path.exists():
                try:
                    existing = json.loads(pool_path.read_text())
                except Exception:
                    existing = []
            existing_ideas = {h.get("idea", "").lower() for h in existing}
            added = 0
            for idea in uninjected:
                idea_text = idea.get("idea", "")
                if idea_text.lower() not in existing_ideas:
                    existing.append(
                        {
                            "idea": idea_text,
                            "source": "arxiv_scanner",
                            "source_ref": idea.get("arxiv_id", ""),
                            "url": idea.get("url", ""),
                            "tested": False,
                            "injected_at": datetime.now().isoformat(),
                        }
                    )
                    added += 1
            if added:
                pool_path.write_text(json.dumps(existing, indent=2))
                # Mark as injected
                for idea in uninjected:
                    idea["injected"] = True
                arxiv_pool.write_text(json.dumps(ideas, indent=2))
                return added
            break

    return 0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _log(intelligence_dir: Path, message: str) -> None:
    """Append a timestamped log line."""
    try:
        log_path = intelligence_dir / "scanner.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("a") as fh:
            fh.write(f"[{datetime.now().isoformat()}] {message}\n")
    except Exception:
        pass
