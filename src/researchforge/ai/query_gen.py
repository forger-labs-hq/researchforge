"""AI-powered arXiv search query generation.

Replaces the purely algorithmic fallback in ``queries.py`` when an AI
provider is available.  Produces domain-specific queries that dramatically
improve paper relevance vs the tokenisation-based fallback.

Example — objective: "Improve YOLOv5s mAP@0.5 on COCO128 while keeping
inference under 10ms on Apple Silicon MPS"

Algorithmic (bad): "improve yolov5s mAP without exceeding latency budget",
                   "improve yolov5s", "yolov5s mAP", ...

AI-generated (good): "YOLOv5 model pruning COCO object detection",
                     "knowledge distillation YOLO real-time detection",
                     "anchor optimization YOLOv5 mAP improvement",
                     "attention mechanism YOLO inference speed",
                     "neural architecture search efficient object detection"
"""

from __future__ import annotations

import json
import re

from researchforge.ai.providers import AiProvider
from researchforge.config.settings import ResearchSettings
from researchforge.domain.repo_scan import RepoScan

_SYSTEM = """\
You are a research librarian generating arXiv search queries.
Given a software improvement objective and optional repository context, \
produce concise, domain-specific arXiv search queries that will retrieve \
the most relevant machine learning and computer science papers.

Rules:
- Return ONLY a JSON array of query strings, nothing else.
- Each query must be specific and technical (not generic phrases like \
"improve performance" or "latency budget").
- Use arXiv-friendly terminology: model names, technique names, task names.
- Queries should be diverse — cover different research sub-directions.
- Do NOT include instructions, commentary, or explanation.

Example output:
["YOLOv5 model pruning object detection", \
"knowledge distillation YOLO real-time inference", \
"anchor optimization COCO mAP improvement"]
"""


def generate_queries_with_ai(
    objective: str,
    scan: RepoScan | None,
    settings: ResearchSettings,
    provider: AiProvider,
) -> list[str]:
    """Use an AI provider to generate targeted arXiv search queries.

    Falls back to an empty list on any error so callers can degrade to the
    algorithmic fallback without crashing.
    """
    repo_context = ""
    if scan is not None:
        keywords = ", ".join(scan.keywords[:10]) if scan.keywords else "none"
        repo_context = (
            f"\nRepository keywords: {keywords}"
            f"\nRepository compatibility: {scan.compatibility}"
        )
        if scan.readme.title:
            repo_context += f"\nProject title: {scan.readme.title}"

    user_prompt = (
        f"Objective: {objective}{repo_context}\n\n"
        f"Generate between {settings.min_queries} and {settings.max_queries} "
        "specific arXiv search queries for this objective. "
        "Return ONLY a JSON array of strings."
    )

    try:
        raw = provider.generate(_SYSTEM, user_prompt, max_tokens=512)
        # Extract JSON array from response (model may wrap it in markdown)
        queries = _parse_json_array(raw)
        # Clamp to configured bounds
        queries = [q.strip() for q in queries if isinstance(q, str) and q.strip()]
        return queries[: settings.max_queries]
    except Exception:  # noqa: BLE001  # always fall back
        return []


def _parse_json_array(text: str) -> list[str]:
    """Extract a JSON array from model output, tolerating markdown fences."""
    # Strip ```json ... ``` fences
    text = re.sub(r"```(?:json)?\s*", "", text).strip().rstrip("`").strip()
    # Find first '[' ... ']'
    start = text.find("[")
    end = text.rfind("]")
    if start == -1 or end == -1:
        return []
    parsed: object = json.loads(text[start : end + 1])
    if not isinstance(parsed, list):
        return []
    return [item for item in parsed if isinstance(item, str)]
