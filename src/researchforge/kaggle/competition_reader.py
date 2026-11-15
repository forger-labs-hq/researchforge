"""Read real competition metadata from Kaggle and extract signal with AI.

Strategy:
  1. Fetch raw competition info via Kaggle CLI
  2. Fetch the public competition page for description + evaluation section
  3. Use the configured AI provider to extract: metric name, direction, task type,
     data description, and anything else the patch proposer needs
  4. Persist to .overnight_<slug>/competition_metadata.json so it is not re-fetched
"""

from __future__ import annotations

import contextlib
import json
import subprocess
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from researchforge.ai.providers import AiProvider


# ---------------------------------------------------------------------------
# Domain model
# ---------------------------------------------------------------------------


@dataclass
class CompetitionMetadata:
    """Everything the loop needs to know about the competition."""

    slug: str
    title: str = ""
    description: str = ""
    evaluation_section: str = ""  # raw text of the Evaluation tab
    metric_name: str = ""  # e.g. "AUC", "RMSE", "log_loss"
    metric_direction: str = ""  # "maximize" or "minimize"
    # Where metric_direction came from: "leaderboard" (rank order), "ai" (read from
    # the evaluation text), "keyword" (a known metric name), "guess" (nothing to go
    # on — confirm with the user) or "user" (set with `kaggle settings`).
    metric_direction_source: str = ""
    task_type: str = ""  # "binary classification", "regression", …
    data_description: str = ""  # brief summary of available files
    tags: list[str] = field(default_factory=list)
    deadline: str = ""
    prize: str = ""
    is_code_competition: bool = False  # True = notebook submission required
    estimated_data_gb: float = 0.0  # 0 = unknown
    # AI-extracted patterns for parsing benchmark output – list of regex strings
    metric_extraction_patterns: list[str] = field(default_factory=list)
    # Raw JSON from kaggle CLI (for debugging / future use)
    raw_kaggle_info: dict[str, Any] = field(default_factory=dict)

    def save(self, artifacts_dir: Path) -> None:
        path = artifacts_dir / "competition_metadata.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w") as fh:
            json.dump(asdict(self), fh, indent=2)

    @classmethod
    def load(cls, artifacts_dir: Path) -> CompetitionMetadata | None:
        path = artifacts_dir / "competition_metadata.json"
        if not path.exists():
            return None
        with path.open() as fh:
            data = json.load(fh)
        return cls(**data)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def read_competition_metadata(
    slug: str,
    artifacts_dir: Path,
    ai_provider: AiProvider | None = None,
    force_refresh: bool = False,
) -> CompetitionMetadata:
    """Return fully populated CompetitionMetadata.

    Loads from cache if available and force_refresh is False.  Uses the AI
    provider to extract the metric when one is configured; falls back to
    heuristics otherwise.
    """
    # Load from cache if present
    if not force_refresh:
        cached = CompetitionMetadata.load(artifacts_dir)
        if cached is not None:
            return cached

    meta = CompetitionMetadata(slug=slug)

    # --- 1. Kaggle CLI: competitions list (gives title + deadline + category) ---
    raw = _fetch_kaggle_info(slug)
    meta.raw_kaggle_info = raw
    meta.title = raw.get("title", slug)
    meta.deadline = raw.get("deadline", "")
    meta.prize = raw.get("reward", "")
    meta.tags = [t.strip() for t in raw.get("tags", "").split(",") if t.strip()]

    # --- 2. Public page: description + evaluation section ---
    page_text = _fetch_public_page(slug)
    if page_text:
        meta.description = _extract_description(page_text)
        meta.evaluation_section = _extract_evaluation_section(page_text)

    # --- 3. AI extraction (optional) ---
    if ai_provider is not None and (meta.description or meta.evaluation_section):
        _ai_enrich(meta, ai_provider)
    else:
        _heuristic_enrich(meta)

    # --- 3b. The leaderboard's order is a measurement, not a reading ---
    from researchforge.kaggle.api import leaderboard_direction

    measured = leaderboard_direction(slug)
    if measured:
        meta.metric_direction = measured
        meta.metric_direction_source = "leaderboard"

    # --- 4. Detect code competition and estimate data size ---
    meta.is_code_competition = _detect_code_competition(slug, page_text)
    meta.estimated_data_gb = _estimate_data_size(slug)

    meta.save(artifacts_dir)
    return meta


# ---------------------------------------------------------------------------
# Kaggle CLI helpers
# ---------------------------------------------------------------------------


def _fetch_kaggle_info(slug: str) -> dict[str, Any]:
    """Call 'kaggle competitions list --search <slug>' and parse the first match."""
    try:
        result = subprocess.run(
            ["kaggle", "competitions", "list", "--search", slug, "-v"],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        if result.returncode != 0 or not result.stdout.strip():
            return {}
        lines = [
            ln
            for ln in result.stdout.strip().splitlines()
            if ln.strip() and not ln.strip().lower().startswith("warning")
        ]
        if len(lines) < 2:
            return {}
        # CSV: ref,deadline,category,reward,teamCount,userHasEntered
        headers = [h.strip() for h in lines[0].split(",")]
        for line in lines[1:]:
            parts = line.split(",")
            if len(parts) >= len(headers) and slug in parts[0]:
                row = dict(zip(headers, [p.strip() for p in parts], strict=False))
                row["title"] = parts[0].split("/")[-1].replace("-", " ").title()
                return row
        # fallback: take first data row
        if len(lines) >= 2:
            parts = lines[1].split(",")
            headers = [h.strip() for h in lines[0].split(",")]
            return dict(zip(headers, [p.strip() for p in parts], strict=False))
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass
    return {}


# ---------------------------------------------------------------------------
# Public page scraping (no auth needed)
# ---------------------------------------------------------------------------


def _fetch_public_page(slug: str) -> str:
    """Fetch the plain-text content of the competition's public overview page."""
    url = f"https://www.kaggle.com/competitions/{slug}/overview"
    try:
        req = urllib.request.Request(
            url,
            headers={
                "User-Agent": "ResearchForge/1.0 (+https://github.com/forger-labs-hq/researchforge)"
            },
        )
        context = None
        try:
            import ssl

            import certifi

            context = ssl.create_default_context(cafile=certifi.where())
        except ImportError:
            pass
        with urllib.request.urlopen(req, timeout=15, context=context) as resp:
            html = resp.read().decode("utf-8", errors="replace")
        return _html_to_text(html)
    except (urllib.error.URLError, OSError):
        return ""


def _html_to_text(html: str) -> str:
    """Very minimal HTML → text: strip tags, decode entities."""
    import re

    text = re.sub(r"<script[^>]*>.*?</script>", " ", html, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"<style[^>]*>.*?</style>", " ", text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"<[^>]+>", " ", text)
    text = text.replace("&amp;", "&").replace("&lt;", "<").replace("&gt;", ">")
    text = text.replace("&quot;", '"').replace("&#39;", "'").replace("&nbsp;", " ")
    text = re.sub(r"\s{3,}", "\n\n", text)
    return text.strip()


def _extract_description(text: str) -> str:
    """Return the first 1 500 chars as competition description."""
    return text[:1500].strip()


def _extract_evaluation_section(text: str) -> str:
    """Try to isolate the Evaluation/Metric section from page text."""
    import re

    # Look for typical section header
    for pattern in [
        r"(?i)evaluation\s*\n(.*?)(?=\n[A-Z][a-z]|\Z)",
        r"(?i)metric[s]?\s*\n(.*?)(?=\n[A-Z][a-z]|\Z)",
    ]:
        m = re.search(pattern, text, re.DOTALL)
        if m:
            return m.group(1)[:800].strip()
    return ""


# ---------------------------------------------------------------------------
# AI enrichment
# ---------------------------------------------------------------------------


_EXTRACTION_SYSTEM = """\
You are a data science expert. Given a Kaggle competition's description and evaluation section,
extract structured information as JSON. Return ONLY valid JSON, no prose.

JSON schema:
{
  "metric_name": "<primary metric name, e.g. AUC, RMSE, log_loss, F1, accuracy>",
  "metric_direction": "<maximize or minimize>",
  "task_type": "<e.g. binary/multi-class classification, regression, object detection, NLP>",
  "data_description": "<one sentence summarising the data files and their key columns>",
  "metric_extraction_patterns": ["<Python regex for the metric in stdout; group 1 = the number>"]
}

Rules:
- metric_direction is 'minimize' for RMSE, MAE, log_loss and error rates; else 'maximize'
- Include 2-4 regex patterns in metric_extraction_patterns covering common print formats;
  let group 1 take a sign, whole numbers and exponents: (-?[0-9]+(?:\\.[0-9]+)?(?:[eE][-+]?[0-9]+)?)
- Be concise; data_description must be under 120 chars
"""


def _ai_enrich(meta: CompetitionMetadata, provider: AiProvider) -> None:
    """Call AI to extract structured fields from description + evaluation text."""
    user_prompt = (
        f"Competition slug: {meta.slug}\n"
        f"Title: {meta.title}\n\n"
        f"=== Description ===\n{meta.description[:600]}\n\n"
        f"=== Evaluation ===\n{meta.evaluation_section[:600]}"
    )
    try:
        raw = provider.generate(_EXTRACTION_SYSTEM, user_prompt, max_tokens=512)
        # Strip markdown fences if present
        raw = raw.strip()
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[1]
            raw = raw.rsplit("```", 1)[0]
        parsed = json.loads(raw)
        meta.metric_name = parsed.get("metric_name", meta.metric_name)
        meta.metric_direction = parsed.get("metric_direction", meta.metric_direction)
        meta.metric_direction_source = "ai"
        meta.task_type = parsed.get("task_type", meta.task_type)
        meta.data_description = parsed.get("data_description", meta.data_description)
        meta.metric_extraction_patterns = parsed.get(
            "metric_extraction_patterns", meta.metric_extraction_patterns
        )
    except (json.JSONDecodeError, KeyError):
        # AI response malformed — fall through to heuristic
        _heuristic_enrich(meta)


# ---------------------------------------------------------------------------
# Heuristic fallback
# ---------------------------------------------------------------------------


def _heuristic_enrich(meta: CompetitionMetadata) -> None:
    """Fill in metric info from slug/description keywords when AI is unavailable."""
    import re

    slug = meta.slug.lower()
    desc = (meta.description + " " + meta.evaluation_section).lower()

    # Metric detection priority: explicit keywords in description first
    if re.search(r"\broc[- ]?auc\b|\bauc\b", desc):
        meta.metric_name = meta.metric_name or "auc"
        meta.metric_direction = meta.metric_direction or "maximize"
    elif re.search(r"\brmse\b", desc):
        meta.metric_name = meta.metric_name or "rmse"
        meta.metric_direction = meta.metric_direction or "minimize"
    elif re.search(r"\bmae\b", desc):
        meta.metric_name = meta.metric_name or "mae"
        meta.metric_direction = meta.metric_direction or "minimize"
    elif re.search(r"\blog.?loss\b", desc):
        meta.metric_name = meta.metric_name or "log_loss"
        meta.metric_direction = meta.metric_direction or "minimize"
    elif re.search(r"\bf1\b", desc):
        meta.metric_name = meta.metric_name or "f1"
        meta.metric_direction = meta.metric_direction or "maximize"
    elif re.search(r"\baccuracy\b", desc):
        meta.metric_name = meta.metric_name or "accuracy"
        meta.metric_direction = meta.metric_direction or "maximize"
    # Slug-based fallbacks: a guess from the URL, not from the evaluation text
    elif re.search(r"regress|price|cost|sale|demand|forecast", slug):
        meta.metric_name = meta.metric_name or "rmse"
        meta.metric_direction = meta.metric_direction or "minimize"
        meta.metric_direction_source = meta.metric_direction_source or "guess"
    elif re.search(r"detect|class|cancer|fraud|churn|sentiment", slug):
        meta.metric_name = meta.metric_name or "auc"
        meta.metric_direction = meta.metric_direction or "maximize"
        meta.metric_direction_source = meta.metric_direction_source or "guess"
    else:
        meta.metric_name = meta.metric_name or "score"
        meta.metric_direction = meta.metric_direction or "maximize"
        meta.metric_direction_source = meta.metric_direction_source or "guess"
    meta.metric_direction_source = meta.metric_direction_source or "keyword"

    # Task type from slug
    if not meta.task_type:
        if re.search(r"regress|price|forecast", slug):
            meta.task_type = "regression"
        elif re.search(r"detect|segmen", slug):
            meta.task_type = "object detection"
        elif re.search(r"class|sentiment|toxic", slug):
            meta.task_type = "binary classification"
        else:
            meta.task_type = "classification"

    # Default extraction patterns based on metric name
    if not meta.metric_extraction_patterns:
        meta.metric_extraction_patterns = [
            rf"(?i){re.escape(meta.metric_name)}[:\s=]+([0-9]+\.[0-9]+)",
            r"(?i)CV Score[:\s=]+([0-9]+\.[0-9]+)",
            r"(?i)Metric[:\s=]+([0-9]+\.[0-9]+)",
            r"(?i)Score[:\s=]+([0-9]+\.[0-9]+)",
        ]


# ---------------------------------------------------------------------------
# Competition type and data size detection
# ---------------------------------------------------------------------------


def _detect_code_competition(slug: str, page_text: str) -> bool:
    """Return True if this is a Code Competition (notebook submission required).

    Code competitions don't accept CSV uploads — experiments must run as
    committed Kaggle notebooks on Kaggle's servers.

    Strategy (in order):
      1. Authenticated Kaggle REST API — most reliable, returns isKernelSubmissionsOnly
      2. Slug/pattern heuristics — fast offline fallback
    """
    import re

    # --- 1. Authenticated Kaggle REST API ---
    api_result = _fetch_competition_api(slug)
    if api_result is not None:
        return api_result

    # --- 2. Slug-pattern heuristics (fast, offline) ---
    code_patterns = [
        r"arc-prize",
        r"arc-agi",
        r"lmsys",
        r"neurips.*llm",
        r"rsna.*\d{4}",  # Most recent RSNA competitions are code
    ]
    return any(re.search(pat, slug) for pat in code_patterns)


def _fetch_competition_api(slug: str) -> bool | None:
    """Whether Kaggle says this is a code competition (kernels-only submissions).

    Returns:
        True  — confirmed code competition
        False — confirmed NOT a code competition
        None  — could not determine (no credentials, SDK unavailable, etc.)
    """
    value = competition_facts(slug).get("kernels_only")
    return None if value is None else bool(value)


def competition_facts(slug: str) -> dict[str, Any]:
    """What Kaggle's API says about one competition: ``kernels_only`` (a code
    competition) and ``max_daily`` (submissions allowed per day). {} when unknown."""
    try:
        from kaggle import api as kapi  # type: ignore[import-not-found]
    except Exception:
        # The kaggle CLI is often installed for another Python than ResearchForge's;
        # its SDK answers the same question there.
        return _competition_facts_via_cli(slug)
    try:
        kapi.authenticate()
        results = kapi.competitions_list(search=slug, page=1)
        for comp in getattr(results, "competitions", None) or []:
            # ref is like "https://www.kaggle.com/competitions/{slug}"
            if (getattr(comp, "ref", "") or "").rstrip("/").endswith(slug):
                return _facts(
                    getattr(comp, "is_kernels_submissions_only", None),
                    getattr(comp, "max_daily_submissions", None),
                )
    except Exception:
        pass
    return {}


def _facts(kernels_only: Any, max_daily: Any) -> dict[str, Any]:
    out: dict[str, Any] = {}
    if kernels_only is not None:
        out["kernels_only"] = bool(kernels_only)
    with contextlib.suppress(TypeError, ValueError):
        if max_daily is not None and int(max_daily) > 0:
            out["max_daily"] = int(max_daily)
    return out


_FACTS_SNIPPET = """
import sys
from kaggle.api.kaggle_api_extended import KaggleApi
api = KaggleApi()
api.authenticate()
for c in api.competitions_list(search=sys.argv[1], page=1).competitions or []:
    if (c.ref or "").rstrip("/").endswith(sys.argv[1]):
        print("RF_KERNELS_ONLY=" + str(bool(c.is_kernels_submissions_only)))
        print("RF_MAX_DAILY=" + str(c.max_daily_submissions or 0))
"""


def _competition_facts_via_cli(slug: str) -> dict[str, Any]:
    """competition_facts through the Python the `kaggle` command runs on."""
    import shutil

    exe = shutil.which("kaggle")
    if not exe:
        return {}
    try:
        first = Path(exe).read_text(encoding="utf-8", errors="replace").split("\n", 1)[0]
        python = first[2:].strip().split()[0] if first.startswith("#!") else ""
        if not python or not Path(python).is_file():
            return {}
        out = subprocess.run(
            [python, "-I", "-c", _FACTS_SNIPPET, slug],
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        ).stdout
    except (OSError, subprocess.TimeoutExpired):
        return {}
    wanted = ("RF_KERNELS_ONLY=", "RF_MAX_DAILY=")
    found = dict(line.split("=", 1) for line in out.splitlines() if line.startswith(wanted))
    if not found:
        return {}
    return _facts(
        found["RF_KERNELS_ONLY"].strip() == "True" if "RF_KERNELS_ONLY" in found else None,
        found.get("RF_MAX_DAILY"),
    )


def _estimate_data_size(slug: str) -> float:
    """Estimate competition data size in GB by querying the Kaggle CLI.

    Returns 0.0 if the size cannot be determined.
    """
    try:
        result = subprocess.run(
            ["kaggle", "competitions", "files", slug, "--csv"],
            capture_output=True,
            text=True,
            timeout=20,
            check=False,
        )
        if result.returncode != 0:
            return 0.0
        lines = [
            ln
            for ln in result.stdout.strip().splitlines()
            if ln.strip() and not ln.strip().lower().startswith("warning")
        ]
        total_bytes = 0
        # CSV columns are name,size,creationDate — size is index 1, not the last column
        for line in lines[1:]:  # skip header
            parts = line.split(",")
            if len(parts) >= 2:
                with contextlib.suppress(ValueError):
                    total_bytes += int(parts[1].strip())
        return round(total_bytes / 1e9, 1)
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return 0.0
