"""AI-powered evaluation script generation.

Given a repository scan and project objective, generates two files:
  benchmarks/evaluate.py  — the benchmark wrapper ResearchForge runs
  src/config.py           — tunable constants experiments patch / override

The AI never invents metrics: it infers them from the objective text and the
repository's existing dependencies/README.
"""

from __future__ import annotations

import re

from researchforge.ai.providers import AiProvider
from researchforge.domain.project import Project
from researchforge.domain.repo_scan import RepoScan

_SYSTEM = """\
You are generating two Python files for a ResearchForge benchmark project.

ResearchForge runs experiments by:
1. Applying a git patch (or env-var overrides) to src/
2. Running the benchmark command
3. Reading artifacts/results.json

You must produce EXACTLY two files wrapped in XML tags:
<eval_script>   → content of  benchmarks/evaluate.py
</eval_script>
<config>        → content of  src/config.py
</config>

RULES FOR benchmarks/evaluate.py:
- Run in two modes: `--quick` (fast screening, ≤30s) and default (full eval)
- ALWAYS write artifacts/results.json with this exact schema:
  {"schema_version": 1, "primary_metric": {"name": "<METRIC>", "value": <float>},
   "secondary_metrics": {}, "seed": 42}
- Import tunable constants from src.config (e.g. `from src.config import MODEL, LR`)
- sys.path.insert(0, str(pathlib.Path(__file__).parent.parent)) before the import
- Handle ImportError cleanly with a helpful message and sys.exit(1)
- Quick mode: for ML training tasks use inference latency only (write metric=0.0)
- Full mode: run actual training/inference, write the real metric value
- Print progress so the user knows what's happening
- Never hardcode paths outside of artifacts/ and /tmp/

RULES FOR src/config.py:
- Only tunable constants (MODEL_VARIANT, LR, EPOCHS, etc.) with type annotations
- One constant per line with a comment explaining the range/options
- No functions, no imports, no classes
- Constants must match what benchmarks/evaluate.py imports

Produce realistic, runnable Python. If the framework is unknown, use a sensible
stub that the user can fill in — but never produce broken syntax.
"""


def _build_prompt(project: Project, scan: RepoScan) -> str:
    lines: list[str] = []

    lines.append(f"## Objective\n{project.objective or 'Not specified'}\n")

    # Infer primary metric from objective text
    obj = (project.objective or "").lower()
    if any(w in obj for w in ("map", "mAP", "detection", "yolo", "object detect")):
        metric_hint = "map50 (maximize)"
    elif any(w in obj for w in ("f1", "precision", "recall", "classification")):
        metric_hint = "f1 (maximize)"
    elif any(w in obj for w in ("rmse", "mse", "error", "loss", "regression")):
        metric_hint = "rmse (minimize)"
    elif any(w in obj for w in ("accuracy", "acc", "correct")):
        metric_hint = "accuracy (maximize)"
    elif any(w in obj for w in ("rouge", "bleu", "nlp", "text")):
        metric_hint = "rouge_l (maximize)"
    else:
        metric_hint = "score (maximize) — infer from context"
    lines.append(f"## Inferred primary metric\n{metric_hint}\n")

    lines.append("## Repository scan\n")
    lines.append(f"- Path: {scan.repo_path}")
    lines.append(f"- Compatibility: {scan.compatibility.value}")
    if scan.python.package_name:
        lines.append(f"- Package: {scan.python.package_name}")
    if scan.python.dependencies:
        deps = scan.python.dependencies[:20]
        lines.append(f"- Dependencies: {', '.join(deps)}")
    if scan.python.requirements_files:
        lines.append(f"- Requirements files: {', '.join(scan.python.requirements_files)}")
    if scan.benchmark_candidates:
        lines.append(f"- Existing benchmark scripts: {', '.join(scan.benchmark_candidates)}")
    if scan.suggested_editable_paths:
        lines.append(f"- Suggested editable paths: {', '.join(scan.suggested_editable_paths)}")
    if scan.readme.title:
        lines.append(f"- README title: {scan.readme.title}")
    if scan.readme.excerpt:
        lines.append(f"- README excerpt: {scan.readme.excerpt[:400]}")
    if scan.keywords:
        lines.append(f"- Keywords: {', '.join(scan.keywords[:15])}")

    lines.append("\n## Task")
    lines.append(
        "Write benchmarks/evaluate.py and src/config.py for this project.\n"
        "The config should expose the most impactful tunable parameters for the objective.\n"
        "Use the existing dependencies — do not add new ones unless unavoidable."
    )

    return "\n".join(lines)


def _extract_tag(text: str, tag: str) -> str | None:
    m = re.search(rf"<{tag}>(.*?)</{tag}>", text, re.DOTALL)
    if m:
        content = m.group(1).strip()
        # Strip optional markdown fences
        content = re.sub(r"^```(?:python)?\s*\n?", "", content, flags=re.MULTILINE)
        content = re.sub(r"\n?```\s*$", "", content, flags=re.MULTILINE)
        return content.strip()
    return None


def generate_eval_files(
    project: Project,
    scan: RepoScan,
    provider: AiProvider,
) -> tuple[str, str]:
    """Call AI and return (evaluate_py_content, config_py_content).

    Raises ValueError with a descriptive message if parsing fails.
    """
    user_prompt = _build_prompt(project, scan)
    raw = provider.generate(_SYSTEM, user_prompt, max_tokens=8192)

    eval_script = _extract_tag(raw, "eval_script")
    config = _extract_tag(raw, "config")

    if eval_script is None:
        raise ValueError(
            "AI response did not contain <eval_script>…</eval_script>. "
            "Try again or use --provider to switch models."
        )
    if config is None:
        raise ValueError(
            "AI response did not contain <config>…</config>. "
            "Try again or use --provider to switch models."
        )

    return eval_script, config
