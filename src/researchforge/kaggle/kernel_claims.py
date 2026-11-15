"""The leaderboard score a public Kaggle notebook claims for itself.

Kaggle's API exposes no score for kernels, so the only evidence is what the
author wrote: the title ("CASMI26 LB 0.480") or the notebook itself ("**Public
LB: 0.477**"). These are *claims*, recorded with where they came from and a
short quote so a reader can check them. They are never treated as measurements.

The notebook is downloaded to a temporary directory, scanned and deleted: only
the number and a short quote are kept, never the code.
"""

from __future__ import annotations

import json
import re
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from researchforge.kaggle.notebook_baseline import claimed_lb_score

# A labelled leaderboard figure inside a notebook: "Public LB: 0.477",
# "LB = 0.455", "leaderboard score 0.46", "**LB 0.472**".
_LABELLED = re.compile(
    r"(?i)\b(?:public\s+)?(?:LB|leaderboard(?:\s+score)?|public\s+score)\b"
    r"[\s*_`:=\-]{0,6}([0-9]+\.[0-9]{2,})"
)
SNIPPET_CHARS = 120


def extract_claim(markdown: str, code: str) -> tuple[float, str] | None:
    """The first labelled LB claim — headline markdown first, then code comments/prints."""
    for text in (markdown, code):
        m = _LABELLED.search(text or "")
        if m:
            start = max(0, m.start() - 40)
            snippet = " ".join(text[start : m.end() + 40].split())
            return float(m.group(1)), snippet[:SNIPPET_CHARS]
    return None


def _notebook_text(path: Path) -> tuple[str, str]:
    raw = path.read_text(encoding="utf-8", errors="replace")
    if path.suffix != ".ipynb":
        return "", raw
    try:
        nb = json.loads(raw)
    except ValueError:
        return "", raw
    md: list[str] = []
    code: list[str] = []
    for cell in nb.get("cells", []):
        src = cell.get("source", "")
        text = "".join(src) if isinstance(src, list) else str(src)
        (md if cell.get("cell_type") == "markdown" else code).append(text)
    return "\n".join(md), "\n".join(code)


def notebook_claim(ref: str, timeout: float = 60.0) -> dict[str, Any] | None:
    """Download `ref` to a temp dir, read its claim, delete it. None if it has none."""
    with tempfile.TemporaryDirectory(prefix="rf-kernel-") as tmp:
        try:
            res = subprocess.run(
                ["kaggle", "kernels", "pull", ref, "-p", tmp],
                capture_output=True,
                text=True,
                timeout=timeout,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired):
            return None
        if res.returncode != 0:
            return None
        files = [p for p in Path(tmp).iterdir() if p.suffix in (".ipynb", ".py", ".r", ".R")]
        if not files:
            return None
        found = extract_claim(*_notebook_text(files[0]))
    if found is None:
        return None
    value, snippet = found
    return {"claimed_score": value, "claim_source": "notebook", "claim_snippet": snippet}


def title_claim(title: str) -> dict[str, Any] | None:
    value = claimed_lb_score(title)
    if value is None:
        return None
    return {"claimed_score": value, "claim_source": "title", "claim_snippet": title[:SNIPPET_CHARS]}
