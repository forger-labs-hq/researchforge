"""Sizing the context exchanged when an IDE drives the loop.

When planning happens in Claude Code or Cursor, the model call is made by the
IDE. ResearchForge never sees the request, the response, or the provider's token
count — so there is nothing to meter. What it does know exactly is how much
context it handed over (the exported `context.json`) and how much came back (the
plan and its patches).

That is worth recording: a team asking "what does this loop cost us" should not
see a blank page just because the spend moved to a subscription. But it is an
*estimate*, and it is kept structurally separate from metered counts — recorded
with `estimated=True`, never converted to dollars, and labelled wherever it is
shown. An estimate presented as a measurement is worse than no number at all.
"""

from __future__ import annotations

from pathlib import Path

from researchforge.ai.usage import AiCall, Usage

#: Characters per token. Deliberately crude: the point is an order of magnitude
#: for "how much context moved", not a billing figure. Roughly right for English
#: prose and source code across the common tokenizers.
CHARS_PER_TOKEN = 4

IDE_PROVIDER = "ide"
IDE_MODEL = "ide-session"


def _tokens(text: str) -> int:
    return len(text) // CHARS_PER_TOKEN


def _read(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""


def size_handshake(
    context_path: Path, plan_path: Path, patches_dir: Path | None = None
) -> AiCall | None:
    """Estimate the context volume of one IDE planning exchange.

    Returns None when there is nothing to size — a plan imported from a file the
    user wrote by hand, with no exported context behind it, was not an IDE
    exchange and must not be recorded as one.
    """
    context = _read(context_path)
    if not context:
        return None

    produced = _read(plan_path)
    if patches_dir is not None and patches_dir.is_dir():
        for patch in sorted(patches_dir.glob("*.patch")):
            produced += _read(patch)

    usage = Usage(input_tokens=_tokens(context), output_tokens=_tokens(produced))
    if usage.total == 0:
        return None

    return AiCall(
        purpose="planning",
        provider=IDE_PROVIDER,
        model=IDE_MODEL,
        usage=usage,
        estimated=True,
    )
