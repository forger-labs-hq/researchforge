"""Recognising a hypothesis the project has already tried.

Re-synthesis runs against the same papers every round, so the AI will happily
propose an idea that was measured two rounds ago.  Comparison here is lexical —
shared content words between two hypotheses' titles and claims — which catches
restatements of the same idea, not every paraphrase of it.  A borderline pair
is therefore kept, not dropped: re-running an experiment wastes a round, while
discarding a genuinely new idea loses it for good.
"""

from __future__ import annotations

import re

from researchforge.domain.hypothesis import Hypothesis

DUPLICATE_THRESHOLD = 0.6

MIN_TERM_LENGTH = 3

# Words that say nothing about which idea a hypothesis describes. Kept small
# and general: an aggressive list would make unrelated hypotheses look alike.
STOPWORDS = frozenset(
    {
        "and",
        "are",
        "because",
        "but",
        "can",
        "for",
        "from",
        "how",
        "into",
        "its",
        "may",
        "more",
        "not",
        "should",
        "than",
        "that",
        "the",
        "their",
        "then",
        "this",
        "will",
        "with",
        "without",
        "would",
    }
)


def terms(text: str) -> set[str]:
    """The content words of a text, lowercased and stripped of punctuation."""
    return {
        token
        for token in re.findall(r"[a-z0-9]+", text.lower())
        if len(token) >= MIN_TERM_LENGTH and token not in STOPWORDS
    }


def overlap(left: str, right: str) -> float:
    """Jaccard overlap of two texts' content words, 0.0 when either is empty."""
    left_terms = terms(left)
    right_terms = terms(right)
    if not left_terms or not right_terms:
        return 0.0
    return len(left_terms & right_terms) / len(left_terms | right_terms)


def signature(hypothesis: Hypothesis) -> str:
    """The text two hypotheses are compared on: what it claims, not why."""
    return f"{hypothesis.title} {hypothesis.claim}"


def find_duplicate(
    candidate: Hypothesis,
    existing: list[Hypothesis],
    threshold: float = DUPLICATE_THRESHOLD,
) -> Hypothesis | None:
    """The stored hypothesis this candidate restates, or None.

    When several are close the most similar one is named, so the warning points
    at the best explanation of why the candidate was dropped.
    """
    scored = [
        (overlap(signature(candidate), signature(stored)), stored) for stored in existing
    ]
    matches = [(score, stored) for score, stored in scored if score >= threshold]
    if not matches:
        return None
    return max(matches, key=lambda pair: (pair[0], pair[1].hypothesis_id))[1]
