"""Recognising a hypothesis that restates one already on record."""

from researchforge.domain.hypothesis import Hypothesis, Level, NoveltyConfidence
from researchforge.research.hypothesis_dedup import (
    find_duplicate,
    overlap,
    signature,
    terms,
)


def _hypothesis(hypothesis_id: str, title: str, claim: str) -> Hypothesis:
    return Hypothesis(
        hypothesis_id=hypothesis_id,
        title=title,
        claim=claim,
        rationale="Fixture rationale.",
        feasibility=Level.HIGH,
        estimated_effort=Level.LOW,
        novelty_confidence=NoveltyConfidence.UNKNOWN,
        proposed_experiment="Run it and measure.",
    )


ROUTING = _hypothesis(
    "hyp-001",
    "Entropy routing cuts inference cost",
    "Routing queries by entropy threshold reduces average inference cost.",
)
CALIBRATION = _hypothesis(
    "hyp-002",
    "Temperature scaling improves calibration",
    "Calibrated confidence scores beat raw logits for downstream selection.",
)


class TestTerms:
    def test_lowercases_and_splits_on_punctuation(self) -> None:
        assert terms("Entropy-threshold routing!") == {"entropy", "threshold", "routing"}

    def test_drops_short_tokens(self) -> None:
        assert terms("a to be or not") == set()

    def test_drops_stopwords(self) -> None:
        assert terms("the routing and the cost") == {"routing", "cost"}

    def test_the_length_rule_applies_to_numbers_too(self) -> None:
        """Two variants of one idea differing only in a target are still one idea."""
        assert terms("reduces cost by 15 percent") == terms("reduces cost by 30 percent")


class TestOverlap:
    def test_identical_text_is_one(self) -> None:
        assert overlap("entropy routing cost", "entropy routing cost") == 1.0

    def test_disjoint_text_is_zero(self) -> None:
        assert overlap("entropy routing", "batch normalization") == 0.0

    def test_empty_text_is_zero(self) -> None:
        assert overlap("", "entropy routing") == 0.0

    def test_wording_changes_do_not_break_the_match(self) -> None:
        assert overlap("the entropy routing cost", "entropy routing cost") == 1.0

    def test_partial_sharing_is_between(self) -> None:
        score = overlap("entropy routing cost", "entropy routing latency")
        assert 0.0 < score < 1.0


class TestSignature:
    def test_covers_title_and_claim(self) -> None:
        text = signature(ROUTING)
        assert "Entropy routing cuts inference cost" in text
        assert "reduces average inference cost" in text

    def test_excludes_rationale(self) -> None:
        assert "Fixture rationale" not in signature(ROUTING)


class TestFindDuplicate:
    def test_no_existing_hypotheses_is_no_duplicate(self) -> None:
        assert find_duplicate(ROUTING, []) is None

    def test_an_unrelated_candidate_is_kept(self) -> None:
        assert find_duplicate(CALIBRATION, [ROUTING]) is None

    def test_a_restatement_is_caught(self) -> None:
        restated = _hypothesis(
            "hyp-009",
            "Entropy routing reduces inference cost",
            "Routing each query by an entropy threshold cuts average inference cost.",
        )
        found = find_duplicate(restated, [CALIBRATION, ROUTING])
        assert found is not None
        assert found.hypothesis_id == "hyp-001"

    def test_the_same_hypothesis_again_is_caught(self) -> None:
        found = find_duplicate(ROUTING, [ROUTING])
        assert found is not None
        assert found.hypothesis_id == "hyp-001"

    def test_a_borderline_candidate_is_kept_when_the_threshold_is_strict(self) -> None:
        """Ties go to keeping the idea: a wasted round beats a lost hypothesis."""
        nearby = _hypothesis(
            "hyp-009",
            "Entropy routing raises quality",
            "Routing by entropy threshold improves answer quality on hard queries.",
        )
        assert find_duplicate(nearby, [ROUTING], threshold=0.95) is None

    def test_the_closest_match_is_named(self) -> None:
        looser = _hypothesis(
            "hyp-003",
            "Routing cuts cost",
            "Routing reduces average inference cost.",
        )
        exact = _hypothesis(
            "hyp-004",
            "Entropy routing cuts inference cost",
            "Routing queries by entropy threshold reduces average inference cost.",
        )
        found = find_duplicate(ROUTING, [looser, exact])
        assert found is not None
        assert found.hypothesis_id == "hyp-004"
