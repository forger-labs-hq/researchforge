"""Token accounting: capture, attribution, and pricing.

Pricing is the part that can be confidently wrong, so the rate-matching and the
treatment of models with no rate at all get the most attention here.
"""

from __future__ import annotations

import pytest

from researchforge.ai.usage import AiCall, Usage, purpose, record_call, recording
from researchforge.config.settings import ModelPrice
from researchforge.reporting.economics import price_of, token_spend

PRICES = {
    "gpt-4o": ModelPrice(input=2.50, output=10.0),
    "gpt-4o-mini": ModelPrice(input=0.15, output=0.60),
    "claude-opus-4": ModelPrice(input=15.0, output=75.0),
}


class TestRecording:
    def test_calls_made_inside_the_block_are_captured(self) -> None:
        with recording() as ledger:
            record_call("planning", "openai", "gpt-4o", Usage(1000, 500))

        assert ledger.calls[0].model == "gpt-4o"
        assert (ledger.input_tokens, ledger.output_tokens) == (1000, 500)
        assert ledger.total_tokens == 1500

    def test_recording_nothing_is_not_an_error(self) -> None:
        """Most of the codebase never opens a ledger and must not pay for it."""
        record_call("planning", "openai", "gpt-4o", Usage(1000, 500))  # no block

        with recording() as ledger:
            pass

        assert ledger.calls == []

    def test_the_purpose_label_comes_from_the_caller(self) -> None:
        with recording() as ledger, purpose("synthesis"):
            record_call("synthesis", "anthropic", "claude-opus-4-5", Usage(10, 20))

        assert ledger.calls[0].purpose == "synthesis"

    def test_an_inner_block_does_not_leak_into_the_outer_one(self) -> None:
        """ "What did this cost" must not silently include a nested total."""
        with recording() as outer:
            record_call("planning", "openai", "gpt-4o", Usage(100, 0))
            with recording() as inner:
                record_call("synthesis", "openai", "gpt-4o", Usage(999, 0))

        assert outer.total_tokens == 100
        assert inner.total_tokens == 999

    def test_the_purpose_reverts_after_the_block(self) -> None:
        with recording() as ledger:
            with purpose("synthesis"):
                record_call("synthesis", "openai", "gpt-4o", Usage(1, 1))
            record_call("other", "openai", "gpt-4o", Usage(1, 1))

        assert [c.purpose for c in ledger.calls] == ["synthesis", "other"]


class TestPricing:
    def test_the_longest_matching_prefix_wins(self) -> None:
        """Otherwise the mini model gets billed at the full model's rate."""
        assert price_of("gpt-4o-mini-2026-01", PRICES) == PRICES["gpt-4o-mini"]
        assert price_of("gpt-4o-2026-01", PRICES) == PRICES["gpt-4o"]

    def test_a_dated_release_is_priced_by_its_family(self) -> None:
        assert price_of("claude-opus-4-5-20251101", PRICES) == PRICES["claude-opus-4"]

    def test_an_unknown_model_has_no_rate(self) -> None:
        assert price_of("some-local-model", PRICES) is None

    def test_tokens_are_converted_at_the_published_rate(self) -> None:
        spend = token_spend(
            [AiCall("planning", "openai", "gpt-4o", Usage(1_000_000, 100_000))],
            PRICES,
        )

        assert spend.usd == pytest.approx(2.50 + 1.0)
        assert spend.total_tokens == 1_100_000

    def test_an_unpriced_model_is_named_rather_than_billed_at_zero(self) -> None:
        """A zero that means "unknown" reads as "free" unless it is called out."""
        spend = token_spend(
            [
                AiCall("planning", "openai", "gpt-4o", Usage(1_000_000, 0)),
                AiCall("planning", "ollama", "some-local-model", Usage(500_000, 0)),
            ],
            PRICES,
        )

        assert spend.usd == pytest.approx(2.50)
        assert spend.unpriced_models == ["some-local-model"]
        assert spend.fully_priced is False
        # The tokens still count even though the dollars do not.
        assert spend.total_tokens == 1_500_000

    def test_a_model_that_reported_no_tokens_is_not_called_unpriced(self) -> None:
        """Nothing to price is not the same as a missing rate."""
        spend = token_spend([AiCall("planning", "ollama", "some-local-model", Usage(0, 0))], PRICES)

        assert spend.unpriced_models == []
        assert spend.fully_priced is True
        assert spend.calls == 1

    def test_tokens_are_grouped_by_purpose_biggest_first(self) -> None:
        spend = token_spend(
            [
                AiCall("planning", "openai", "gpt-4o", Usage(100, 0)),
                AiCall("synthesis", "openai", "gpt-4o", Usage(900, 0)),
                AiCall("planning", "openai", "gpt-4o", Usage(200, 0)),
            ],
            PRICES,
        )

        assert list(spend.by_purpose.items()) == [("synthesis", 900), ("planning", 300)]

    def test_an_ide_estimate_is_never_priced(self) -> None:
        """Those tokens were spent in someone's editor, on an unknown tokenizer."""
        spend = token_spend(
            [AiCall("planning", "ide", "ide-session", Usage(50_000, 5_000), estimated=True)],
            PRICES,
        )

        assert spend.usd == 0.0
        assert spend.estimated_tokens == 55_000
        assert spend.estimated_calls == 1
        assert spend.has_estimates is True

    def test_estimates_stay_out_of_the_metered_total(self) -> None:
        """Otherwise a sized number silently inflates a measured one."""
        spend = token_spend(
            [
                AiCall("planning", "openai", "gpt-4o", Usage(1_000_000, 0)),
                AiCall("planning", "ide", "ide-session", Usage(999_999, 0), estimated=True),
            ],
            PRICES,
        )

        assert spend.total_tokens == 1_000_000
        assert spend.estimated_tokens == 999_999
        assert spend.usd == pytest.approx(2.50)

    def test_an_unknown_ide_model_is_not_reported_as_unpriced(self) -> None:
        """ "Unpriced" means a rate is missing; an estimate is never priced at all."""
        spend = token_spend(
            [AiCall("planning", "ide", "ide-session", Usage(100, 0), estimated=True)],
            PRICES,
        )

        assert spend.unpriced_models == []
        assert spend.fully_priced is True

    def test_no_calls_costs_nothing_and_claims_nothing(self) -> None:
        spend = token_spend([], PRICES)

        assert (spend.calls, spend.total_tokens, spend.usd) == (0, 0, 0.0)
        assert spend.fully_priced is True
