"""Sizing an IDE planning exchange.

The figure is an estimate by construction, so what these tests protect is that
it is always *labelled* as one and never produced from nothing.
"""

from __future__ import annotations

from pathlib import Path

from researchforge.experiments.handshake_usage import CHARS_PER_TOKEN, size_handshake


def _write(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


class TestSizingAnExchange:
    def test_context_out_and_plan_back_are_both_counted(self, tmp_path: Path) -> None:
        context = _write(tmp_path / "context.json", "c" * (CHARS_PER_TOKEN * 100))
        plan = _write(tmp_path / "plan.yaml", "p" * (CHARS_PER_TOKEN * 20))

        call = size_handshake(context, plan)

        assert call is not None
        assert call.usage.input_tokens == 100
        assert call.usage.output_tokens == 20

    def test_patches_count_as_part_of_what_came_back(self, tmp_path: Path) -> None:
        context = _write(tmp_path / "context.json", "c" * (CHARS_PER_TOKEN * 100))
        plan = _write(tmp_path / "plan.yaml", "p" * (CHARS_PER_TOKEN * 20))
        patches = tmp_path / "patches"
        _write(patches / "a.patch", "d" * (CHARS_PER_TOKEN * 30))
        _write(patches / "b.patch", "d" * (CHARS_PER_TOKEN * 10))

        call = size_handshake(context, plan, patches)

        assert call is not None
        assert call.usage.output_tokens == 60

    def test_the_result_is_always_marked_as_an_estimate(self, tmp_path: Path) -> None:
        """The single most important property here."""
        context = _write(tmp_path / "context.json", "c" * 400)
        plan = _write(tmp_path / "plan.yaml", "p" * 400)

        call = size_handshake(context, plan)

        assert call is not None
        assert call.estimated is True
        assert call.provider == "ide"

    def test_a_hand_written_plan_with_no_exported_context_is_not_an_exchange(
        self, tmp_path: Path
    ) -> None:
        """Nobody's IDE spent tokens, so there is nothing to record."""
        plan = _write(tmp_path / "plan.yaml", "p" * 400)

        assert size_handshake(tmp_path / "context.json", plan) is None

    def test_an_empty_exchange_records_nothing(self, tmp_path: Path) -> None:
        context = _write(tmp_path / "context.json", "")
        plan = _write(tmp_path / "plan.yaml", "")

        assert size_handshake(context, plan) is None

    def test_a_missing_patches_directory_is_not_an_error(self, tmp_path: Path) -> None:
        context = _write(tmp_path / "context.json", "c" * 400)
        plan = _write(tmp_path / "plan.yaml", "p" * 400)

        call = size_handshake(context, plan, tmp_path / "nope")

        assert call is not None
        assert call.usage.output_tokens == 100
