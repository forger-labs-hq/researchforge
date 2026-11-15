"""The shared hypothesis-pool ranking and the loops that draw from it."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest
from typer.testing import CliRunner

from researchforge.kaggle import hypothesis_pool as hp


@pytest.fixture(autouse=True)
def _isolated_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: home))
    monkeypatch.delenv("RESEARCHFORGE_OVERNIGHT_DIR", raising=False)
    monkeypatch.setenv("RESEARCHFORGE_NO_DASHBOARD", "1")


def _write(d: Path, pool: list[dict], scores: list[str] = ()) -> Path:
    d.mkdir(parents=True, exist_ok=True)
    (d / hp.POOL_FILE).write_text(json.dumps(pool))
    subs = [{"id": i, "publicscore": s} for i, s in enumerate(scores)]
    (d / "submissions.json").write_text(json.dumps({"submissions": subs}))
    return d


def _confidence(monkeypatch: pytest.MonkeyPatch, by_idea: dict[str, str]) -> None:
    from researchforge.intelligence import scanner

    monkeypatch.setattr(
        scanner, "tag_metric_confidence", lambda idea, files: by_idea.get(idea, "medium")
    )


class TestRanking:
    def test_order_user_then_claim_ahead_then_confidence_then_relevance(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _confidence(monkeypatch, {"gpu-only": "blind"})
        d = _write(
            tmp_path / ".overnight_c",
            [
                {"idea": "small claim", "source": "kernel_watcher", "claimed_score": 0.20},
                {"idea": "big claim", "source": "kernel_watcher", "claimed_score": 0.48},
                {"idea": "behind us", "source": "kernel_watcher", "claimed_score": 0.10},
                {
                    "idea": "repeat",
                    "source": "kernel_watcher",
                    "relevance": "repeats_failure",
                    "claimed_score": 0.90,
                },
                {"idea": "gpu-only", "source": "arxiv"},
                {"idea": "mine", "source": "user"},
                {"idea": "done", "source": "user", "tested": True},
                {"idea": "waiting", "source": "user"},
            ],
            scores=["0.166", "0.12", "pending"],
        )
        ranked = hp.rank_untested(d, "maximize", blocked={"waiting"})
        assert [h["idea"] for h in ranked] == [
            "mine",
            "repeat",  # claims the most over our best LB: a leaderboard lead outranks relevance
            "big claim",
            "small claim",
            "behind us",  # its claim is below our best, so it does not lead
            "gpu-only",  # blind: needs a Kaggle GPU run, so last
        ]
        big = ranked[2]
        assert big["claim_vs_our_best"] == pytest.approx(0.314)
        assert big["our_best_lb"] == 0.166
        assert big["metric_confidence"] == "medium" and big["confidence_note"]

    def test_lower_is_better_competitions_flip_the_gap(self, tmp_path: Path) -> None:
        d = _write(tmp_path / ".overnight_c", [], scores=["0.30", "0.25"])
        assert hp.our_best_public_lb(d, "minimize") == 0.25
        assert hp.claim_gap({"claimed_score": 0.20}, 0.25, "minimize") == pytest.approx(0.05)
        assert hp.claim_gap({"claimed_score": 0.20}, None, "minimize") is None

    def test_describe_marks_claims_as_unverified(self) -> None:
        line = hp.describe(
            {
                "idea": "Dual SOTA",
                "source": "kernel_watcher",
                "kernel_ref": "a/b",
                "claimed_score": 0.482,
                "claim_vs_our_best": 0.316,
                "our_best_lb": 0.166,
            }
        )
        assert "notebook a/b" in line and "+0.3160 vs our best 0.166" in line
        assert "unverified" in line

    def test_mark_tested_matches_the_exact_idea(self, tmp_path: Path) -> None:
        d = _write(tmp_path / ".overnight_c", [{"idea": "Try LightGBM dart"}, {"idea": "Other"}])
        assert hp.mark_tested(d, {"hypothesis": "try lightgbm dart"}, 7, "improved")
        pool = hp.load_pool(d)
        assert pool[0]["tested_in_round"] == 7 and pool[0]["tested_outcome"] == "improved"
        assert not pool[1].get("tested")
        assert hp.mark_tested(d, {"hypothesis": "nothing like it"}, 8, "failed") is None


class TestRepoAutorun:
    def test_leads_come_from_the_primary_overnight_dir_next_to_the_db(self, tmp_path: Path) -> None:
        root = tmp_path / "proj"
        _write(
            root / ".overnight_comp",
            [{"idea": "lead A", "source": "kernel_watcher", "kernel_ref": "x/y"}],
        )
        _write(root / ".overnight_comp.night1-backup", [{"idea": "stale copy"}])
        (root / ".researchforge").mkdir()
        conn = sqlite3.connect(root / ".researchforge" / "researchforge.db")
        try:
            assert hp.overnight_dirs_for(conn) == [(root / ".overnight_comp").resolve()]
            lines = hp.leads_instructions(conn)
            assert "KAGGLE LEADS" in lines[0] and "not evidence" in lines[0]
            assert any("lead A" in ln and "notebook x/y" in ln for ln in lines)
            assert not any("stale copy" in ln for ln in lines)
            summary = hp.pool_summary(conn)
            assert summary[0]["untested"] == 1
        finally:
            conn.close()

    def test_no_overnight_dir_means_no_leads(self, tmp_path: Path) -> None:
        conn = sqlite3.connect(":memory:")
        try:
            assert hp.leads_instructions(conn) == []
        finally:
            conn.close()


class TestHistoryAndResume:
    def test_load_rounds_reads_the_log_and_tolerates_none(self, tmp_path: Path) -> None:
        from researchforge.kaggle.experiment_loop import OvernightExperimentLoop, load_rounds

        assert load_rounds(tmp_path / "missing.db") == []
        loop = OvernightExperimentLoop.__new__(OvernightExperimentLoop)
        loop.db_path = tmp_path / "experiment_log.db"
        loop._init_database()
        with sqlite3.connect(loop.db_path) as c:
            c.execute(
                "INSERT INTO experiments (round_num, patch_description, patch_code, "
                "full_passed, full_metric, full_improved) VALUES (1, 'a', '', 1, 0.2, 1), "
                "(2, 'b', '', 1, 0.1, 0)"
            )
            c.execute(
                "INSERT INTO experiments (round_num, patch_description, patch_code, "
                "rejected_reason) VALUES (3, 'c', '', 'crashed')"
            )
        rounds = load_rounds(loop.db_path)
        assert [(r.round_num, r.outcome) for r in rounds] == [
            (1, "improved"),
            (2, "no_improvement"),
            (3, "failed"),
        ]

    def test_overnight_resumes_a_set_up_project_instead_of_redoing_setup(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from researchforge.cli import app
        from researchforge.kaggle.orchestrator import OvernightOrchestrator

        d = _write(tmp_path / ".overnight_comp", [{"idea": "next idea"}])
        (d / "overnight_setup.json").write_text(
            json.dumps({"competition_slug": "comp", "baseline_result": {"cv_score": 0.5}})
        )
        monkeypatch.chdir(tmp_path)

        def _no_setup(*_a: object, **_k: object) -> None:
            raise AssertionError("setup must not run for a set-up project")

        monkeypatch.setattr(OvernightOrchestrator, "run_full_flow", _no_setup)
        result = CliRunner().invoke(app, ["kaggle", "overnight", "comp"])
        assert result.exit_code == 0, result.output
        assert "Resuming comp" in result.output
        assert "Next: researchforge kaggle prepare-round comp" in result.output
        assert "1 untested ideas" in result.output

        status = CliRunner().invoke(app, ["kaggle", "status", "comp", "--json"])
        payload = json.loads(status.output)
        assert payload["set_up"] is True
        assert payload["next_command"] == "researchforge kaggle prepare-round comp"


class TestApiKeyLoop:
    def test_proposer_prompt_carries_the_ranked_pool_and_reads_back_the_idea(
        self, tmp_path: Path
    ) -> None:
        from types import SimpleNamespace

        from researchforge.kaggle.patch_proposer import PatchProposer

        d = _write(
            tmp_path / ".overnight_c",
            [
                {"idea": "plain idea", "source": "arxiv"},
                {
                    "idea": "notebook lead",
                    "source": "kernel_watcher",
                    "kernel_ref": "a/b",
                    "claimed_score": 0.48,
                },
            ],
            scores=["0.166"],
        )
        proposer = PatchProposer.__new__(PatchProposer)
        proposer.setup = SimpleNamespace(artifacts_dir=d, contract=None)  # type: ignore[assignment]
        section = proposer._pool_section()
        assert section.index("notebook lead") < section.index("plain idea")
        assert "+0.3140 vs our best 0.166" in section

        class _Provider:
            def generate(self, system: str, prompt: str, max_tokens: int) -> str:
                assert '"hypothesis"' in system
                return json.dumps(
                    {"title": "t", "code_changes": "x = 1", "hypothesis": "notebook lead"}
                )

        proposer._provider = _Provider()  # type: ignore[assignment]
        assert proposer._call_ai("ctx", 3).hypothesis == "notebook lead"
