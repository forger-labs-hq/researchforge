"""One history per competition, and public notebooks run as rounds without the person."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from researchforge.kaggle import dashboard_state, history_import, hypothesis_pool
from researchforge.kaggle.experiment_loop import OvernightExperimentLoop, load_rounds

SLUG = "comp"


def _overnight(root: Path) -> Path:
    d = root / f".overnight_{SLUG}"
    d.mkdir()
    contract = {"objective": {"primary_metric": {"name": "score", "direction": "maximize"}}}
    (d / "overnight_setup.json").write_text(
        json.dumps({"competition_slug": SLUG, "selected_gpu": "kaggle_gpu", "contract": contract})
    )
    (d / "pending_hypotheses.json").write_text(json.dumps([{"idea": "tilted bands"}]))
    loop = OvernightExperimentLoop.__new__(OvernightExperimentLoop)
    loop.db_path = d / "experiment_log.db"
    loop._init_database()
    return d


class TestMerge:
    def test_repo_experiments_join_the_one_history_without_being_compared(
        self, validated_project: Path, isolated_project_dir: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        home = isolated_project_dir / "home"
        home.mkdir()
        monkeypatch.setattr(Path, "home", classmethod(lambda cls: home))
        monkeypatch.setenv("RESEARCHFORGE_NO_DASHBOARD", "1")
        d = _overnight(isolated_project_dir)
        repo_db = isolated_project_dir / ".researchforge" / "researchforge.db"
        before = repo_db.read_bytes()

        result = history_import.import_repo_history(d)
        assert result is not None and result["experiments"] == 2
        rows = {r["repo_id"]: r for r in history_import.imported_experiments(d)}
        assert set(rows) == {"exp-001", "exp-002"}
        assert rows["exp-002"]["status"] == "failed"
        assert repo_db.read_bytes() == before  # the repo project is only read
        assert history_import.import_repo_history(d)["unchanged"]  # type: ignore[index]
        assert load_rounds(d / "experiment_log.db") == []  # never a round, parent or "best"
        assert history_import.earlier_work(d)[0]["id"] == "exp-001"

        (p,) = dashboard_state.build_state(isolated_project_dir)["projects"]
        assert p["kinds"] == ["kaggle"] and p["repo"] is None
        assert {e["id"] for e in p["experiments"]} == {"exp-001", "exp-002"}
        assert all(e["source"] == "overnight" and e["imported"] for e in p["experiments"])
        assert p["best"] is None and "never best" in p["experiments"][0]["reason"]
        shown = {e["id"]: e for e in p["experiments"]}
        assert shown["exp-001"]["status"] == "complete"  # "improved" is the loop's own verdict
        assert "improved on its baseline" in shown["exp-001"]["reason"]

    def test_the_cli_merges_now(self, validated_project: Path, isolated_project_dir: Path) -> None:
        from researchforge.cli import app

        _overnight(isolated_project_dir)
        out = CliRunner().invoke(app, ["kaggle", "merge-history", SLUG])
        assert out.exit_code == 0, out.output
        assert "2 repo experiment(s)" in out.output and "not comparable" in out.output


class TestSkipIdea:
    def test_a_skipped_idea_leaves_the_ranking_with_its_reason(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from researchforge.cli import app

        d = _overnight(tmp_path)
        monkeypatch.chdir(tmp_path)
        bad = CliRunner().invoke(app, ["kaggle", "skip-idea", SLUG, "tilted bands"])
        assert bad.exit_code == 1  # a reason is required
        out = CliRunner().invoke(
            app, ["kaggle", "skip-idea", SLUG, "Tilted bands", "--reason", "needs a local build"]
        )
        assert out.exit_code == 0, out.output
        assert hypothesis_pool.rank_untested(d, "maximize", blocked=set()) == []
        (h,) = hypothesis_pool.load_pool(d)
        assert h["skipped"] and h["skip_reason"] == "needs a local build"
        shown = dashboard_state._hypothesis_from_pool(0, h)
        assert shown["status"] == "rejected" and "needs a local build" in shown["detail"]


NOTEBOOK = {
    "cells": [
        {"cell_type": "markdown", "source": ["# Title"]},
        {
            "cell_type": "code",
            "source": ["!pip install -q lib\n", "import os\n", "%matplotlib inline\n"],
        },
        {"cell_type": "code", "source": "%%writefile helper.py\nX = 1\n"},
        {"cell_type": "code", "source": ["for i in range(2):\n", "    !echo hi\n", "display(i)\n"]},
    ]
}


class TestProposeFromKernel:
    def _kernel(self, root: Path) -> Path:
        k = root / "reference_kernels" / "nb-480"
        k.mkdir(parents=True)
        (k / "nb-480.ipynb").write_text(json.dumps(NOTEBOOK))
        (k / "kernel-metadata.json").write_text(
            json.dumps({"dataset_sources": ["a/scorer"], "kernel_sources": ["b/feats"]})
        )
        return k

    def test_a_notebook_becomes_a_runnable_proposal(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from researchforge.cli import app

        d = _overnight(tmp_path)
        self._kernel(tmp_path)
        (tmp_path / "score.py").write_text("print('score: 0.5')\n")
        monkeypatch.chdir(tmp_path)
        out = CliRunner().invoke(
            app,
            [
                "kaggle", "propose-from-kernel", SLUG, "owner/nb-480",
                "--title", "Adopt the 0.480 notebook", "--parent", "R2",
                "--append", "score.py", "--dataset", "c/candidates", "--hypothesis", "nb 480",
            ],
        )  # fmt: skip
        assert out.exit_code == 0, out.output
        assert "Notebook outputs attached: b/feats" in out.output
        prop = json.loads((d / "proposal.json").read_text())
        assert prop["dataset_sources"] == ["a/scorer", "c/candidates"]
        assert prop["kernel_sources"] == ["b/feats"]  # the round attaches them too
        assert (prop["parent_round_num"], prop["source_credit"]) == (2, "owner/nb-480")
        code = prop["code_changes"]
        assert "run('pip install -q lib', shell=True" in code
        assert "    __import__(\"subprocess\").run('echo hi'" in code  # indentation kept
        assert "with open('helper.py', 'w')" in code and "%matplotlib" not in code.split("pass")[0]
        assert code.rstrip().endswith("print('score: 0.5')")

    def test_a_broken_script_is_refused_and_nothing_is_written(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from researchforge.cli import app

        d = _overnight(tmp_path)
        self._kernel(tmp_path)
        (tmp_path / "bad.py").write_text("def broken(:\n")
        monkeypatch.chdir(tmp_path)
        args = ["kaggle", "propose-from-kernel", SLUG, "owner/nb-480", "--title", "x"]
        out = CliRunner().invoke(app, [*args, "--append", "bad.py"])
        assert out.exit_code == 1 and "does not parse" in out.output
        assert not (d / "proposal.json").exists()
