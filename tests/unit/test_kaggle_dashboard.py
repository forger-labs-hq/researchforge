"""Kaggle dashboard: discovery, live-state contract, sensors, producers, HTTP service."""

from __future__ import annotations

import json
import sqlite3
import threading
import urllib.error
import urllib.request
from pathlib import Path

import pytest

from researchforge.kaggle import dashboard_state, sensors, workspace


@pytest.fixture(autouse=True)
def _isolated_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: home))
    monkeypatch.delenv("RESEARCHFORGE_OVERNIGHT_DIR", raising=False)
    monkeypatch.delenv("RESEARCHFORGE_SCAN_ROOTS", raising=False)
    dashboard_state._LAUNCHD_CACHE.clear()
    return home


def _overnight(root: Path, slug: str, *, setup: bool = True, baseline: float | None = 0.5) -> Path:
    d = root / f".overnight_{slug}"
    d.mkdir(parents=True)
    if setup:
        (d / "overnight_setup.json").write_text(
            json.dumps(
                {
                    "competition_slug": slug,
                    "baseline_result": {"cv_score": baseline} if baseline is not None else {},
                    "ai_provider_mode": "claude_code",
                }
            )
        )
    return d


def _db(d: Path, rows: list[tuple], *, old_schema: bool = False) -> None:
    with sqlite3.connect(d / "experiment_log.db") as c:
        if old_schema:
            c.execute(
                "CREATE TABLE experiments (round_num INTEGER PRIMARY KEY, patch_description TEXT "
                "NOT NULL, "
                "patch_code TEXT NOT NULL, screening_passed INTEGER, screening_metric REAL, "
                "full_passed INTEGER, "
                "full_metric REAL, full_improved INTEGER, rejected_reason TEXT, started_at TEXT, "
                "completed_at TEXT)"
            )
            for r in rows:
                c.execute(
                    "INSERT INTO experiments (round_num, patch_description, patch_code, "
                    "full_metric, full_improved, "
                    "started_at, completed_at) VALUES (?,?,'',?,?,?,?)",
                    r,
                )
        else:
            from researchforge.kaggle.experiment_loop import OvernightExperimentLoop

            loop = OvernightExperimentLoop.__new__(OvernightExperimentLoop)
            loop.db_path = d / "experiment_log.db"
            loop._init_database()
            for r in rows:
                c.execute(
                    "INSERT INTO experiments (round_num, parent_round_num, branch_tag, "
                    "patch_description, "
                    "patch_code, full_metric, full_improved, started_at, completed_at) VALUES "
                    "(?,?,?,?,'',?,?,?,?)",
                    r,
                )


def _project(state: dict, location: Path) -> dict:
    return next(p for p in state["projects"] if p["location"] == str(location.resolve()))


# ---------------------------------------------------------------------------
# Discovery + registry + producer resolution
# ---------------------------------------------------------------------------


class TestWorkspace:
    def test_registry_reads_v1_and_keeps_several_dirs_per_slug(self, tmp_path: Path) -> None:
        a, b = _overnight(tmp_path / "a", "comp"), _overnight(tmp_path / "b", "comp")
        workspace.registry_path().parent.mkdir(parents=True)
        workspace.registry_path().write_text(json.dumps({"comp": str(a)}))  # legacy {slug: path}
        assert [e["artifacts_dir"] for e in workspace.load_registry()] == [str(a)]
        workspace.register_overnight_dir("comp", b)
        dirs = sorted(e["artifacts_dir"] for e in workspace.load_registry())
        assert dirs == sorted([str(a), str(b.resolve())])

    def test_discovers_nested_repos_worktrees_and_backups(self, tmp_path: Path) -> None:
        ws = tmp_path / "ws"
        _overnight(ws, "top")
        nested = _overnight(ws / "kaggle" / "proj", "deep")
        (ws / "kaggle" / "proj" / ".researchforge" / "worktrees" / "baseline").mkdir(parents=True)
        wt = _overnight(
            ws / "kaggle" / "proj" / ".researchforge" / "worktrees" / "baseline",
            "deep",
            setup=False,
        )
        bak = _overnight(ws / "old", "top.night1-backup")
        (ws / "node_modules" / "x").mkdir(parents=True)
        _overnight(ws / "node_modules" / "x", "ignored")

        locs, report = workspace.discover_locations(ws)
        found = {str(d) for loc in locs for d in loc.overnight_dirs}
        assert (
            str(nested.resolve()) in found
            and str(wt.resolve()) in found
            and str(bak.resolve()) in found
        )
        assert not any("ignored" in f for f in found)
        assert workspace.location_role(wt) == "worktree"
        assert workspace.location_role(bak) == "backup"
        nested_loc = next(loc for loc in locs if loc.root == nested.parent.resolve())
        assert nested_loc.repo_dir is not None  # repo project + overnight share one root
        assert report["scan_roots"] == [str(ws.resolve())]

    def test_registered_dirs_outside_cwd_are_discovered(self, tmp_path: Path) -> None:
        elsewhere = _overnight(tmp_path / "elsewhere" / "proj", "remote")
        workspace.register_overnight_dir("remote", elsewhere)
        cwd = tmp_path / "unrelated"
        cwd.mkdir()
        locs, _ = workspace.discover_locations(cwd)
        assert any(elsewhere.resolve() in loc.overnight_dirs for loc in locs)

    def test_hidden_roots_leave_the_view_and_take_their_worktrees(self, tmp_path: Path) -> None:
        ws = tmp_path / "ws"
        keep = _overnight(ws / "keep", "keep")
        old = _overnight(ws / "old", "old")
        wt_dir = ws / "old" / ".researchforge" / "worktrees" / "baseline"
        wt_dir.mkdir(parents=True)
        _overnight(wt_dir, "old", setup=False)
        workspace.atomic_write_json(
            workspace.dashboard_config_path(), {"hidden_roots": [str(ws / "old"), str(ws)]}
        )
        locs, report = workspace.discover_locations(ws)
        assert [loc.root for loc in locs] == [keep.parent.resolve()]
        assert old.exists()  # hiding never touches the files
        assert report["hidden_roots"] == sorted(str(p.resolve()) for p in (ws / "old", wt_dir))

    def test_resolve_prefers_set_up_project_over_watcher_only_copy(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        real = _overnight(tmp_path / "kaggle" / "proj", "comp")
        workspace.register_overnight_dir("comp", real)
        hub = tmp_path / "hub"
        _overnight(hub, "comp", setup=False)  # watcher-only state written from another cwd
        assert workspace.resolve_artifacts_dir("comp", hub) == real.resolve()
        # From inside the project (or a subdir) the local set-up dir wins.
        sub = real.parent / "src"
        sub.mkdir()
        assert workspace.resolve_artifacts_dir("comp", sub) == real.resolve()
        # Unknown slug → new dir in cwd.
        assert (
            workspace.resolve_artifacts_dir("new-comp", hub)
            == hub.resolve() / ".overnight_new-comp"
        )
        # launchd jobs pin the dir explicitly.
        monkeypatch.setenv("RESEARCHFORGE_OVERNIGHT_DIR", str(hub / ".overnight_comp"))
        assert workspace.resolve_artifacts_dir("comp", tmp_path) == hub / ".overnight_comp"

    def test_atomic_write_never_leaves_partial_files(self, tmp_path: Path) -> None:
        target = tmp_path / "out" / "x.json"
        workspace.atomic_write_json(target, {"a": 1})
        assert json.loads(target.read_text()) == {"a": 1}
        assert [p.name for p in target.parent.iterdir()] == ["x.json"]


# ---------------------------------------------------------------------------
# Live-state contract
# ---------------------------------------------------------------------------


class TestState:
    def test_old_schema_db_is_read_not_silently_empty(self, tmp_path: Path) -> None:
        d = _overnight(tmp_path, "old", baseline=0.6)
        _db(
            d,
            [
                (1, "ridge", 0.7, 1, "2026-01-01T10:00:00", "2026-01-01T10:05:00"),
                (2, "lasso", 0.65, 1, "2026-01-01T11:00:00", "2026-01-01T11:05:00"),
            ],
            old_schema=True,
        )
        p = _project(dashboard_state.build_state(tmp_path), d)
        assert [e["id"] for e in p["experiments"]] == ["R1", "R2"]
        assert all(e["parents"] == [] for e in p["experiments"])
        assert any("older schema" in i["message"] for i in p["issues"])
        assert p["best"] == {"id": "R1", "metric": 0.7}

    def test_dag_parents_inflight_and_hypothesis_links(self, tmp_path: Path) -> None:
        d = _overnight(tmp_path, "dag")
        _db(
            d,
            [
                (1, None, "base", "first", 0.55, 1, "2026-01-01T10:00:00", "2026-01-01T10:05:00"),
                (2, 1, "feat", "child", 0.58, 1, "2026-01-01T11:00:00", "2026-01-01T11:05:00"),
            ],
        )
        import os

        (d / "inflight_rounds.json").write_text(
            json.dumps(
                {
                    "3": {
                        "round_num": 3,
                        "title": "grandchild",
                        "parent_round_num": 2,
                        "pid": os.getpid(),
                        "stage": "full",
                    },
                    "4": {
                        "round_num": 4,
                        "title": "orphan",
                        "parent_round_num": 2,
                        "pid": 999999,
                        "stage": "screening",
                    },
                }
            )
        )
        (d / "pending_hypotheses.json").write_text(
            json.dumps(
                [
                    {
                        "idea": "first",
                        "source": "claude_observation",
                        "tested": True,
                        "tested_in_round": 1,
                    },
                    {
                        "idea": "later",
                        "source": "kernel_watcher",
                        "tested": False,
                        "injected_at": "2026-01-02T00:00:00",
                    },
                ]
            )
        )
        p = _project(dashboard_state.build_state(tmp_path), d)
        by = {e["id"]: e for e in p["experiments"]}
        assert by["R2"]["parents"] == ["R1"] and by["R3"]["parents"] == ["R2"]
        assert by["R3"]["status"] == "running" and by["R4"]["status"] == "stale"
        assert by["R1"]["hypothesis_ids"] == ["pool-0"]
        assert p["hypotheses"][0]["tested_by"] == ["R1"]
        assert p["state"] == "running"
        assert any("dead PID" in i["message"] for i in p["issues"])

    def test_node_patch_proposal_merge_candidate_and_stalled(self, tmp_path: Path) -> None:
        d = _overnight(tmp_path, "merge")
        _db(d, [])
        with sqlite3.connect(d / "experiment_log.db") as c:
            for num, tag, metric in ((1, "features", 0.61), (2, "model", 0.63), (3, "model", 0.62)):
                c.execute(
                    "INSERT INTO experiments (round_num, branch_tag, patch_description, "
                    "patch_code, "
                    "full_passed, full_metric, full_improved, started_at, completed_at) "
                    "VALUES (?,?,?,?,1,?,1,'2026-01-01T10:00:00','2026-01-01T10:05:00')",
                    (num, tag, f"r{num}", f"code for {num}", metric),
                )
        (d / "proposal_round_1.json").write_text(
            json.dumps(
                {"description": "add ratios", "rationale": "kernel X", "expected_impact": "+1%"}
            )
        )
        (d / "inflight_rounds.json").write_text(
            json.dumps({"4": {"round_num": 4, "title": "crashed", "pid": 999999}})
        )
        p = _project(dashboard_state.build_state(tmp_path), d)
        by = {e["id"]: e for e in p["experiments"]}
        assert by["R1"]["patch"] == "code for 1" and by["R1"]["proposal"]["rationale"] == "kernel X"
        assert by["R2"]["proposal"] is None
        # prepare-round's rule: best improved round per branch tag, top two.
        assert p["merge_candidates"][0]["parents"] == ["R2", "R1"]
        assert p["state"] == "stalled"

    def test_waiting_for_input_and_tokens_per_round(self, tmp_path: Path) -> None:
        import os
        import time

        d = _overnight(tmp_path, "wait")
        _db(d, [(5, None, "", "five", 0.6, 1, "2026-01-01T10:00:00", "2026-01-01T10:05:00")])
        (d / "user_answers.json").write_text("{}")
        old = time.time() - 60
        os.utime(d / "user_answers.json", (old, old))
        (d / "mid_loop_questions.json").write_text(
            json.dumps([{"id": "q", "text": "External data allowed?"}])
        )
        (d / "token_usage.jsonl").write_text(
            json.dumps(
                {
                    "purpose": "proposal_round_5",
                    "model": "claude-x",
                    "input_tokens": 100,
                    "output_tokens": 20,
                }
            )
            + "\n"
        )
        p = _project(dashboard_state.build_state(tmp_path), d)
        assert p["state"] == "waiting"
        assert p["overnight"]["waiting_input"]["questions"] == ["External data allowed?"]
        assert p["tokens"]["by_round"] == {"R5": 120} and p["experiments"][0]["tokens"] == 120
        (d / "user_answers.json").write_text('{"x": 1}')  # answered after the questions
        assert _project(dashboard_state.build_state(tmp_path), d)["state"] != "waiting"

    def test_a_reported_cost_beats_list_prices_and_cached_input_is_shown(self) -> None:
        calls = [
            {
                "model": "claude-opus-5-5",
                "input_tokens": 2014,
                "output_tokens": 94,
                "cache_read_tokens": 100,
                "cost_usd": 0.02,
                "purpose": "auto_run",
            },
            {"model": "claude-sonnet-4", "input_tokens": 1_000_000, "output_tokens": 0},
        ]
        t = dashboard_state._tokens(calls, "claude_code")
        assert (t["input_tokens"], t["output_tokens"], t["cache_read_tokens"]) == (
            1_002_014,
            94,
            100,
        )
        assert t["reported_cost_usd"] == 0.02 and t["estimated_cost_usd"] == pytest.approx(3.02)
        assert t["by_model"]["claude-opus-5-5"]["cost_usd"] == 0.02
        assert t["unobservable"] and "own Claude Code chat" in t["note"]

    def test_metered_provider_keeps_the_callers_round_label(self, tmp_path: Path) -> None:
        from researchforge.ai.usage import Usage, purpose, record_call
        from researchforge.kaggle.token_log import metered

        class Fake:
            name = "fake/m"

            def generate(self, system, user, *, max_tokens=8192):
                from researchforge.ai.usage import current_purpose

                record_call(current_purpose(), "fake", "m", Usage(3, 4))
                return "ok"

        prov = metered(Fake(), tmp_path, "setup")
        prov.generate("s", "u")
        with purpose("proposal_round_9"):
            prov.generate("s", "u")
        rows = [json.loads(x) for x in (tmp_path / "token_usage.jsonl").read_text().splitlines()]
        assert [r["purpose"] for r in rows] == ["setup", "proposal_round_9"]

    def test_repo_merge_rule_matches_autorun(self) -> None:
        from researchforge.autorun.merges import find_merge_opportunities

        def exp(i, hyp, files, parents=(), status="promising"):
            return {
                "experiment_id": i,
                "hypothesis_id": hyp,
                "changed_files": list(files),
                "parent_experiment_ids": list(parents),
                "status": status,
            }

        pairs = find_merge_opportunities(
            [
                exp("exp-1", "h1", ["a.py"]),
                exp("exp-2", "h2", ["b.py"]),
                exp("exp-3", "h2", ["c.py"]),  # same hypothesis as exp-2
                exp("exp-4", "h4", ["a.py"]),  # overlaps exp-1's files
                exp("exp-5", "h5", ["d.py"], status="rejected"),  # not an improvement
            ]
        )
        got = {(m["parent_a"], m["parent_b"]) for m in pairs}
        assert ("exp-1", "exp-2") in got and ("exp-2", "exp-3") not in got
        assert ("exp-1", "exp-4") not in got and not any("exp-5" in g for g in got)

    def test_improvement_without_baseline_is_not_claimed(self, tmp_path: Path) -> None:
        d = _overnight(tmp_path, "nobase", baseline=None)
        _db(d, [(1, None, "", "x", 0.8, 1, "2026-01-01T10:00:00", "2026-01-01T10:05:00")])
        e = _project(dashboard_state.build_state(tmp_path), d)["experiments"][0]
        assert e["status"] == "complete" and "did not beat an earlier round" in e["reason"]

    def test_unreadable_files_surface_as_issues(self, tmp_path: Path) -> None:
        d = _overnight(tmp_path, "broken")
        (d / "pending_hypotheses.json").write_text('[{"idea": ')
        p = _project(dashboard_state.build_state(tmp_path), d)
        assert any(
            i["level"] == "error" and "pending_hypotheses" in i["message"] for i in p["issues"]
        )

    def test_watcher_only_copy_points_at_real_project(self, tmp_path: Path) -> None:
        real = _overnight(tmp_path / "kaggle" / "p", "comp")
        stray = _overnight(tmp_path, "comp", setup=False)
        state = dashboard_state.build_state(tmp_path)
        issues = _project(state, stray)["issues"]
        assert any(str(real.resolve()) in i["hint"] for i in issues)

    def test_sensor_state_is_never_faked(self, tmp_path: Path) -> None:
        fresh = _overnight(tmp_path / "a", "fresh")
        legacy = _overnight(tmp_path / "b", "legacy")
        (legacy / "monitor_state.json").write_text(
            json.dumps({"last_arxiv_at": "2026-01-01T00:00:00"})
        )
        state = dashboard_state.build_state(tmp_path)
        f = _project(state, fresh)["overnight"]["sensors"]
        assert not f["has_state"] and {s["status"] for s in f["items"]} == {"never"}
        lg = {s["name"]: s for s in _project(state, legacy)["overnight"]["sensors"]["items"]}
        assert _project(state, legacy)["overnight"]["sensors"]["legacy_state"]
        assert lg["arxiv_scanner_sensor"]["status"] == "unknown"
        assert lg["kernel_watcher_sensor"]["status"] == "never"

    def test_payload_fingerprint_tracks_changes(self, tmp_path: Path) -> None:
        d = _overnight(tmp_path, "fp")
        first = dashboard_state.build_state(tmp_path)["fingerprint"]
        assert dashboard_state.build_state(tmp_path)["fingerprint"] == first
        (d / "pending_hypotheses.json").write_text(json.dumps([{"idea": "new"}]))
        assert dashboard_state.build_state(tmp_path)["fingerprint"] != first


# ---------------------------------------------------------------------------
# Sensors + producer helpers
# ---------------------------------------------------------------------------


class TestSensorsAndProducers:
    def test_toggle_custom_and_legacy_migration(self, tmp_path: Path) -> None:
        d = _overnight(tmp_path, "s")
        (d / "monitor_state.json").write_text(
            json.dumps({"disabled_sensors": ["arxiv_scanner_sensor"]})
        )
        cfg = sensors.load_config(d, "s")
        assert not sensors.is_enabled(cfg, "arxiv_scanner_sensor")
        sensors.set_enabled(d, "s", "arxiv_scanner_sensor", True)
        sensors.set_enabled(d, "s", "morning_briefing_sensor", False)
        sensors.add_custom(d, "s", "disk", "echo ok")
        saved = json.loads((d / "scheduler_config.json").read_text())
        assert saved["disabled_sensors"] == ["morning_briefing_sensor"]
        assert saved["custom_sensors"][0]["name"] == "disk"
        with pytest.raises(ValueError):
            sensors.add_custom(d, "s", "kernel_watcher_sensor", "echo")
        assert sensors.run_custom({"command": "echo healthy", "timeout_seconds": 5}, tmp_path)[
            :2
        ] == ("ok", "healthy")
        assert (
            sensors.run_custom({"command": "exit 3", "timeout_seconds": 5}, tmp_path)[0] == "error"
        )
        assert sensors.remove_custom(d, "s", "disk")

    def test_mark_hypothesis_tested_and_inflight(self, tmp_path: Path) -> None:
        from researchforge.kaggle.cli import _mark_hypothesis_tested, _set_inflight

        d = _overnight(tmp_path, "h")
        (d / "pending_hypotheses.json").write_text(
            json.dumps([{"idea": "Mixup augmentation", "tested": False}])
        )
        assert (
            _mark_hypothesis_tested(d, {"hypothesis": "mixup augmentation"}, 7, "improved")
            == "Mixup augmentation"
        )
        h = json.loads((d / "pending_hypotheses.json").read_text())[0]
        assert h["tested"] and h["tested_in_round"] == 7 and h["tested_outcome"] == "improved"
        assert _mark_hypothesis_tested(d, {"title": "unrelated"}, 8, "failed") is None
        _set_inflight(d, 9, {"round_num": 9, "stage": "screening"})
        _set_inflight(d, 9, {"stage": "full"})
        assert json.loads((d / "inflight_rounds.json").read_text())["9"]["stage"] == "full"
        _set_inflight(d, 9, None)
        assert not (d / "inflight_rounds.json").exists()

    def test_sensors_cli(self, tmp_path: Path, monkeypatch) -> None:
        from typer.testing import CliRunner

        from researchforge.kaggle.cli import kaggle_app

        d = _overnight(tmp_path, "cli-comp")
        monkeypatch.chdir(tmp_path)
        r = CliRunner().invoke(
            kaggle_app,
            [
                "sensors",
                "cli-comp",
                "--disable",
                "arxiv_scanner_sensor",
                "--add",
                "ping",
                "--command",
                "true",
                "--json",
            ],
        )
        assert r.exit_code == 0, r.output
        rows = {x["name"]: x for x in json.loads(r.output)["sensors"]}
        assert rows["arxiv_scanner_sensor"]["enabled"] is False and rows["ping"]["custom"]
        assert json.loads((d / "scheduler_config.json").read_text())["disabled_sensors"] == [
            "arxiv_scanner_sensor"
        ]

    def test_one_shot_arxiv_scan_completes(self, tmp_path: Path, monkeypatch) -> None:
        # Regression: an orphaned merge-detection block after the report raised
        # NameError('rounds') at the end of every one-shot scan.
        from typer.testing import CliRunner

        import researchforge.intelligence.scanner as scanner
        from researchforge.kaggle.cli import kaggle_app

        _overnight(tmp_path, "scan-comp")
        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr("researchforge.ai.providers.get_provider", lambda *a, **k: None)
        monkeypatch.setattr(scanner, "_run_arxiv_cycle", lambda *a, **k: None)
        monkeypatch.setattr(scanner, "inject_arxiv_ideas_into_hypothesis_pool", lambda *a, **k: 0)
        r = CliRunner().invoke(
            kaggle_app, ["arxiv-scan", "scan-comp", "--query", "tabular", "--once"]
        )
        assert r.exit_code == 0, r.output
        assert "No ideas found" in r.output

    # ---------------------------------------------------------------------------
    # HTTP service
    # ---------------------------------------------------------------------------

    def test_watch_interval_starts_the_sensor_monitor(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, _isolated_home: Path
    ) -> None:
        import subprocess

        from researchforge.kaggle import cli as kcli

        d = _overnight(tmp_path / "proj", "demo-comp")
        calls: list[dict] = []
        monkeypatch.setattr(
            subprocess, "Popen", lambda args, **kw: calls.append({"args": args, **kw})
        )
        monkeypatch.setattr("researchforge.kaggle.api.list_competition_kernels", lambda *a, **k: [])

        kcli._start_watcher_background("demo-comp", d, 4.0)

        assert len(calls) == 1
        call = calls[0]
        assert call["args"][1:] == [
            "-m",
            "researchforge",
            "kaggle",
            "monitor",
            "demo-comp",
            "--poll-hours",
            "4.0",
        ]
        assert call["env"]["RESEARCHFORGE_OVERNIGHT_DIR"] == str(d)
        assert call["cwd"] == str(d.parent)
        assert (_isolated_home / ".local/share/researchforge/logs/monitor_demo-comp.log").exists()
        assert (
            json.loads((d / "scheduler_config.json").read_text())["watcher_interval_hours"] == 4.0
        )


class TestService:
    @pytest.fixture
    def server(self, tmp_path: Path):
        import http.server

        from researchforge.kaggle.dashboard import DashboardService, make_handler

        _overnight(tmp_path / "ws", "svc")
        service = DashboardService(tmp_path / "ws")
        httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), make_handler(service, 0))
        port = httpd.server_address[1]
        httpd.RequestHandlerClass = make_handler(service, port)
        threading.Thread(target=httpd.serve_forever, daemon=True).start()
        yield service, port
        httpd.shutdown()

    def _req(self, port, path, *, method="GET", body=None, headers=None):
        req = urllib.request.Request(
            f"http://127.0.0.1:{port}{path}",
            method=method,
            data=json.dumps(body).encode() if body is not None else None,
            headers={"Content-Type": "application/json", **(headers or {})},
        )
        try:
            with urllib.request.urlopen(req, timeout=10) as r:
                return r.status, dict(r.headers), r.read()
        except urllib.error.HTTPError as e:
            return e.code, dict(e.headers), e.read()

    def test_shell_state_etag_and_actions(self, server) -> None:
        service, port = server
        status, _, page = self._req(port, "/")
        assert (
            status == 200 and service.token.encode() in page and b'http-equiv="refresh"' not in page
        )
        status, headers, body = self._req(port, "/api/state")
        payload = json.loads(body)
        assert status == 200 and payload["schema_version"] == dashboard_state.SCHEMA_VERSION
        assert self._req(port, "/api/state", headers={"If-None-Match": headers["ETag"]})[0] == 304

        pid = payload["projects"][0]["id"]
        body = {"project": pid, "name": "arxiv_scanner_sensor", "enabled": False}
        assert self._req(port, "/api/sensor", method="POST", body=body)[0] == 403  # no token
        assert (
            self._req(
                port, "/api/sensor", method="POST", body=body, headers={"X-RF-Token": "wrong"}
            )[0]
            == 403
        )
        status, _, out = self._req(
            port, "/api/sensor", method="POST", body=body, headers={"X-RF-Token": service.token}
        )
        assert status == 200 and json.loads(out)["ok"]
        art = Path(payload["projects"][0]["overnight"]["artifacts_dir"])
        assert json.loads((art / "scheduler_config.json").read_text())["disabled_sensors"] == [
            "arxiv_scanner_sensor"
        ]
        status, _, out = self._req(
            port,
            "/api/sensor",
            method="POST",
            headers={"X-RF-Token": service.token},
            body={"project": pid, "name": "nope", "enabled": True},
        )
        assert status == 400

    def test_rejects_foreign_host(self, server) -> None:
        _, port = server
        assert self._req(port, "/api/state", headers={"Host": "evil.example"})[0] == 403


# ---------------------------------------------------------------------------
# Public-notebook LB claims
# ---------------------------------------------------------------------------


class TestKernelClaims:
    def test_kernel_listing_keeps_votes_and_quoted_commas(self) -> None:
        from researchforge.kaggle.api import _parse_kernel_csv

        out = (
            "Warning: Looks like you're using an outdated `kaggle` version\n"
            "ref,title,author,lastRunTime,totalVotes\n"
            'a/one,"Spectra, Structures [0.468]",Ann,2026-10-21 14:37:15,173\n'
            "b/two,Plain title,Bob,2026-10-22 09:00:00,0\n"
        )
        kernels = _parse_kernel_csv(out)
        assert [(k.ref, k.title, k.author, k.total_votes) for k in kernels] == [
            ("a/one", "Spectra, Structures [0.468]", "Ann", 173),
            ("b/two", "Plain title", "Bob", 0),
        ]

    def test_title_claims_are_read_as_written(self) -> None:
        from researchforge.kaggle.notebook_baseline import claimed_lb_score as claim

        assert claim("casmi26-0.482") == 0.482
        assert claim("CASMI26 LB 0.480 | Neural Ensemble") == 0.48
        assert claim("[0.468] CASMI: From Peaks to Structures") == 0.468
        assert claim("ARC agent score 32.51") == 32.51  # never rescaled
        for none in ("casmi26-peaks-to-structures-v2", "v0.22 release", "casmi26-lb472-r3"):
            assert claim(none) is None, none

    def test_notebook_claim_prefers_headline_markdown(self) -> None:
        from researchforge.kaggle.kernel_claims import extract_claim

        value, snippet = extract_claim("# LB 0.477 · Two 0.472 fixes", "print('LB 0.460')")
        assert value == 0.477 and "LB 0.477" in snippet
        assert extract_claim("", "# public score: 0.455\n") == (0.455, "# public score: 0.455")
        assert extract_claim("EDA only", "x = 0.5") is None

    def test_pool_is_linked_and_compared_with_our_best_lb(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import researchforge.kaggle.kernel_claims as kc
        from researchforge.kaggle.api import PublicKernel
        from researchforge.kaggle.kernel_watcher import annotate_kernel_ideas

        d = _overnight(tmp_path, "claims")
        (d / "competition_metadata.json").write_text(json.dumps({"metric_direction": "maximize"}))
        (d / "pending_hypotheses.json").write_text(
            json.dumps(
                [
                    {
                        "idea": "fork m477",
                        "source": "kernel_watcher",
                        "source_refs": ["a/x", "b/y"],
                    },
                    {"idea": "LB 0.40 baseline", "source": "kernel_watcher"},
                    {"idea": "my own idea", "source": "claude_observation"},
                ]
            )
        )
        (d / "submissions.json").write_text(
            json.dumps(
                {
                    "submissions": [
                        {"id": "1", "status": "COMPLETE", "publicscore": "0.166"},
                        {"id": "2", "status": "COMPLETE", "publicscore": "0.136"},
                    ]
                }
            )
        )
        fetched: list[str] = []

        def fake_claim(ref: str, timeout: float = 60.0) -> dict[str, object]:
            fetched.append(ref)
            return {
                "claimed_score": 0.477,
                "claim_source": "notebook",
                "claim_snippet": "# LB 0.477",
            }

        monkeypatch.setattr(kc, "notebook_claim", fake_claim)
        kernels = [
            PublicKernel("u/fork-m477", "fork m477", "u", "2026-11-01", 12),
            PublicKernel("v/base", "LB 0.40 baseline", "v", "2026-11-02", 3),
        ]
        assert annotate_kernel_ideas(d, kernels) == 2
        assert fetched == ["u/fork-m477"]  # the titled claim needed no download
        assert annotate_kernel_ideas(d, kernels) == 0 and fetched == ["u/fork-m477"]  # cached

        p = _project(dashboard_state.build_state(tmp_path), d)
        assert p["our_best_lb"]["value"] == 0.166
        by = {h["title"]: h for h in p["hypotheses"]}
        k = by["fork m477"]["kernel"]
        assert k["url"] == "https://www.kaggle.com/code/u/fork-m477" and k["votes"] == 12
        assert k["claimed_score"] == 0.477 and k["claim_source"] == "notebook"
        assert k["vs_our_best"] == pytest.approx(0.311)
        assert by["LB 0.40 baseline"]["kernel"]["claim_source"] == "title"
        assert "kernel" not in by["my own idea"]
