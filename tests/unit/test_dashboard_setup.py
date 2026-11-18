"""Setting up, signing in and steering from the dashboard (and the CLI behind it)."""

from __future__ import annotations

import http.server
import json
import stat
import threading
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from researchforge.cli import app
from researchforge.kaggle import keys, new_project, steering

SLUG = "demo-comp"
SECRET = "sk-ant-api03-THIS-IS-A-TEST-VALUE"


@pytest.fixture(autouse=True)
def home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    h = tmp_path / "home"
    h.mkdir()
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: h))
    for name in (*keys.ENV_SLOT_KEYS, "KAGGLE_API_TOKEN", "NTFY_TOPIC", "AWS_ACCESS_KEY_ID"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.delenv("RESEARCHFORGE_OVERNIGHT_DIR", raising=False)
    # Colab and AWS status would ask the real CLIs; these tests are about the other keys.
    monkeypatch.setattr("researchforge.kaggle.colab.cached", lambda *a, **k: {"signed_in": False})
    monkeypatch.setattr("researchforge.kaggle.keys.shutil.which", lambda _n: None)
    return h


def _overnight(root: Path) -> Path:
    d = root / f".overnight_{SLUG}"
    d.mkdir(parents=True)
    (d / "overnight_setup.json").write_text(
        json.dumps(
            {
                "competition_slug": SLUG,
                "data_dir": str(root / "data"),
                "artifacts_dir": str(d),
                "selected_gpu": "kaggle_gpu",
                "baseline_result": None,
            }
        )
    )
    return d


def _kaggle(*args: str) -> tuple[int, str]:
    result = CliRunner().invoke(app, ["kaggle", *args])
    return result.exit_code, result.output


# ── keys ────────────────────────────────────────────────────────────────────


class TestKeys:
    def test_set_saves_privately_and_status_never_carries_the_value(self, home: Path) -> None:
        where = keys.set_key("anthropic", f"  {SECRET}  ")
        env = home / ".config/researchforge/auto_run.env"
        assert where == str(env) and f"ANTHROPIC_API_KEY={SECRET}" in env.read_text()
        assert stat.S_IMODE(env.stat().st_mode) == 0o600
        row = keys.status("anthropic")[0]
        assert row["set"] and row["where"] == str(env) and row["test"] is None
        assert SECRET not in json.dumps(keys.status())

    def test_a_new_value_forgets_the_old_test_and_remove_clears_it(
        self, home: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        keys.set_key("openai", "sk-old")
        monkeypatch.setattr(keys, "_http_ok", lambda url, headers: (False, "refused (HTTP 401)"))
        assert keys.test_key("openai") == {
            "ok": False,
            "detail": "refused (HTTP 401)",
            "at": keys.status("openai")[0]["test"]["at"],
        }  # noqa: E501
        assert not keys.ready("openai")
        keys.set_key("openai", "sk-new")
        assert keys.status("openai")[0]["test"] is None and keys.ready("openai")
        assert keys.remove_key("openai").startswith("removed from")
        assert not keys.status("openai")[0]["set"] and keys.remove_key("openai") == "not set"

    def test_test_sends_the_stored_key_to_the_provider(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        keys.set_key("anthropic", SECRET)
        seen: dict[str, Any] = {}
        monkeypatch.setattr(
            keys, "_http_ok", lambda url, headers: seen.update(url=url, h=headers) or (True, "ok")
        )
        assert keys.test_key("anthropic")["ok"]
        assert (
            seen["url"].startswith("https://api.anthropic.com/")
            and seen["h"]["x-api-key"] == SECRET
        )

    def test_kaggle_token_or_legacy_key(self, home: Path) -> None:
        assert keys.set_key("kaggle", "KGAT_abc123") == str(home / ".kaggle/access_token")
        assert (home / ".kaggle/access_token").read_text() == "KGAT_abc123"
        with pytest.raises(ValueError, match="username"):
            keys.set_key("kaggle", "0123456789abcdef")
        keys.set_key("kaggle", "0123456789abcdef", username="me")
        assert json.loads((home / ".kaggle/kaggle.json").read_text()) == {
            "username": "me",
            "key": "0123456789abcdef",
        }
        assert stat.S_IMODE((home / ".kaggle/kaggle.json").stat().st_mode) == 0o600
        keys.set_key("kaggle", '{"username": "u2", "key": "k2"}')
        assert json.loads((home / ".kaggle/kaggle.json").read_text())["username"] == "u2"

    def test_sign_ins_and_bad_input_are_refused(self) -> None:
        with pytest.raises(ValueError, match="signed in"):
            keys.set_key("colab", "x")
        with pytest.raises(ValueError, match="one line"):
            keys.set_key("runpod", "a\nb")
        with pytest.raises(KeyError):
            keys.set_key("nope", "x")
        with pytest.raises(ValueError):
            keys.remove_key("kaggle")  # Kaggle's own files are left alone

    def test_cloud_rounds_find_a_key_saved_on_the_page(self) -> None:
        from researchforge.kaggle.cloud_gpu_config import CloudGPUConfig

        assert CloudGPUConfig(provider="runpod").api_key() == ""
        keys.set_key("runpod", "rp-123")
        assert CloudGPUConfig(provider="runpod").api_key() == "rp-123"

    def test_cli_status_and_set_refuses_without_a_terminal(self) -> None:
        keys.set_key("gemini", SECRET)
        code, out = _kaggle("keys", "status", "--json")
        rows = {r["name"]: r for r in json.loads(out)}
        assert code == 0 and rows["gemini"]["set"] and SECRET not in out
        code, out = _kaggle("keys", "set", "openai")
        assert code == 1 and "your own terminal" in out
        assert _kaggle("keys", "wait", "openai", "--timeout", "1")[0] == 1


# ── setup wizard ────────────────────────────────────────────────────────────

FACTS = {
    "slug": SLUG,
    "metric_direction": "maximize",
    "recommended_compute": "kaggle_gpu",
    "is_code_competition": False,
    "auto_run_driver": "cursor",
}


class TestNewProject:
    def test_default_answers_follow_the_facts(self) -> None:
        form = new_project.default_form(FACTS)
        assert form["compute"] == "kaggle" and form["auto_run"]["driver"] == "cursor"
        assert (
            form["sensors"]["auto_run_sensor"] is False and form["sensors"]["kernel_watcher_sensor"]
        )
        code = new_project.default_form(
            {**FACTS, "recommended_compute": "local", "is_code_competition": True}
        )
        assert code["compute"] == "kaggle"

    def test_commands_match_what_overnight_runs(self) -> None:
        form = new_project.default_form(FACTS)
        form |= {"compute": "colab", "time_hours": "4", "metric_direction": "minimize"}
        form["colab"] = {"plan": "pro", "gpu": "L4", "high_mem": True}
        form["sensors"] |= {"arxiv_scanner_sensor": False, "auto_run_sensor": True}
        form["answers"]["known_approaches"] = "CatBoost"
        cmds = new_project.setup_commands(SLUG, form, FACTS)
        assert cmds[0] == ["colab-setup", SLUG, "--plan", "pro", "--gpu", "L4", "--high-mem"]
        overnight = cmds[1]
        assert overnight[:4] == ["overnight", SLUG, "--gpu", "colab"] and "4" in overnight
        assert json.loads(overnight[overnight.index("--answers-json") + 1]) == {
            "known_approaches": "CatBoost",
            "external_data": "unknown",
        }
        assert ["settings", SLUG, "--metric-direction", "minimize"] in cmds
        sensors = next(c for c in cmds if c[0] == "sensors")
        assert "--disable" in sensors and "arxiv_scanner_sensor" in sensors
        assert ["driver", SLUG, "cursor"] in cmds
        auto = next(c for c in cmds if c[0] == "auto-run")
        assert auto[-1] == "--on" and auto[auto.index("--driver") + 1] == "cursor"
        assert cmds[-1] == ["scheduler-install", SLUG, "--interval-hours", "6.0"]

    @pytest.mark.parametrize(
        ("patch", "error"),
        [
            ({"compute": "mainframe"}, "compute must be"),
            ({"compute": "ssh", "ssh": {"host": "server"}}, "user@server"),
            ({"compute": "cloud", "cloud": {"provider": "aws", "gpu_type": "x"}}, "AWS"),
            ({"compute": "cloud", "cloud": {"provider": "runpod", "gpu_type": ""}}, "GPU type"),
            ({"time_hours": 500}, "time budget"),
            ({"auto_run": {"driver": "gpt"}}, "driver must be"),
        ],
    )
    def test_invalid_answers_are_refused(self, patch: dict[str, Any], error: str) -> None:
        with pytest.raises(ValueError, match=error):
            new_project.setup_commands(SLUG, {**new_project.default_form(FACTS), **patch}, FACTS)

    def test_a_code_competition_only_runs_on_kaggle(self) -> None:
        facts = {**FACTS, "is_code_competition": True}
        form = {**new_project.default_form(facts), "compute": "local"}
        with pytest.raises(ValueError, match="Kaggle GPU"):
            new_project.setup_commands(SLUG, form, facts)

    def test_missing_keys_names_what_the_choices_need(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        form = {
            **new_project.default_form(FACTS),
            "compute": "cloud",
            "cloud": {"provider": "vast"},
        }
        assert new_project.missing_keys(form) == ["kaggle", "vast"]
        keys.set_key("kaggle", "KGAT_x")
        keys.set_key("vast", "v")
        assert new_project.missing_keys(form) == []

    def test_draft_cli_is_shared_with_the_page(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        root = tmp_path / "comp"
        art = root / f".overnight_{SLUG}"
        art.mkdir(parents=True)
        (art / "facts.json").write_text(json.dumps(FACTS))
        monkeypatch.chdir(root)
        assert _kaggle("draft", SLUG)[0] == 1
        code, out = _kaggle("draft", SLUG, "--start")
        assert code == 0 and "researchforge kaggle overnight" in out
        _kaggle("draft", SLUG, "--form-json", '{"compute": "colab", "colab": {"gpu": "L4"}}')
        draft = json.loads(_kaggle("draft", SLUG, "--json")[1])
        assert draft["via"] == "chat" and draft["form"]["colab"] == {
            "plan": "free",
            "gpu": "L4",
            "high_mem": False,
        }
        assert (
            draft["commands"][0][:2] == ["colab-setup", SLUG] and "kaggle" in draft["missing_keys"]
        )
        code, out = _kaggle("draft", SLUG, "--run")
        assert code == 1 and "set these keys first" in out


# ── steering ────────────────────────────────────────────────────────────────


class TestSteering:
    def test_next_round_directions_stay_until_a_round_follows_them(self, tmp_path: Path) -> None:
        d = _overnight(tmp_path)
        nxt = steering.add(d, "try a ViT next")
        keep = steering.add(d, "never use external data", "standing")
        active = steering.mark_round(d, 4)
        assert [x.id for x in active] == [keep.id, nxt.id]  # the newest first
        read = steering.load(d)[1]
        assert read.status == "read" and read.read_by_round == 4
        assert [x.id for x in steering.mark_round(d, 5)] == [keep.id, nxt.id]  # read, not followed
        steering.record_round(d, 5, {"follows": [nxt.id]})
        assert [x.id for x in steering.mark_round(d, 6)] == [keep.id]
        assert steering.load(d)[1].status == "applied" and steering.load(d)[1].followed_by == [5]
        steering.remove(d, keep.id)
        assert steering.active(d) == []

    def test_directions_lead_person_notes_and_want_one_reply(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from researchforge.kaggle import control

        d = _overnight(tmp_path)
        sent: list[str] = []
        monkeypatch.setattr(control, "notify", lambda _d, _c, msg, **_k: sent.append(msg) or True)
        x = steering.add(d, "stop tuning LightGBM")
        assert control.person_notes(d)[0] == (
            f"Up next #1 [{x.id}] Direction (head of the queue: follow it or decline it with a "
            "reason): stop tuning LightGBM"
        )
        assert steering.for_round(steering.active(d))[0]["reply_needed"]
        steering.reply(d, x.id, "a note from me", by="person")
        assert steering.for_round(steering.active(d))[0]["reply_needed"] and not sent
        steering.reply(d, x.id, "R5 switches to a transformer")
        assert not steering.for_round(steering.active(d))[0]["reply_needed"]
        assert sent and "R5 switches" in sent[0]

    def test_bad_directions_are_refused(self, tmp_path: Path) -> None:
        d = _overnight(tmp_path)
        with pytest.raises(ValueError):
            steering.add(d, "   ")
        with pytest.raises(ValueError):
            steering.add(d, "x", "forever")
        with pytest.raises(KeyError):
            steering.reply(d, "d-nope", "hi")

    def test_steer_cli_and_prepare_round(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        d = _overnight(tmp_path)
        monkeypatch.setenv("RESEARCHFORGE_OVERNIGHT_DIR", str(d))
        # The proposal context needs a full experiment contract; directions do not.
        monkeypatch.setattr(
            "researchforge.kaggle.patch_proposer.PatchProposer._build_context",
            lambda self, round_num: "context",
        )
        assert _kaggle("steer", SLUG, "use group features", "--standing")[0] == 0
        code, out = _kaggle("prepare-round", SLUG)
        ctx = json.loads(out[out.index("{") :])
        (direction,) = ctx["directions"]
        assert code == 0 and direction["kind"] == "standing" and direction["reply_needed"]
        assert f"steer {SLUG} --reply" in ctx["directions_note"]
        assert _kaggle("steer", SLUG, "on it in R1", "--reply", direction["id"])[0] == 0
        assert "↳ agent: on it in R1" in _kaggle("steer", SLUG)[1]
        assert _kaggle("steer", SLUG, "--remove", "d-nope")[0] == 1


# ── Up next: the person's queue ─────────────────────────────────────────────


def _pool(d: Path, *ideas: str) -> None:
    (d / "pending_hypotheses.json").write_text(
        json.dumps([{"idea": i, "source": "kernel_watcher", "tested": False} for i in ideas])
    )


class TestUpNext:
    def test_newest_first_and_reorderable(self, tmp_path: Path) -> None:
        d = _overnight(tmp_path)
        a, b, c = (steering.add(d, t) for t in ("try A", "try B", "try C"))
        assert [x["id"] for x in steering.up_next(d)] == [c.id, b.id, a.id]
        steering.reorder(d, [a.id])
        assert [x["id"] for x in steering.up_next(d)] == [a.id, c.id, b.id]
        with pytest.raises(KeyError):
            steering.reorder(d, ["d-nope"])

    def test_a_direction_for_n_rounds_stays_until_n_rounds_follow_it(self, tmp_path: Path) -> None:
        d = _overnight(tmp_path)
        x = steering.add(d, "try RL", rounds=2)
        steering.record_round(d, 7, {"follows": [x.id]})
        assert steering.up_next(d)[0]["rounds_left"] == 1
        steering.record_round(d, 8, {"follows": x.id})
        assert steering.up_next(d) == [] and steering.load(d)[0].followed_by == [7, 8]
        with pytest.raises(ValueError):
            steering.add(d, "x", rounds=0)

    def test_the_head_is_followed_or_declined_with_a_reason(self, tmp_path: Path) -> None:
        d = _overnight(tmp_path)
        assert steering.check_proposal(d, {}) == ""  # nothing queued
        old = steering.add(d, "try A")
        new = steering.add(d, "try RL")
        refusal = steering.check_proposal(d, {"follows": [old.id]})
        assert new.id in refusal and "try RL" in refusal  # the newest is the head
        assert steering.check_proposal(d, {"declines": {new.id: ""}}) == refusal  # needs a reason
        why = {new.id: "needs 4 rollouts per task; over the 2 h limit"}
        assert steering.check_proposal(d, {"follows": [old.id], "declines": why}) == ""
        steering.record_round(d, 9, {"follows": [old.id], "declines": why})
        (head,) = steering.up_next(d)  # A is done; RL still waits, with the reason
        assert head["id"] == new.id
        assert steering.load(d)[0].declined[0]["round"] == 9
        assert steering.check_proposal(d, {"follows": [new.id]}) == ""

    def test_pinned_ideas_come_after_directions_in_your_order(self, tmp_path: Path) -> None:
        from researchforge.kaggle import hypothesis_pool as pool

        d = _overnight(tmp_path)
        _pool(d, "idea one", "idea two", "idea three")
        assert [h["idea"] for h in pool.rank_untested(d, "maximize")][0] == "idea one"
        pool.pin(d, "idea three")
        two = pool.pin(d, "idea two", place=1)[0]
        assert [h["idea"] for h in pool.pinned(d)] == ["idea two", "idea three"]
        ranked = [h["idea"] for h in pool.rank_untested(d, "maximize")]
        assert ranked == ["idea two", "idea three", "idea one"]
        pool.set_pins(d, ["idea three", two["pin_id"]])  # by text or by id
        assert [h["pin"] for h in pool.pinned(d)] == [1, 2]
        with pytest.raises(ValueError, match="no untested idea"):
            pool.pin(d, "no such idea")
        x = steering.add(d, "try RL")
        assert [i["kind"] for i in steering.up_next(d)] == [
            "direction",
            "pinned_idea",
            "pinned_idea",
        ]
        steering.record_round(d, 3, {"follows": [x.id]})
        head = steering.up_next(d)[0]
        assert head["text"] == "idea three" and steering.check_proposal(d, {})
        assert steering.check_proposal(d, {"hypothesis": "Idea three"}) == ""  # testing it follows
        pool.mark_tested(d, {"hypothesis": "idea three"}, 4, "failed")
        assert [h["idea"] for h in pool.pinned(d)] == ["idea two"]
        pool.unpin(d, "idea two")
        assert pool.pinned(d) == [] and steering.up_next(d) == []

    def test_steered_rounds_remember_what_each_round_did(self, tmp_path: Path) -> None:
        from researchforge.kaggle import hypothesis_pool as pool

        d = _overnight(tmp_path)
        _pool(d, "idea one")
        pool.pin(d, "idea one")
        rl = steering.add(d, "try RL")
        why = {rl.id: "no GPU today"}
        steering.record_round(d, 4, {"hypothesis": "idea one", "declines": why})
        steering.record_round(d, 5, {"follows": [rl.id]})
        steering.record_round(d, 6, {"title": "nothing queued any more"})
        got = steering.steered_rounds(d)
        assert [(g["kind"], g["how"]) for g in got[4]] == [
            ("direction", "declined"),
            ("pinned_idea", "followed"),
        ]
        assert got[4][0]["reason"] == "no GPU today"
        assert [g["id"] for g in got[5]] == [rl.id] and 6 not in got
        # Older directions knew only the round that read them.
        old = steering.Direction("d-old", "try X", "next", "", status="applied", read_by_round=2)
        steering._save(d, [*steering.load(d), old])
        assert steering.steered_rounds(d)[2][0]["id"] == "d-old"

    def test_execute_round_refuses_a_proposal_that_ignores_the_head(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        d = _overnight(tmp_path)
        monkeypatch.setenv("RESEARCHFORGE_OVERNIGHT_DIR", str(d))
        x = steering.add(d, "try RL")
        (d / "proposal.json").write_text(json.dumps({"title": "t", "code_changes": "print(1)"}))
        code, out = _kaggle("execute-round", SLUG)
        assert code == 1 and f'"follows": ["{x.id}"]' in out
        assert not (d / "execute_round.lock").exists()

    def test_cli_rounds_order_and_pin(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        d = _overnight(tmp_path)
        monkeypatch.setenv("RESEARCHFORGE_OVERNIGHT_DIR", str(d))
        _pool(d, "idea one", "idea two")
        code, out = _kaggle("steer", SLUG, "try RL", "--rounds", "3")
        assert code == 0 and "the next 3 rounds" in out
        first = steering.add(d, "try A")
        rl = steering.load(d)[1]
        assert _kaggle("steer", SLUG, "--order", rl.id)[0] == 0
        assert steering.up_next(d)[0]["id"] == rl.id and steering.up_next(d)[1]["id"] == first.id
        assert _kaggle("pin", SLUG, "idea two")[0] == 0
        code, out = _kaggle("pin", SLUG)
        assert code == 0 and "1. [p-" in out and "idea two" in out
        assert _kaggle("pin", SLUG, "nope")[0] == 1
        assert "Up next:" in _kaggle("steer", SLUG)[1]

    def test_the_api_key_loop_is_told_once_then_refused(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from types import SimpleNamespace

        from researchforge.kaggle.patch_proposer import PatchProposal, PatchProposer

        d = _overnight(tmp_path)
        x = steering.add(d, "try RL")
        proposer = PatchProposer.__new__(PatchProposer)
        proposer.setup = SimpleNamespace(artifacts_dir=d)  # type: ignore[assignment]
        proposer._provider = object()
        seen: list[str] = []
        answers = [
            PatchProposal("a", "", "", "", ""),
            PatchProposal("b", "", "", "", "", follows=[x.id]),
        ]
        monkeypatch.setattr(proposer, "_build_context", lambda n: "ctx")
        monkeypatch.setattr(proposer, "_call_ai", lambda ctx, n: seen.append(ctx) or answers.pop(0))
        assert proposer.propose_patch(5).title == "b"
        assert "REFUSED" in seen[1] and x.id in seen[1]
        answers[:] = [PatchProposal("c", "", "", "", "")] * 2
        with pytest.raises(RuntimeError, match="Proposal refused"):
            proposer.propose_patch(6)

    def test_prepare_round_hands_over_the_queue(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from researchforge.kaggle import hypothesis_pool as pool

        d = _overnight(tmp_path)
        monkeypatch.setenv("RESEARCHFORGE_OVERNIGHT_DIR", str(d))
        monkeypatch.setattr(
            "researchforge.kaggle.patch_proposer.PatchProposer._build_context",
            lambda self, round_num: "context",
        )
        _pool(d, "idea one", "idea two")
        pool.pin(d, "idea two")
        code, out = _kaggle("prepare-round", SLUG)
        ctx = json.loads(out[out.index("{") :])
        assert code == 0 and ctx["recommended_next"] == "idea two"
        assert ctx["up_next"][0]["kind"] == "pinned_idea" and "follows" in ctx["up_next_note"]
        assert "follows" in ctx["proposal_schema"] and "declines" in ctx["proposal_schema"]


# ── the dashboard service ───────────────────────────────────────────────────


class TestDashboardEndpoints:
    @pytest.fixture
    def server(self, tmp_path: Path):
        from researchforge.kaggle.dashboard import DashboardService, make_handler

        ws = tmp_path / "ws"
        ws.mkdir()
        service = DashboardService(ws)
        httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), make_handler(service, 0))
        port = httpd.server_address[1]
        httpd.RequestHandlerClass = make_handler(service, port)
        threading.Thread(target=httpd.serve_forever, daemon=True).start()
        yield service, port
        httpd.shutdown()

    @staticmethod
    def _post(port: int, path: str, body: dict[str, Any], token: str) -> tuple[int, dict[str, Any]]:
        req = urllib.request.Request(
            f"http://127.0.0.1:{port}{path}",
            method="POST",
            data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json", "X-RF-Token": token},
        )
        try:
            with urllib.request.urlopen(req, timeout=10) as r:
                return r.status, json.loads(r.read())
        except urllib.error.HTTPError as e:
            return e.code, json.loads(e.read())

    def test_keys_round_trip_never_returns_the_value(
        self, server: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        service, port = server
        monkeypatch.setattr(keys, "_http_ok", lambda url, headers: (True, "accepted (HTTP 200)"))
        assert (
            self._post(port, "/api/keys/set", {"name": "anthropic", "value": SECRET}, "bad")[0]
            == 403
        )
        status, out = self._post(
            port, "/api/keys/set", {"name": "anthropic", "value": SECRET}, service.token
        )
        assert status == 200 and out["test"]["ok"] and SECRET not in json.dumps(out)
        state = json.dumps(service.state(force=True))
        assert SECRET not in state and '"name": "anthropic", "label": "Anthropic API key"' in state
        status, out = self._post(
            port, "/api/keys/set", {"name": "colab", "value": "x"}, service.token
        )
        assert status == 400 and "signed in" in out["error"]

    def test_new_competition_refuses_a_set_up_slug_and_unknown_drafts(
        self, server: Any, tmp_path: Path
    ) -> None:
        from researchforge.kaggle.workspace import register_overnight_dir

        service, port = server
        d = _overnight(tmp_path / "old")
        register_overnight_dir(SLUG, d)
        status, out = self._post(port, "/api/new/facts", {"competition": SLUG}, service.token)
        assert status == 400 and "already set up" in out["error"]
        status, out = self._post(
            port, "/api/new/setup", {"artifacts_dir": str(d), "form": {}}, service.token
        )
        assert status == 400 and "no setup draft" in out["error"]
        status, out = self._post(
            port, "/api/new/facts", {"competition": "other", "root": "/etc"}, service.token
        )
        assert status == 400 and "inside" in out["error"]

    def test_setup_job_runs_the_draft_and_removes_it(
        self, server: Any, home: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        service, port = server
        art = home / "comp" / f".overnight_{SLUG}"
        new_project.save_draft(
            art, {"slug": SLUG, "facts": FACTS, "form": new_project.default_form(FACTS)}
        )
        keys.set_key("kaggle", "KGAT_x")
        ran: list[list[str]] = []

        def fake_stream(
            argv: list[str], cwd: Path, env: dict[str, str], rec: dict, changed: Any
        ) -> int:
            ran.append(argv)
            assert env["RESEARCHFORGE_OVERNIGHT_DIR"] == str(art) and "CLAUDECODE" not in env
            rec["output"] = "ok\n"
            return 0

        monkeypatch.setenv("CLAUDECODE", "1")
        monkeypatch.setattr(type(service), "_stream", staticmethod(fake_stream))
        status, out = self._post(port, "/api/new/setup", {"artifacts_dir": str(art)}, service.token)
        assert status == 200, out
        job = service._jobs[out["job"]]
        for _ in range(200):
            if job["status"] != "running":
                break
            threading.Event().wait(0.02)
        assert (
            job["status"] == "done"
            and ran[0][0] == "overnight"
            and ["driver", SLUG, "cursor"] in ran
        )
        assert not (art / new_project.DRAFT_FILE).exists()  # not written back after removal

    def test_steer_and_loop_endpoints(self, server: Any, tmp_path: Path) -> None:
        service, port = server
        d = _overnight(service.cwd)
        pid = next(p["id"] for p in service.state(force=True)["projects"] if p["slug"] == SLUG)
        status, out = self._post(
            port,
            "/api/steer",
            {"project": pid, "op": "add", "text": "try X", "kind": "standing"},
            service.token,
        )
        assert status == 200 and out["direction"]["kind"] == "standing"
        project = next(p for p in service.state(force=True)["projects"] if p["id"] == pid)
        assert project["overnight"]["steering"][0]["text"] == "try X"
        status, out = self._post(
            port, "/api/loop", {"project": pid, "op": "explode"}, service.token
        )
        assert status == 400
        assert steering.load(d)[0].text == "try X"

    def test_up_next_endpoints(self, server: Any, tmp_path: Path) -> None:
        from researchforge.kaggle import hypothesis_pool as pool

        service, port = server
        d = _overnight(service.cwd)
        _pool(d, "idea one", "idea two")
        pid = next(p["id"] for p in service.state(force=True)["projects"] if p["slug"] == SLUG)
        post = lambda body: self._post(port, "/api/steer", {"project": pid, **body}, service.token)  # noqa: E731
        assert post({"op": "add", "text": "try A", "kind": "next", "rounds": 2})[0] == 200
        assert post({"op": "add", "text": "try B"})[0] == 200
        a, b = steering.load(d)[1], steering.load(d)[0]
        status, out = post({"op": "order", "ids": [a.id]})
        assert status == 200 and [x["id"] for x in out["up_next"]] == [a.id, b.id]
        two = pool.pin_id("idea two")
        status, out = post({"op": "pin", "id": two})
        assert status == 200 and out["up_next"][-1]["id"] == two
        project = next(p for p in service.state(force=True)["projects"] if p["id"] == pid)
        rows = {h["title"]: h for h in project["hypotheses"]}
        assert rows["idea two"]["pin"] == 1 and rows["idea one"]["pin"] is None
        assert [x["text"] for x in project["overnight"]["engine_next"]] == ["idea one"]
        assert all(e["steered"] == [] for e in project["experiments"])
        assert rows["idea one"]["steered"] == []
        assert post({"op": "unpin", "id": two})[0] == 200 and pool.pinned(d) == []
        assert post({"op": "pin", "id": "p-nope"})[0] == 400
