"""Colab through Google's Colab CLI."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from researchforge.kaggle import auto_run as ar
from researchforge.kaggle import colab, run_config
from researchforge.kaggle.experiment_loop import OvernightExperimentLoop

SLUG = "comp"

# A stand-in for Google's `colab`: signed out until a code is pasted, then signed in.
# The "VM" is a local folder standing in for /content: `upload` puts the round there (the
# data download skipped), the `exec` that starts the round runs it to the end, the other
# `exec`s run ResearchForge's snippets against that folder, and `download` copies back.
FAKE_COLAB = """
import os, re, shutil, subprocess, sys, pathlib
state = pathlib.Path(os.environ["FAKE_COLAB_STATE"])
vm = pathlib.Path(str(state) + ".vm")
def here(path):
    return pathlib.Path(path.replace("/content", str(vm), 1))
def log(line):
    with open(str(state) + ".argv", "a") as fh:
        fh.write(line + "\\n")
cmd, args = sys.argv[1], sys.argv[2:]
log(" ".join([cmd, *args]))
if cmd == "whoami":
    if state.exists():
        print("you@example.com"); sys.exit(0)
    print("To authorize colab-cli, visit this URL in any browser:")
    print("  https://accounts.google.com/o/oauth2/auth?x=1", flush=True)
    code = sys.stdin.readline().strip()
    if not code:
        sys.exit(1)
    state.write_text(code); print("you@example.com"); sys.exit(0)
if cmd == "usage":
    print("Current balance: 87.5 compute units"); print("Usage rate: 0/hr"); sys.exit(0)
if cmd in ("new", "stop"):
    (vm / "rf_data").mkdir(parents=True, exist_ok=True); sys.exit(0)
if cmd == "upload":
    src = pathlib.Path(args[-2]).read_text()
    code = re.search(r"^CODE = (.*)$", src, re.M).group(1)
    here(args[-1]).write_text("import os\\nos.chdir(%r)\\nexec(%s, {'__name__': '__main__'})\\n"
                              % (str(vm / "rf_data"), code))
    sys.exit(0)
if cmd == "download":
    shutil.copy(here(args[-2]), args[-1]); sys.exit(0)
if cmd == "exec":
    snippet = pathlib.Path(args[args.index("-f") + 1]).read_text().replace("/content", str(vm))
    if "RF_STARTED" in snippet:
        r = vm / "rf_round"
        r.mkdir(exist_ok=True)
        (r / "started").write_text("0")
        with open(r / "out.log", "w") as out:
            rc = subprocess.run([sys.executable, str(vm / "rf_round.py")], stdout=out,
                                stderr=subprocess.STDOUT).returncode
        (r / "rc").write_text(str(rc))
        print("RF_STARTED 1"); sys.exit(0)
    exec(compile(snippet, "snippet", "exec"), {"__name__": "__main__"})
    sys.exit(0)
sys.exit(2)
"""


@pytest.fixture(autouse=True)
def _isolated(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: home))
    for key in ("RESEARCHFORGE_OVERNIGHT_DIR", "CLAUDECODE", "AI_AGENT", "CLAUDE_CODE_EXECPATH"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("KAGGLE_USERNAME", "me")
    monkeypatch.setenv("KAGGLE_KEY", "secret")
    colab.forget_cache()
    return home


def _fake_cli(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, signed_in: bool = False) -> Path:
    cli = tmp_path / "colab"
    cli.write_text(f"#!{sys.executable}\n{FAKE_COLAB}")
    cli.chmod(0o755)
    state = tmp_path / "colab_state"
    if signed_in:
        state.write_text("ok")
    monkeypatch.setenv("FAKE_COLAB_STATE", str(state))
    monkeypatch.setattr(colab, "find_cli", lambda: str(cli))
    return state


def _project(root: Path) -> Path:
    d = root / f".overnight_{SLUG}"
    d.mkdir(parents=True)
    contract = {"objective": {"primary_metric": {"name": "score", "direction": "maximize"}}}
    (d / "overnight_setup.json").write_text(
        json.dumps({"competition_slug": SLUG, "selected_gpu": "colab_gpu", "contract": contract})
    )
    (d / "pending_hypotheses.json").write_text(json.dumps([{"idea": "a"}]))
    loop = OvernightExperimentLoop.__new__(OvernightExperimentLoop)
    loop.db_path = d / "experiment_log.db"
    loop._init_database()
    return d


class TestSettings:
    def test_the_plan_limits_gpu_and_high_ram(self, tmp_path: Path) -> None:
        d = _project(tmp_path)
        assert colab.ColabConfig.load(d).gpu == "T4"  # free by default
        with pytest.raises(ValueError, match="T4"):
            colab.set_config(d, gpu="A100")
        with pytest.raises(ValueError, match="Pro"):
            colab.set_config(d, high_mem=True)
        cfg = colab.set_config(d, plan="pro", gpu="a100", high_mem=True, via="dashboard")
        assert (cfg.plan, cfg.gpu, cfg.high_mem) == ("pro", "A100", True)
        cfg = colab.set_config(d, plan="free")  # downgrading drops what free cannot use
        assert (cfg.gpu, cfg.high_mem) == ("T4", False)
        fields = {c["field"] for c in run_config.change_log(d)}
        assert {"colab_plan", "colab_gpu", "colab_high_mem"} <= fields

    def test_run_configs_edits_colab_through_the_same_setter(self, tmp_path: Path) -> None:
        d = _project(tmp_path)
        run_config.set_field(d, "colab_plan", "payg", via="dashboard")
        run_config.set_field(d, "colab_gpu", "L4", via="dashboard")
        view = run_config.config_view(d)["colab"]
        assert (view["plan"], view["gpu"]) == ("payg", "L4")
        assert view["plan_gpus"]["free"] == ["T4"] and "pro" in view["high_mem_plans"]


class TestCli:
    def test_status_never_waits_on_a_sign_in(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(colab, "find_cli", lambda: "")
        assert colab.status()["installed"] is False
        _fake_cli(tmp_path, monkeypatch)
        st = colab.status()
        assert st["installed"] and st["signed_in"] is False and "driver-login colab" in st["fix"]

    def test_signing_in_from_the_dashboard(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _fake_cli(tmp_path, monkeypatch)
        flow = colab.SignIn()
        url = flow.start(timeout=20)
        assert url.startswith("https://accounts.google.com/")
        st = flow.finish("4/abc", timeout=20)
        assert st["signed_in"] and st["account"] == "you@example.com"
        assert colab.usage() == {"balance": "87.5 compute units", "rate": "0/hr"}


class TestRounds:
    def _runner(self, d: Path) -> object:
        from researchforge.kaggle.benchmark_runner import BenchmarkRunner
        from researchforge.kaggle.gpu_detector import GPUBackend

        r = BenchmarkRunner.__new__(BenchmarkRunner)
        r.gpu = GPUBackend.COLAB_GPU
        r.baseline_script = None
        metric = SimpleNamespace(direction="maximize", name="score")
        r.setup = SimpleNamespace(  # type: ignore[assignment]
            artifacts_dir=d,
            competition_slug=SLUG,
            baseline_result=SimpleNamespace(cv_score=0.5),
            contract=SimpleNamespace(objective=SimpleNamespace(primary_metric=metric)),
            competition_metadata=None,
        )
        return r

    def test_a_round_runs_on_colab_and_its_score_is_read(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        state = _fake_cli(tmp_path, monkeypatch, signed_in=True)
        d = _project(tmp_path)
        colab.set_config(d, plan="pro", gpu="L4")
        monkeypatch.setattr(colab, "POLL_SECONDS", 0.01)
        code = "print('score: 0.61'); open('submission.csv', 'w').write('id,y\\n1,0\\n')"
        res = self._runner(d).run_full_benchmark(code, 1800)  # type: ignore[attr-defined]
        assert res.passed and res.metric_value == pytest.approx(0.61) and res.metric_improved
        calls = [
            c
            for c in Path(str(state) + ".argv").read_text().splitlines()
            if c.split()[0] not in ("whoami", "usage")  # the readiness check
        ]
        assert calls[0].startswith("new ") and "--gpu L4" in calls[0]
        assert any(c.startswith("exec ") and "--env KAGGLE_USERNAME=me" in c for c in calls)
        assert calls[-1].startswith("stop ")
        kept = Path(res.outputs["submission_file"])
        assert kept.read_text() == "id,y\n1,0\n" and kept.is_relative_to(d / "round_outputs")

    def test_nothing_runs_here_when_colab_is_not_ready(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _fake_cli(tmp_path, monkeypatch)  # installed, signed out
        d = _project(tmp_path)
        res = self._runner(d).run_full_benchmark("print('score: 0.9')", 60)  # type: ignore[attr-defined]
        assert not res.passed and "Colab is not ready" in (res.error_message or "")

    def test_auto_run_waits_for_colab(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _fake_cli(tmp_path, monkeypatch)
        d = _project(tmp_path)
        assert any("Colab is not ready" in g for g in ar.gates(d, ar.AutoRunConfig.load(d)))


def test_the_vm_signs_in_to_kaggle_like_this_machine(
    _isolated: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    assert colab.kaggle_env() == {"KAGGLE_USERNAME": "me", "KAGGLE_KEY": "secret"}
    token = _isolated / ".kaggle" / "access_token"
    token.parent.mkdir()
    token.write_text("KGAT_x\n")
    assert colab.kaggle_env() == {"KAGGLE_API_TOKEN": "KGAT_x"}  # the current CLI's token wins
