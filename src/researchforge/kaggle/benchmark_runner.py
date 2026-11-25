"""Execute experiment patches and measure metrics.

Two execution paths — selected automatically based on setup.selected_gpu:

  CPU (local):
    Write the patched script, run it with subprocess, parse stdout for metric.

  Kaggle GPU / Competition GPU:
    Write the patched script into a kernel directory, push to Kaggle,
    poll until complete, download output, parse for metric.
    Screening is skipped (Kaggle has no "fast subset" mode) — every run
    is a full benchmark.
"""

from __future__ import annotations

import json
import os
import re
import sys
import tempfile
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

from researchforge.execution.runner import SubprocessRunner
from researchforge.kaggle.gpu_detector import GPUBackend

if TYPE_CHECKING:
    from researchforge.kaggle.experiment_loop import BenchmarkResult, ScreeningResult
    from researchforge.kaggle.orchestrator import OvernightSetup


# A printed metric: sign, integer or decimal, optional exponent.
_NUMBER = r"(-?[0-9]+(?:\.[0-9]+)?(?:[eE][-+]?[0-9]+)?)"


# A run that finished and wrote a submission file but printed no score. Public notebooks
# for hidden-test competitions do this: only Kaggle's leaderboard can score them. The
# round is kept unscored with its file, so it can still be offered for submission
# (kaggle.submission.offer_missed). Older rounds carry the earlier wordings.
UNSCORED_REASON = "Ran to the end and wrote a submission file, but printed no score (unscored)"
# A notebook round whose collected output had neither a score nor a submission file.
NO_SUBMISSION_REASON = "Kernel completed but no metric found in output files"
UNSCORED_PATTERN = re.compile(
    r"printed no score \(unscored\)|no metric found in output files|Could not parse metric"
)

# A round whose submission is not ./submission.* names it on its own line.
_SUBMISSION_LINE = re.compile(r"^SUBMISSION_FILE:\s*(\S.*?)\s*$", re.M)


# A round that fixes a failed submission says whether the error is gone, and how it knows.
_FIX_CHECK_LINE = re.compile(r"^FIX_CHECK:\s*(passed|failed)\b[\s:—-]*(.*?)\s*$", re.M | re.I)


def declared_fix_check(text: str) -> dict[str, object] | None:
    """The last ``FIX_CHECK: passed|failed <evidence>`` line a run printed, or None."""
    found = _FIX_CHECK_LINE.findall(text or "")
    if not found:
        return None
    verdict, evidence = found[-1]
    return {"passed": verdict.lower() == "passed", "evidence": evidence[:500]}


def _with_fix_check(outputs: dict[str, object] | None, text: str) -> dict[str, object] | None:
    check = declared_fix_check(text)
    return {**(outputs or {}), "fix_check": check} if check else outputs


def declared_submission_file(text: str) -> str:
    """The last ``SUBMISSION_FILE: <path>`` line a run printed, or ""."""
    found = _SUBMISSION_LINE.findall(text or "")
    return found[-1] if found else ""


def kernel_log_text(path: Path) -> str:
    """What a Kaggle run printed. Its log is a JSON list of {stream_name, data} chunks."""
    try:
        raw = path.read_text(errors="replace")
    except OSError:
        return ""
    try:
        chunks = json.loads(raw)
    except ValueError:
        return raw
    if not isinstance(chunks, list):
        return raw
    return "".join(str(c.get("data", "")) for c in chunks if isinstance(c, dict))


def datetime_stamp() -> str:
    from datetime import datetime

    return datetime.now().strftime("%Y%m%d-%H%M%S-%f")


def _signed_match(pattern: str, text: str) -> float | None:
    """Group 1 of the first match as a float, keeping a minus sign written just before it."""
    try:
        m = re.search(pattern, text)
    except re.error:
        return None
    if not m:
        return None
    try:
        value = float(m.group(1))
    except (ValueError, IndexError):
        return None
    if value > 0 and m.start(1) > 0 and text[m.start(1) - 1] == "-":
        value = -value
    return value


class BenchmarkRunner:
    """Executes experiment patches and returns metric results."""

    def __init__(self, setup: OvernightSetup) -> None:
        self.setup = setup
        self.gpu = setup.selected_gpu or GPUBackend.CPU

        # Bug 14 fix: Kaggle GPU experiments must never access local data.
        # The data lives on Kaggle's servers, attached to the kernel automatically.
        if self.gpu in (GPUBackend.KAGGLE_GPU, GPUBackend.COMPETITION_GPU):
            # Guard: if someone passes a local data path into Kaggle GPU code, fail loudly.
            self._local_data_forbidden = True
        else:
            self._local_data_forbidden = False

        # Source directory for the baseline kernel code (pulled by notebook_baseline)
        baseline_source = setup.artifacts_dir / "baseline" / "source"
        if baseline_source.exists():
            scripts = list(baseline_source.glob("*.py"))
            self.baseline_script: Path | None = scripts[0] if scripts else None
        else:
            self.baseline_script = None

        # For CPU/SSH: local working directory (Kaggle GPU ignores this)
        if self.gpu in (GPUBackend.KAGGLE_GPU, GPUBackend.COMPETITION_GPU):
            self.working_dir = setup.artifacts_dir  # unused for remote targets
        elif setup.competition_dir and setup.competition_dir.exists():
            self.working_dir = setup.competition_dir
        else:
            self.working_dir = setup.data_dir

        self._subprocess_runner = SubprocessRunner()

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def run_screening(self, patch_code: str, time_budget: float) -> ScreeningResult:
        """Run a fast screening phase.

        For CPU: runs with --screening flag on a data subset.
        For Kaggle GPU / SSH: screening is skipped (pass-through) because remote
        startup cost dominates, making a separate screening step wasteful.
        """
        from researchforge.kaggle.experiment_loop import ScreeningResult

        if self.gpu in (
            GPUBackend.KAGGLE_GPU,
            GPUBackend.COMPETITION_GPU,
            GPUBackend.SSH_SERVER,
            GPUBackend.CLOUD_GPU,
            GPUBackend.COLAB_GPU,
        ):
            return ScreeningResult(
                passed=True,
                metric_value=None,
                duration_seconds=0.0,
                error_message=None,
            )

        return self._run_local(patch_code, time_budget, screening=True)

    def run_full_benchmark(self, patch_code: str, time_budget: float) -> BenchmarkResult:
        """Run the full benchmark with the patch applied."""

        if self.gpu in (GPUBackend.KAGGLE_GPU, GPUBackend.COMPETITION_GPU):
            return self._run_kaggle_gpu(patch_code, time_budget)

        if self.gpu == GPUBackend.SSH_SERVER:
            return self._run_ssh_server(patch_code, time_budget)

        if self.gpu == GPUBackend.CLOUD_GPU:
            return self._run_cloud_gpu_server(patch_code, time_budget)

        if self.gpu == GPUBackend.COLAB_GPU:
            # Never fall back to running here: Colab was chosen so data and GPU stay there.
            return self._run_colab_full(patch_code, time_budget)

        return self._run_local_full(patch_code, time_budget)

    def _run_colab_full(self, patch_code: str, time_budget: float) -> BenchmarkResult:
        """One round on a fresh Colab VM through Google's Colab CLI."""
        from researchforge.kaggle import colab
        from researchforge.kaggle.experiment_loop import BenchmarkResult

        ok, detail, fix = colab.ready()
        if not ok:
            return BenchmarkResult(
                passed=False,
                metric_value=0.0,
                metric_improved=False,
                duration_seconds=0.0,
                error_message=f"Colab is not ready ({detail}); nothing ran. Fix: {fix}",
            )
        start = time.monotonic()
        with tempfile.TemporaryDirectory() as tmpdir:
            script = self._write_patched_script(patch_code, Path(tmpdir))
            result = colab.run_round(
                script.read_text(),
                self.setup.competition_slug,
                colab.ColabConfig.load(self.setup.artifacts_dir),
                Path(tmpdir) / "colab",
                time_budget,
            )
            kept = self._keep_file(str(result.get("submission_file") or ""))
        duration = time.monotonic() - start
        metric = self._parse_metric(result) if result["ok"] else None
        if metric is None:
            err = (result.get("stderr_text") or "").strip()[-300:]
            return BenchmarkResult(
                passed=False,
                metric_value=0.0,
                metric_improved=False,
                duration_seconds=duration,
                error_message=(
                    (UNSCORED_REASON if kept else "Could not parse metric from the Colab output")
                    if result["ok"]
                    else f"Colab exit {result.get('exit_code')}: {err}"
                ),
                outputs={"submission_file": kept} if kept and result["ok"] else None,
            )
        baseline = (
            self.setup.baseline_result.cv_score if self.setup.baseline_result else None
        ) or 0.0
        return BenchmarkResult(
            passed=True,
            metric_value=metric,
            metric_improved=self._metric_improved(metric, baseline),
            duration_seconds=duration,
            outputs={"submission_file": kept} if kept else None,
        )

    def _keep_file(self, path: str) -> str:
        """Keep a submission file fetched from a remote VM, per round (see below)."""
        import shutil

        src = Path(path) if path else None
        if src is None or not src.is_file():
            return ""
        dest = self.setup.artifacts_dir / "round_outputs" / datetime_stamp()
        dest.mkdir(parents=True, exist_ok=True)
        return str(shutil.copy2(src, dest / f"submission{src.suffix}"))

    # ------------------------------------------------------------------
    # Local CPU execution
    # ------------------------------------------------------------------

    def _run_local(self, patch_code: str, time_budget: float, screening: bool) -> ScreeningResult:
        from researchforge.kaggle.experiment_loop import ScreeningResult

        start = time.monotonic()
        with tempfile.TemporaryDirectory() as tmpdir:
            script = self._write_patched_script(patch_code, Path(tmpdir))
            flags = ["--screening"] if screening else []
            result = self._exec_local(script, time_budget, flags)
        duration = time.monotonic() - start

        if result["ok"]:
            metric = self._parse_metric(result)
            return ScreeningResult(
                passed=metric is not None,
                metric_value=metric,
                duration_seconds=duration,
            )
        return ScreeningResult(
            passed=False,
            metric_value=None,
            duration_seconds=duration,
            error_message=f"Exit code {result['exit_code']}: {result['stderr_text'][:200]}",
        )

    def _run_local_full(self, patch_code: str, time_budget: float) -> BenchmarkResult:
        from researchforge.kaggle.experiment_loop import BenchmarkResult

        start, started_at = time.monotonic(), time.time()
        with tempfile.TemporaryDirectory() as tmpdir:
            script = self._write_patched_script(patch_code, Path(tmpdir))
            result = self._exec_local(script, time_budget, [])
        duration = time.monotonic() - start
        kept = self._keep_submission_file(
            started_at, declared_submission_file(str(result.get("stdout_text") or ""))
        )

        # cv_score can be None even when baseline_result exists (e.g. "no_baseline"
        # status from setup) — `or 0.0` only guards a falsy baseline_result itself.
        baseline_score = (
            self.setup.baseline_result.cv_score if self.setup.baseline_result else None
        ) or 0.0
        if result["ok"]:
            metric = self._parse_metric(result)
            if metric is None:
                return BenchmarkResult(
                    passed=False,
                    metric_value=0.0,
                    metric_improved=False,
                    duration_seconds=duration,
                    error_message=UNSCORED_REASON if kept else "Could not parse metric from output",
                    outputs=_with_fix_check(
                        {"submission_file": kept} if kept else None,
                        str(result.get("stdout_text") or ""),
                    ),
                )
            improved = self._metric_improved(metric, baseline_score)
            return BenchmarkResult(
                passed=True,
                metric_value=metric,
                metric_improved=improved,
                duration_seconds=duration,
                outputs=_with_fix_check(
                    {"submission_file": kept} if kept else None,
                    str(result.get("stdout_text") or ""),
                ),
            )
        return BenchmarkResult(
            passed=False,
            metric_value=baseline_score or 0.0,
            metric_improved=False,
            duration_seconds=duration,
            error_message=f"Exit code {result['exit_code']}: {result['stderr_text'][:200]}",
        )

    def _run_ssh_server(self, patch_code: str, time_budget: float) -> BenchmarkResult:
        """Run experiment on a configured remote SSH server."""

        from researchforge.kaggle.compute_router import (
            ExperimentSpec,
            RoutingDecision,
            _run_ssh,
        )
        from researchforge.kaggle.experiment_loop import BenchmarkResult

        spec = ExperimentSpec(
            hypothesis_id="ssh_run",
            code=patch_code,
            title="SSH experiment",
        )
        decision = RoutingDecision(
            target="ssh_server",
            reason="SSH_SERVER backend selected",
            data_path=str(self.setup.data_dir),
        )

        result = _run_ssh(spec, decision, self.setup, timeout=time_budget)

        baseline_score = (
            self.setup.baseline_result.cv_score if self.setup.baseline_result else None
        ) or 0.0
        if result["passed"] and result["metric_value"] is not None:
            metric = result["metric_value"]
            improved = self._metric_improved(metric, baseline_score)
            return BenchmarkResult(
                passed=True,
                metric_value=metric,
                metric_improved=improved,
                duration_seconds=result["duration_seconds"],
            )
        return BenchmarkResult(
            passed=False,
            metric_value=baseline_score,
            metric_improved=False,
            duration_seconds=result["duration_seconds"],
            error_message=result.get("error") or "SSH experiment failed",
        )

    def _run_cloud_gpu_server(self, patch_code: str, time_budget: float) -> BenchmarkResult:
        """Provision a cloud GPU instance, run experiment, terminate — via _run_cloud_gpu."""
        from researchforge.kaggle.compute_router import (
            ExperimentSpec,
            RoutingDecision,
            _run_cloud_gpu,
        )
        from researchforge.kaggle.experiment_loop import BenchmarkResult

        spec = ExperimentSpec(
            hypothesis_id="cloud_run",
            code=patch_code,
            title="Cloud GPU experiment",
        )
        decision = RoutingDecision(
            target="cloud_gpu",
            reason="CLOUD_GPU backend selected",
            data_path=str(self.setup.data_dir),
        )

        result = _run_cloud_gpu(spec, decision, self.setup, timeout=time_budget)

        baseline_score = (
            self.setup.baseline_result.cv_score if self.setup.baseline_result else None
        ) or 0.0
        if result["passed"] and result["metric_value"] is not None:
            metric = result["metric_value"]
            improved = self._metric_improved(metric, baseline_score)
            return BenchmarkResult(
                passed=True,
                metric_value=metric,
                metric_improved=improved,
                duration_seconds=result["duration_seconds"],
            )
        return BenchmarkResult(
            passed=False,
            metric_value=baseline_score,
            metric_improved=False,
            duration_seconds=result["duration_seconds"],
            error_message=result.get("error") or "Cloud GPU experiment failed",
        )

    def _keep_submission_file(self, since: float, declared: str = "") -> str:
        """Copy the submission file this run wrote to keep it per round.

        That is the file the run named (``SUBMISSION_FILE: <path>``), else a fresh
        ``submission.*`` in its working dir. Every local round writes to the same
        place, so the next round would overwrite it before anyone could approve it.
        """
        import shutil

        named = Path(declared).expanduser() if declared else None
        if named is not None and not named.is_absolute():
            named = Path(self.working_dir) / named
        if named is not None and named.is_file() and named.stat().st_mtime >= since:
            fresh = [named]
        else:
            fresh = [
                p
                for p in Path(self.working_dir).glob("submission.*")
                if p.is_file() and p.stat().st_mtime >= since
            ]
        if not fresh:
            return ""
        src = max(fresh, key=lambda p: p.stat().st_mtime)
        dest = self.setup.artifacts_dir / "round_outputs" / datetime_stamp()
        dest.mkdir(parents=True, exist_ok=True)
        return str(shutil.copy2(src, dest / f"submission{src.suffix}"))

    def _exec_local(self, script: Path, timeout_seconds: float, flags: list[str]) -> dict[str, Any]:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            stdout_path = tmp / "stdout.txt"
            stderr_path = tmp / "stderr.txt"
            outcome = self._subprocess_runner.run(
                [sys.executable, str(script)] + flags,
                cwd=self.working_dir,
                env=os.environ.copy(),
                timeout_seconds=timeout_seconds,
                stdout_path=stdout_path,
                stderr_path=stderr_path,
            )
            return {
                "exit_code": outcome.exit_code,
                "ok": outcome.ok,
                "timed_out": outcome.timed_out,
                "stdout_text": stdout_path.read_text() if stdout_path.exists() else "",
                "stderr_text": stderr_path.read_text() if stderr_path.exists() else "",
                "duration_seconds": outcome.duration_seconds,
            }

    # ------------------------------------------------------------------
    # Kaggle GPU execution  (push → poll → pull → parse)
    # ------------------------------------------------------------------

    def _run_kaggle_gpu(
        self,
        patch_code: str,
        time_budget: float,
        dataset_sources: list[str] | None = None,
        inputs: dict[str, Any] | None = None,
    ) -> BenchmarkResult:
        """One round as a Kaggle notebook. ``inputs`` attaches what a ported public
        notebook needs: other notebooks' outputs, Kaggle Models, its Docker image and
        GPU type (see ``api.kernel_inputs``)."""
        from researchforge.kaggle.api import (
            KaggleAPIError,
            build_kernel_metadata,
            get_kaggle_username,
            poll_kernel_until_complete,
            push_kernel_versioned,
        )
        from researchforge.kaggle.experiment_loop import BenchmarkResult

        # cv_score can be None even when baseline_result exists (e.g. "no_baseline"
        # status from setup) — `or 0.0` only guards a falsy baseline_result itself.
        baseline_score = (
            self.setup.baseline_result.cv_score if self.setup.baseline_result else None
        ) or 0.0
        start = time.monotonic()

        # Build a unique slug for this experiment
        import datetime
        import hashlib

        run_id = hashlib.md5(
            (patch_code[:50] + datetime.datetime.now().isoformat()).encode()
        ).hexdigest()[:8]
        slug = f"rf-{self.setup.competition_slug[:20]}-{run_id}".lower()
        slug = re.sub(r"[^a-z0-9-]", "-", slug)[:50]

        try:
            username = get_kaggle_username()
        except KaggleAPIError as exc:
            return BenchmarkResult(
                passed=False,
                metric_value=baseline_score,
                metric_improved=False,
                duration_seconds=time.monotonic() - start,
                error_message=str(exc),
            )

        # Write kernel directory
        with tempfile.TemporaryDirectory() as tmpdir:
            kernel_dir = Path(tmpdir) / slug
            kernel_dir.mkdir()

            script_name = "experiment.py"
            self._write_patched_script(patch_code, kernel_dir, filename=script_name)

            # Write metadata — dataset_sources from proposal (bug #20 fix)
            # title must slugify to exactly `slug`, or Kaggle silently pushes
            # under a slug derived from the title instead of the declared id
            # (bug #23) — every later poll/fetch then targets a ref that
            # doesn't exist and burns the full timeout budget.
            meta = build_kernel_metadata(
                username=username,
                kernel_slug=slug,
                script_filename=script_name,
                competition_slug=self.setup.competition_slug,
                enable_gpu=(self.gpu == GPUBackend.KAGGLE_GPU),
                enable_internet=False,
                title=slug,
                dataset_sources=dataset_sources,
                inputs=inputs,
            )
            (kernel_dir / "kernel-metadata.json").write_text(json.dumps(meta, indent=2))

            # Push
            try:
                kernel_ref, kernel_version = push_kernel_versioned(kernel_dir)
            except KaggleAPIError as exc:
                return BenchmarkResult(
                    passed=False,
                    metric_value=baseline_score,
                    metric_improved=False,
                    duration_seconds=time.monotonic() - start,
                    error_message=f"Push failed: {exc}",
                )

            print(
                f"  Kernel pushed: {kernel_ref} — polling every 30s (up to {time_budget / 60:.0f} "
                "min)...",
                flush=True,
            )

            # Poll until done (up to time_budget).
            # Bug #19 fix: the status includes the raw output on timeout.
            final_status = poll_kernel_until_complete(
                kernel_ref,
                timeout_seconds=time_budget,
                poll_interval_seconds=30,
            )

            # poll returns "complete", "error", or "timeout (after N polls, last raw: ...)"
            if final_status.startswith("timeout"):
                # The kernel keeps running on Kaggle; the monitor collects it when it
                # finishes (late_rounds), so the round is not lost as a 0.0 failure.
                return BenchmarkResult(
                    passed=False,
                    metric_value=baseline_score,
                    metric_improved=False,
                    duration_seconds=time.monotonic() - start,
                    error_message=(
                        f"still running on Kaggle as {kernel_ref} after "
                        f"{time_budget / 60:.0f} min; collected when it finishes"
                    ),
                    outputs={
                        "kernel_ref": kernel_ref,
                        "kernel_version": kernel_version,
                        "pending": True,
                    },
                )
            if not final_status.startswith("complete"):
                return BenchmarkResult(
                    passed=False,
                    metric_value=baseline_score,
                    metric_improved=False,
                    duration_seconds=time.monotonic() - start,
                    error_message=f"Kernel ended with status: {final_status}",
                )
        return self.collect_kernel(kernel_ref, kernel_version, time.monotonic() - start)

    def collect_kernel(
        self, kernel_ref: str, kernel_version: int | None, duration: float
    ) -> BenchmarkResult:
        """Read a completed Kaggle kernel's score and submission file into a result."""
        from researchforge.kaggle.api import fetch_kernel_output
        from researchforge.kaggle.experiment_loop import BenchmarkResult

        baseline_score = (
            self.setup.baseline_result.cv_score if self.setup.baseline_result else None
        ) or 0.0
        output_dir = self.setup.artifacts_dir / "kernel_outputs" / kernel_ref.split("/")[-1]
        # Only the log and the submission file: a notebook can leave hundreds of files.
        output_files = fetch_kernel_output(kernel_ref, output_dir, file_pattern=r"^submission\.")
        # The score is what the run printed: its log, never a data file it wrote.
        metric, declared, check = None, "", None
        for output_file in (p for p in output_files if p.suffix == ".log"):
            text = kernel_log_text(output_file)
            if metric is None:
                metric = self._parse_metric({"stdout_text": text, "stderr_text": ""})
            declared = declared or declared_submission_file(text)
            check = check or declared_fix_check(text)
        if declared and not (output_dir / Path(declared).name).is_file():
            output_files += fetch_kernel_output(
                kernel_ref, output_dir, file_pattern=f"^{re.escape(Path(declared).name)}$"
            )

        wanted = Path(declared).name if declared else ""
        written = sorted(
            p
            for p in output_files
            if p.is_file() and (p.name == wanted or p.name.startswith("submission."))
        )
        written.sort(key=lambda p: p.name != wanted)
        outputs: dict[str, object] = {
            "kernel_ref": kernel_ref,
            "kernel_version": kernel_version,
            "submission_file": str(written[0]) if written else "",
        }
        if check:
            outputs["fix_check"] = check
        if metric is None:
            # The notebook still ran: keep it, so it can be submitted and scored by Kaggle.
            return BenchmarkResult(
                passed=False,
                metric_value=baseline_score,
                metric_improved=False,
                duration_seconds=duration,
                error_message=UNSCORED_REASON if written else NO_SUBMISSION_REASON,
                outputs=outputs,
            )

        return BenchmarkResult(
            passed=True,
            metric_value=metric,
            metric_improved=self._metric_improved(metric, baseline_score),
            duration_seconds=duration,
            outputs=outputs,
        )

    # ------------------------------------------------------------------
    # Patch application
    # ------------------------------------------------------------------

    def _write_patched_script(
        self,
        patch_code: str,
        output_dir: Path,
        filename: str = "experiment.py",
    ) -> Path:
        """Write the patch code as the experiment script.

        The AI generates complete, runnable Python code (not a diff), so we
        write it directly. If the patch_code is empty, fall back to the
        baseline source script.
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        script_path = output_dir / filename

        if patch_code.strip():
            script_path.write_text(patch_code)
        elif self.baseline_script and self.baseline_script.exists():
            script_path.write_text(self.baseline_script.read_text())
        else:
            raise RuntimeError(
                "No patch code provided and no baseline script available. "
                "The AI must provide complete Python code in code_changes."
            )
        return script_path

    # ------------------------------------------------------------------
    # Metric parsing
    # ------------------------------------------------------------------

    def _parse_metric(self, result: dict[str, Any]) -> float | None:
        """Parse the primary metric from benchmark output."""
        stdout_text = result.get("stdout_text", "")
        stderr_text = result.get("stderr_text", "")
        combined = stdout_text + "\n" + stderr_text

        # 1. Competition-specific patterns from AI-extracted metadata. Older ones
        #    capture only unsigned decimals, so a minus sign just before the
        #    capture still counts: a lower-is-better score can be negative.
        meta = getattr(self.setup, "competition_metadata", None)
        if meta and meta.metric_extraction_patterns:
            for pattern in meta.metric_extraction_patterns:
                value = _signed_match(pattern, combined)
                if value is not None:
                    return value

        # 2. Generic patterns
        metric_name = (
            self.setup.contract.objective.primary_metric.name if self.setup.contract else "score"
        )
        fallback_patterns = [
            rf"(?i){re.escape(metric_name)}[:\s=]+{_NUMBER}",
            rf"(?i)CV Score[:\s=]+{_NUMBER}",
            rf"(?i)Full CV Score[:\s=]+{_NUMBER}",
            rf"(?i)Metric[:\s=]+{_NUMBER}",
            rf"(?i)Score[:\s=]+{_NUMBER}",
        ]
        for pattern in fallback_patterns:
            value = _signed_match(pattern, combined)
            if value is not None:
                return value
        # 3. The last line that is only a number. Any long decimal anywhere in the
        #    output once counted, so a notebook's timing or loss read as its score.
        bare = re.findall(r"(?m)^[ \t]*(-?[0-9]+\.[0-9]+)[ \t]*$", stdout_text)
        return float(bare[-1]) if bare else None

    def _metric_improved(self, current: float, baseline: float) -> bool:
        from researchforge.domain.contract import MetricDirection

        maximize = (
            not self.setup.contract
            or self.setup.contract.objective.primary_metric.direction == MetricDirection.MAXIMIZE
        )
        bl = self.setup.baseline_result
        if bl is None or bl.cv_score is None:
            # Callers pass 0.0 when setup measured no baseline; comparing with it
            # made every score "improved" (and offered for submission). Without a
            # baseline, a round must beat the best earlier scored round, and the
            # first scored round only sets that reference.
            best = self._best_scored_so_far(maximize)
            if best is None:
                return False
            baseline = best
        return current > baseline if maximize else current < baseline

    def _best_scored_so_far(self, maximize: bool) -> float | None:
        from researchforge.kaggle.experiment_loop import load_rounds

        scores = [
            r.full_result.metric_value
            for r in load_rounds(self.setup.artifacts_dir / "experiment_log.db")
            if r.full_result is not None and r.full_result.passed
        ]
        if not scores:
            return None
        return max(scores) if maximize else min(scores)
