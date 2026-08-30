"""Reading a run's benchmark output back into an observation."""

from pathlib import Path

from researchforge.ai.observation_gen import (
    MAX_LOG_CHARS,
    MAX_OBSERVATION_CHARS,
    ObservationRequest,
    build_prompt,
    generate_observation,
    read_log_tail,
)

LOG = """\
epoch 1 loss 0.91
epoch 2 loss 0.44
epoch 3 loss 0.41
WARNING: dataset truncated to 800 samples
mAP 0.7600
"""

REQUEST = ObservationRequest(
    experiment_id="exp-004",
    title="Lower the confidence threshold",
    change_summary="Sets CONF_THRESHOLD=0.001",
    metric_name="mAP",
    baseline_value=0.74,
    measured_value=0.76,
    status="promising",
    log_tail=LOG,
)


class StubProvider:
    """Returns one canned response and records what it was asked."""

    def __init__(self, response: str) -> None:
        self.response = response
        self.system = ""
        self.user = ""
        self.calls = 0

    @property
    def name(self) -> str:
        return "stub/stub"

    def generate(self, system: str, user: str, *, max_tokens: int = 8192) -> str:
        self.calls += 1
        self.system = system
        self.user = user
        return self.response


class TestReadLogTail:
    def test_reads_a_whole_short_log(self, tmp_path: Path) -> None:
        log = tmp_path / "stdout.log"
        log.write_text(LOG, encoding="utf-8")

        assert read_log_tail(log) == LOG

    def test_keeps_the_end_of_a_long_log(self, tmp_path: Path) -> None:
        log = tmp_path / "stdout.log"
        log.write_text("z" * 20_000 + "final line", encoding="utf-8")

        tail = read_log_tail(log)

        assert len(tail) == MAX_LOG_CHARS
        assert tail.endswith("final line")

    def test_a_missing_log_reads_as_empty(self, tmp_path: Path) -> None:
        assert read_log_tail(tmp_path / "nope.log") == ""

    def test_a_directory_reads_as_empty(self, tmp_path: Path) -> None:
        assert read_log_tail(tmp_path) == ""

    def test_undecodable_bytes_do_not_raise(self, tmp_path: Path) -> None:
        log = tmp_path / "stdout.log"
        log.write_bytes(b"loss 0.4 \xff\xfe done")

        assert "done" in read_log_tail(log)


class TestPrompt:
    def test_names_the_experiment_and_its_change(self) -> None:
        prompt = build_prompt(REQUEST)

        assert "exp-004" in prompt
        assert "Lower the confidence threshold" in prompt
        assert "Sets CONF_THRESHOLD=0.001" in prompt

    def test_includes_the_log_and_the_outcome_on_record(self) -> None:
        prompt = build_prompt(REQUEST)

        assert "WARNING: dataset truncated" in prompt
        assert "mAP" in prompt
        assert "0.76" in prompt
        assert "0.74" in prompt
        assert "promising" in prompt

    def test_an_unmeasured_run_says_so_rather_than_showing_a_number(self) -> None:
        prompt = build_prompt(
            ObservationRequest(
                experiment_id="exp-005",
                title="Broken change",
                change_summary="Sets EPOCHS=0",
                metric_name="mAP",
                baseline_value=0.74,
                measured_value=None,
                status="failed",
                log_tail="Traceback (most recent call last):",
            )
        )

        assert "no usable measurement" in prompt
        assert "Traceback" in prompt


class TestGenerateObservation:
    def test_returns_the_tagged_paragraph(self) -> None:
        provider = StubProvider(
            "<observation>\nLoss was still falling at the final epoch and the run "
            "warned the dataset was truncated to 800 samples.\n</observation>"
        )

        observation = generate_observation(REQUEST, provider)

        assert observation is not None
        assert observation.startswith("Loss was still falling")
        assert "truncated to 800 samples" in observation

    def test_untagged_prose_is_still_accepted(self) -> None:
        provider = StubProvider("The run looks healthy; loss decreased every epoch.")

        assert generate_observation(REQUEST, provider) == (
            "The run looks healthy; loss decreased every epoch."
        )

    def test_whitespace_is_collapsed_into_one_paragraph(self) -> None:
        provider = StubProvider("<observation>Loss fell.\n\n  Then it   plateaued.</observation>")

        assert generate_observation(REQUEST, provider) == "Loss fell. Then it plateaued."

    def test_a_long_answer_is_truncated(self) -> None:
        provider = StubProvider("<observation>" + "word " * 500 + "</observation>")

        observation = generate_observation(REQUEST, provider)

        assert observation is not None
        assert len(observation) == MAX_OBSERVATION_CHARS

    def test_an_empty_answer_reads_as_no_observation(self) -> None:
        provider = StubProvider("<observation>   </observation>")

        assert generate_observation(REQUEST, provider) is None

    def test_an_empty_log_is_not_sent_to_the_ai_at_all(self) -> None:
        provider = StubProvider("<observation>invented</observation>")
        request = ObservationRequest(
            experiment_id="exp-006",
            title="No output",
            change_summary="Sets FLAG=1",
            metric_name="mAP",
            baseline_value=0.74,
            measured_value=0.74,
            status="rejected",
            log_tail="   \n  \n",
        )

        assert generate_observation(request, provider) is None
        assert provider.calls == 0

    def test_the_system_prompt_forbids_recommending_a_decision(self) -> None:
        provider = StubProvider("<observation>fine</observation>")

        generate_observation(REQUEST, provider)

        assert "Never recommend accepting, rejecting or shipping" in provider.system
        assert "untrusted data" in provider.system
