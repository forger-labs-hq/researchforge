"""Human-in-the-loop (HIL) question engine for Kaggle Overnight.

Before the experiment loop starts, the AI generates a set of focused questions
whose answers it genuinely needs to steer the research.  The user answers them
in the CLI.  The answers are stored in OvernightSetup and fed into every
PatchProposal prompt so the AI never has to guess.

During the loop the AI can also request a mid-loop pause (by returning a
special JSON key in its patch proposal) to ask clarifying questions if it is
stuck or discovers a domain constraint.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from researchforge.ai.providers import AiProvider
    from researchforge.kaggle.competition_reader import CompetitionMetadata


# ---------------------------------------------------------------------------
# Domain models
# ---------------------------------------------------------------------------


QuestionType = Literal["text", "choice", "yes_no", "number"]


@dataclass
class Question:
    id: str
    text: str
    question_type: QuestionType = "text"
    choices: list[str] = field(default_factory=list)
    default: str = ""
    is_required: bool = False
    hint: str = ""  # displayed below the question in dim text


@dataclass
class UserAnswers:
    answers: dict[str, str] = field(default_factory=dict)
    asked_at: str = field(default_factory=lambda: datetime.now().isoformat())

    def get(self, question_id: str, fallback: str = "") -> str:
        return self.answers.get(question_id, fallback)

    def save(self, artifacts_dir: Path) -> None:
        path = artifacts_dir / "user_answers.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w") as fh:
            json.dump(asdict(self), fh, indent=2)

    @classmethod
    def load(cls, artifacts_dir: Path) -> UserAnswers:
        path = artifacts_dir / "user_answers.json"
        if not path.exists():
            return cls()
        with path.open() as fh:
            data = json.load(fh)
        return cls(
            answers=data.get("answers", {}),
            asked_at=data.get("asked_at", ""),
        )


# ---------------------------------------------------------------------------
# AI-driven question generation
# ---------------------------------------------------------------------------


_QUESTION_GEN_SYSTEM = """\
You are a Kaggle expert and research assistant.
Given a competition's metadata, generate a concise list of questions to ask the user
before starting the overnight experiment loop. These are questions whose answers will
meaningfully change the research direction or help avoid wasted GPU time.

Return ONLY valid JSON — a list of objects:
[
  {
    "id": "<snake_case unique id>",
    "text": "<the question to display to the user>",
    "question_type": "<text|choice|yes_no|number>",
    "choices": ["<option1>", "<option2>"],    // only for 'choice' type
    "default": "<sensible default>",
    "is_required": false,
    "hint": "<one-line hint shown below the question>"
  },
  ...
]

Rules:
- Ask at most 5 questions. Quality over quantity.
- Do NOT ask about time budget or GPU — those are handled separately.
- Do NOT ask obvious things that can be read from the competition page.
- Focus on: known winning approaches, domain knowledge to avoid, forbidden features,
  ensemble preference, external data allowance, compute constraints.
- Every question must have a sensible default so the user can just press Enter.
"""


def generate_setup_questions(
    metadata: CompetitionMetadata,
    ai_provider: AiProvider | None,
) -> list[Question]:
    """Use AI to generate competition-specific questions, or return minimal defaults."""
    if ai_provider is None:
        return _default_questions(metadata)

    user_prompt = (
        f"Competition: {metadata.slug}\n"
        f"Title: {metadata.title}\n"
        f"Task: {metadata.task_type}\n"
        f"Metric: {metadata.metric_name} ({metadata.metric_direction})\n"
        f"Description: {metadata.description[:400]}\n"
        f"Data: {metadata.data_description}"
    )
    try:
        raw = ai_provider.generate(_QUESTION_GEN_SYSTEM, user_prompt, max_tokens=1024)
        raw = raw.strip()
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[1].rsplit("```", 1)[0]
        items = json.loads(raw)
        return [
            Question(
                id=q.get("id", f"q{i}"),
                text=q.get("text", ""),
                question_type=q.get("question_type", "text"),
                choices=q.get("choices", []),
                default=q.get("default", ""),
                is_required=q.get("is_required", False),
                hint=q.get("hint", ""),
            )
            for i, q in enumerate(items)
            if q.get("text")
        ]
    except (json.JSONDecodeError, KeyError):
        return _default_questions(metadata)


def _default_questions(metadata: CompetitionMetadata) -> list[Question]:
    """Minimal questions when no AI provider is available."""
    return [
        Question(
            id="known_approaches",
            text=(
                "Are there any techniques you know work well for "
                f"{metadata.title or metadata.slug}?"
            ),
            question_type="text",
            default="",
            hint="E.g. 'LightGBM with target encoding' or 'EfficientNet B4 with TTA'",
        ),
        Question(
            id="forbidden_features",
            text="Are there any features that must NOT be used (leakage, competition rules)?",
            question_type="text",
            default="",
            hint="Leave empty if none",
        ),
        Question(
            id="external_data",
            text="Is external data allowed for this competition?",
            question_type="choice",
            choices=["yes", "no", "unknown"],
            default="unknown",
        ),
    ]


# ---------------------------------------------------------------------------
# Interactive CLI session
# ---------------------------------------------------------------------------


def _is_interactive() -> bool:
    """Return True only when both stdin and stdout are real TTYs."""
    import sys

    return sys.stdin.isatty() and sys.stdout.isatty()


def ask_questions_interactive(questions: list[Question]) -> UserAnswers:
    """Present questions in the terminal and collect answers.

    In non-interactive mode (Claude Code background shell, CI, pipes) the
    questions are printed to stderr so the caller (Claude) can see them and
    answer via CLI flags or a follow-up call, and defaults are used immediately.
    """
    if not questions:
        return UserAnswers()

    if not _is_interactive():
        # Non-interactive: print questions to stderr for Claude to handle,
        # return defaults so the setup flow continues unblocked.
        import sys

        print("\n📋  Domain questions (non-interactive — using defaults):", file=sys.stderr)
        for q in questions:
            print(f"  [{q.id}] {q.text} (default: {q.default!r})", file=sys.stderr)
        print("  Pass answers via --answers-json flag or re-run interactively.", file=sys.stderr)
        return UserAnswers(answers={q.id: q.default for q in questions})

    print("\n" + "─" * 60)
    print("📋  A few quick questions before the loop starts")
    print("    (Press Enter to accept the default shown in brackets)")
    print("─" * 60)

    answers: dict[str, str] = {}
    for q in questions:
        answer = _ask_one(q)
        answers[q.id] = answer

    print("─" * 60 + "\n")
    return UserAnswers(answers=answers)


def _ask_one(q: Question) -> str:
    """Prompt for a single question and return the answer string."""
    print()
    # Question text
    required_marker = " *" if q.is_required else ""
    print(f"  {q.text}{required_marker}")
    if q.hint:
        print(f"  \033[2m{q.hint}\033[0m")  # dim text

    if q.question_type == "choice" and q.choices:
        choices_str = " / ".join(q.choices)
        prompt = f"  [{choices_str}] (default: {q.default or q.choices[0]}): "
    elif q.question_type == "yes_no":
        prompt = f"  [yes/no] (default: {q.default or 'no'}): "
    elif q.question_type == "number":
        prompt = f"  (default: {q.default or '0'}): "
    else:
        prompt = f"  (default: {q.default!r}): " if q.default else "  > "

    while True:
        try:
            raw = input(prompt).strip()
        except EOFError:
            raw = ""

        if not raw:
            raw = q.default

        # Validate choice type
        if q.question_type == "choice" and q.choices and raw not in q.choices:
            if not raw:
                raw = q.choices[0]
            else:
                print(f"  ⚠  Please choose one of: {', '.join(q.choices)}")
                continue
        if q.question_type == "yes_no":
            if raw.lower() in ("y", "yes", "1", "true"):
                raw = "yes"
            elif raw.lower() in ("n", "no", "0", "false"):
                raw = "no"
            else:
                raw = q.default or "no"
        if q.question_type == "number" and raw:
            try:
                float(raw)
            except ValueError:
                print("  ⚠  Please enter a number")
                continue
        if q.is_required and not raw:
            print("  ⚠  This question is required")
            continue
        break

    return raw


# ---------------------------------------------------------------------------
# Mid-loop pause support
# ---------------------------------------------------------------------------


def ask_mid_loop_questions(
    questions: list[Question],
    existing_answers: UserAnswers,
    artifacts_dir: Path,
) -> UserAnswers:
    """Ask additional questions mid-loop and merge with existing answers.

    In non-interactive mode, writes the questions to a file for Claude to
    handle and returns existing answers unchanged so the loop keeps running.
    """
    if not _is_interactive():
        import json

        questions_path = artifacts_dir / "mid_loop_questions.json"
        questions_path.write_text(
            json.dumps(
                [
                    {"id": q.id, "text": q.text, "default": q.default, "hint": q.hint}
                    for q in questions
                ],
                indent=2,
            )
        )
        # Also record each question as a non-blocking interaction: the loop keeps running
        # and `researchforge kaggle respond` merges the answer into user_answers.
        from researchforge.kaggle.control import request_interaction

        for q in questions:
            request_interaction(
                artifacts_dir,
                q.text,
                kind="domain_question",
                options=list(q.choices),
                default=q.default,
                answer_key=q.id,
            )
        import sys

        print(
            f"\n\u23f8  Mid-loop questions written to {questions_path}\n"
            "   The loop keeps running. Answer with: researchforge kaggle respond "
            f"{artifacts_dir.name.removeprefix('.overnight_')}",
            file=sys.stderr,
        )
        return existing_answers

    print("\n\u23f8  The loop is pausing to ask a clarifying question...")
    new_answers = ask_questions_interactive(questions)
    merged = {**existing_answers.answers, **new_answers.answers}
    updated = UserAnswers(answers=merged)
    updated.save(artifacts_dir)
    return updated
