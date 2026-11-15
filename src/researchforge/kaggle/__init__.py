"""Kaggle Overnight: Improve your Kaggle score overnight.

Main entry point for automated experimentation on Kaggle competitions.
"""

from researchforge.kaggle.benchmark_runner import BenchmarkRunner
from researchforge.kaggle.competition_reader import CompetitionMetadata, read_competition_metadata
from researchforge.kaggle.experiment_loop import (
    BenchmarkResult,
    ExperimentRound,
    OvernightExperimentLoop,
    RunSummary,
    ScreeningResult,
)
from researchforge.kaggle.gpu_detector import GPUBackend
from researchforge.kaggle.human_questions import (
    Question,
    UserAnswers,
    ask_questions_interactive,
    generate_setup_questions,
)
from researchforge.kaggle.idea_sources import IdeaCollection, IdeaSource, fetch_sources
from researchforge.kaggle.orchestrator import OvernightError, OvernightOrchestrator, OvernightSetup
from researchforge.kaggle.patch_proposer import PatchProposal, PatchProposer

__all__ = [
    # Orchestration
    "OvernightOrchestrator",
    "OvernightSetup",
    "OvernightError",
    # Competition intelligence
    "CompetitionMetadata",
    "read_competition_metadata",
    # Human-in-the-loop
    "Question",
    "UserAnswers",
    "ask_questions_interactive",
    "generate_setup_questions",
    # Idea sources
    "IdeaCollection",
    "IdeaSource",
    "fetch_sources",
    # Experiment loop
    "OvernightExperimentLoop",
    "ExperimentRound",
    "RunSummary",
    "ScreeningResult",
    "BenchmarkResult",
    # AI patch proposer
    "PatchProposal",
    "PatchProposer",
    # Execution
    "BenchmarkRunner",
    "GPUBackend",
]
