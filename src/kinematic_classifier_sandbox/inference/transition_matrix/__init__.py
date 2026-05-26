from __future__ import annotations

from .artifact_io import write_transition_benchmark_artifacts
from .contracts import (
    SwitchingModeSpec,
    SwitchingScenario,
    TransitionBenchmarkArtifacts,
    TransitionBenchmarkResult,
    TransitionBenchmarkSummary,
    TransitionPosteriorStep,
    TransitionRun,
)
from .reporting import render_transition_benchmark_report, render_transition_numeric_walkthrough_markdown
from .runner import (
    _run_mode_accumulator,
    default_switching_mode_specs,
    default_transition_matrix,
    generate_transition_switching_scenarios,
    run_transition_benchmark,
)

__all__ = [
    "SwitchingModeSpec",
    "SwitchingScenario",
    "TransitionPosteriorStep",
    "TransitionRun",
    "TransitionBenchmarkSummary",
    "TransitionBenchmarkResult",
    "TransitionBenchmarkArtifacts",
    "default_switching_mode_specs",
    "default_transition_matrix",
    "generate_transition_switching_scenarios",
    "run_transition_benchmark",
    "render_transition_benchmark_report",
    "render_transition_numeric_walkthrough_markdown",
    "write_transition_benchmark_artifacts",
    "_run_mode_accumulator",
]
