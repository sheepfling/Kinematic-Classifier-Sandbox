from __future__ import annotations

from .backends import (
    BlackBoxOptimizerBackend,
    HeuristicSearchBackend,
    StatelessRlPolicyBackend,
)
from .contracts import (
    TrajectoryExplorationArtifacts,
    TrajectoryExplorationBackend,
    TrajectoryExplorationBenchmarkArtifacts,
    TrajectoryExplorationBenchmarkResult,
    TrajectoryExplorationEvaluation,
    TrajectoryExplorationObjective,
    TrajectoryExplorationProposal,
    TrajectoryExplorationResult,
)
from .objectives import (
    default_trajectory_exploration_objectives,
    trajectory_exploration_evaluation_schema,
    trajectory_exploration_objective_schema,
)

__all__ = [
    "BlackBoxOptimizerBackend",
    "HeuristicSearchBackend",
    "StatelessRlPolicyBackend",
    "TrajectoryExplorationArtifacts",
    "TrajectoryExplorationBackend",
    "TrajectoryExplorationBenchmarkArtifacts",
    "TrajectoryExplorationBenchmarkResult",
    "TrajectoryExplorationEvaluation",
    "TrajectoryExplorationObjective",
    "TrajectoryExplorationProposal",
    "TrajectoryExplorationResult",
    "adapt_adaptive_stress_result",
    "adapt_feature_gap_result",
    "adapt_quality_diversity_result",
    "adapt_search_baseline_result",
    "analyze_trajectory_exploration_benchmarks",
    "default_trajectory_exploration_objectives",
    "run_trajectory_exploration_backend",
    "trajectory_exploration_evaluation_schema",
    "trajectory_exploration_objective_schema",
    "write_trajectory_exploration_artifacts",
]


def run_trajectory_exploration_backend(*args, **kwargs):
    from .runner import run_trajectory_exploration_backend as _impl

    return _impl(*args, **kwargs)


def analyze_trajectory_exploration_benchmarks(*args, **kwargs):
    from .runner import analyze_trajectory_exploration_benchmarks as _impl

    return _impl(*args, **kwargs)


def adapt_search_baseline_result(*args, **kwargs):
    from .runner import adapt_search_baseline_result as _impl

    return _impl(*args, **kwargs)


def adapt_quality_diversity_result(*args, **kwargs):
    from .runner import adapt_quality_diversity_result as _impl

    return _impl(*args, **kwargs)


def adapt_adaptive_stress_result(*args, **kwargs):
    from .runner import adapt_adaptive_stress_result as _impl

    return _impl(*args, **kwargs)


def adapt_feature_gap_result(*args, **kwargs):
    from .runner import adapt_feature_gap_result as _impl

    return _impl(*args, **kwargs)


def write_trajectory_exploration_artifacts(*args, **kwargs):
    from .artifact_io import write_trajectory_exploration_artifacts as _impl

    return _impl(*args, **kwargs)
