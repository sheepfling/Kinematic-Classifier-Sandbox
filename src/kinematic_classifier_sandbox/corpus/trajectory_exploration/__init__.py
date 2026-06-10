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
from .objective_generation import (
    ClassPairRegionSpec,
    FeatureAxisSpec,
    FeatureBinSpec,
    FeatureRowSpec,
    GeneratedTrajectoryObjectiveArtifacts,
    GeneratedTrajectoryObjectiveSuite,
    NoveltyRegionSpec,
    TrajectoryObjectiveGenerationSpec,
    default_objective_generation_spec,
    generated_trajectory_exploration_objectives,
    generate_trajectory_exploration_objective_suite,
    resolve_generated_trajectory_objective,
    write_generated_trajectory_objective_artifacts,
)
from .ppo_boundary_control import (
    SequentialPpoArtifacts,
    SequentialPpoConfig,
    SequentialPpoResult,
    SequentialPpoSweepArtifacts,
    analyze_sequential_ppo_boundary_control,
    has_stable_baselines3_support,
    write_generated_trajectory_objective_ppo_sweep_artifacts,
    write_sequential_ppo_boundary_control_artifacts,
)
from .sequential_gym import (
    SequentialBoundaryControlConfig,
    SequentialTrajectoryGym,
    default_boundary_control_objective,
    evaluate_control_sequence,
    sequential_environment_contract,
)

__all__ = [
    "BlackBoxOptimizerBackend",
    "ClassPairRegionSpec",
    "FeatureAxisSpec",
    "FeatureBinSpec",
    "FeatureRowSpec",
    "GeneratedTrajectoryObjectiveArtifacts",
    "GeneratedTrajectoryObjectiveSuite",
    "NoveltyRegionSpec",
    "HeuristicSearchBackend",
    "SequentialBoundaryControlConfig",
    "SequentialPpoArtifacts",
    "SequentialPpoConfig",
    "SequentialPpoResult",
    "SequentialPpoSweepArtifacts",
    "SequentialTrajectoryGym",
    "StatelessRlPolicyBackend",
    "TrajectoryExplorationArtifacts",
    "TrajectoryExplorationBackend",
    "TrajectoryExplorationBenchmarkArtifacts",
    "TrajectoryExplorationBenchmarkResult",
    "TrajectoryExplorationEvaluation",
    "TrajectoryExplorationObjective",
    "TrajectoryObjectiveGenerationSpec",
    "TrajectoryExplorationProposal",
    "TrajectoryExplorationResult",
    "adapt_adaptive_stress_result",
    "adapt_feature_gap_result",
    "adapt_quality_diversity_result",
    "adapt_search_baseline_result",
    "analyze_sequential_ppo_boundary_control",
    "analyze_trajectory_exploration_benchmarks",
    "default_boundary_control_objective",
    "default_objective_generation_spec",
    "generated_trajectory_exploration_objectives",
    "default_trajectory_exploration_objectives",
    "evaluate_control_sequence",
    "generate_trajectory_exploration_objective_suite",
    "has_stable_baselines3_support",
    "resolve_generated_trajectory_objective",
    "run_trajectory_exploration_backend",
    "sequential_environment_contract",
    "trajectory_exploration_evaluation_schema",
    "trajectory_exploration_objective_schema",
    "write_generated_trajectory_objective_artifacts",
    "write_generated_trajectory_objective_ppo_sweep_artifacts",
    "write_sequential_ppo_boundary_control_artifacts",
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
