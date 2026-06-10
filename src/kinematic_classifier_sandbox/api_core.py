"""Canonical front-door API for the methodology workbench.

This module exposes the curated mainline surface only.
"""

from __future__ import annotations

from .analysis.feature_analysis import (
    BaseFeatureComputationContext,
    FeatureComputationContext,
    OneDimensionalFeatureComputationContext,
    analyze_feature_datasets,
    resolve_feature_names,
)
from .common_experiment.runner import analyze_common_experiment, analyze_common_trajectory_corpus
from .corpus.exploration.feature_gap_trajectory_explorer import (
    analyze_feature_gap_trajectory_explorer,
)
from .corpus.exploration.generic_corpus_exploration import (
    analyze_generic_corpus_exploration,
)
from .corpus.gym import (
    CorpusGymAction,
    CorpusGymEnvironment,
    CorpusGymTarget,
    analyze_corpus_gym_contract,
)
from .corpus.trajectory_exploration import (
    analyze_trajectory_exploration_benchmarks,
    write_trajectory_exploration_artifacts,
)
from .corpus.trajectory_backend_contract import (
    analyze_trajectory_backend_contract,
)
from .inference.kalman_filter_bank import KalmanTrajectory
from .inference.monte_carlo_benchmark import (
    run_accumulator_monte_carlo_benchmark,
)
from .inference.pointwise_baseline import (
    GaussianPointwiseClassifier,
    PointwiseClassSpec,
    PointwiseTrajectory,
    run_pointwise_benchmark,
    run_pointwise_classifier,
)
from .inference.sequential_bayes_accumulator import (
    SequentialBayesAccumulator,
    run_accumulator,
    run_accumulator_benchmark,
)
from .inference.transition_matrix_accumulator import run_transition_benchmark
from .inference.windowed_baseline import run_windowed_benchmark
from .methodology.classification_evidence import (
    EvidenceStep,
    posterior_history_from_evidence_stream,
)
from .rung_sufficiency.analysis import analyze_rung_sufficiency
from .schema.artifacts import (
    ClassifierOutputArtifact,
    TrajectoryArtifact,
    validate_classifier_output_artifact,
    validate_milestone0_sample_run_artifacts,
    validate_trajectory_artifact,
)
from .schema.milestone0 import Milestone0SampleArtifacts
from .validation.shared_evaluation import CallableSharedClassifierAdapter
from .validation.validation_ladder import analyze_validation_ladder

__all__ = [
    "analyze_common_experiment",
    "analyze_common_trajectory_corpus",
    "analyze_corpus_gym_contract",
    "analyze_feature_gap_trajectory_explorer",
    "analyze_generic_corpus_exploration",
    "analyze_trajectory_exploration_benchmarks",
    "analyze_trajectory_backend_contract",
    "run_accumulator_monte_carlo_benchmark",
    "analyze_feature_datasets",
    "analyze_rung_sufficiency",
    "analyze_validation_ladder",
    "ClassifierOutputArtifact",
    "CorpusGymAction",
    "CorpusGymEnvironment",
    "CorpusGymTarget",
    "EvidenceStep",
    "BaseFeatureComputationContext",
    "GaussianPointwiseClassifier",
    "PointwiseClassSpec",
    "PointwiseTrajectory",
    "FeatureComputationContext",
    "KalmanTrajectory",
    "SequentialBayesAccumulator",
    "CallableSharedClassifierAdapter",
    "OneDimensionalFeatureComputationContext",
    "posterior_history_from_evidence_stream",
    "resolve_feature_names",
    "run_accumulator",
    "run_accumulator_benchmark",
    "run_pointwise_benchmark",
    "run_pointwise_classifier",
    "run_transition_benchmark",
    "run_windowed_benchmark",
    "Milestone0SampleArtifacts",
    "validate_classifier_output_artifact",
    "validate_milestone0_sample_run_artifacts",
    "validate_trajectory_artifact",
    "TrajectoryArtifact",
    "write_trajectory_exploration_artifacts",
]
