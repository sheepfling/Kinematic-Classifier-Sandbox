from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ..analysis.common_dataset_comparison import CommonComparisonResult
from ..schema.feature_rows import FeatureValueMappingMixin
from .summary_rows_types import (
    ClassPairDurationSummaryRow,
    ClassPairScenarioSummaryRow,
    CovariateAuditRow,
    FeatureExcitationSummaryRow,
    IdentifiabilitySummaryRow,
    MetricsByClassifierRow,
    MetricsBySensorRegimeRow,
    OracleSummaryRow,
)

if TYPE_CHECKING:
    from .protocols import (
        FeatureExtractor,
        FeatureSigma,
        GaussianLogPdf,
        MeasurementSigma,
        PairSpecBuilder,
        ReferenceBuilder,
        SafeLog,
        TrajectoryGenerator,
    )


@dataclass(frozen=True, slots=True)
class ExecutablePairSpec:
    pair_id: str
    class_a: str
    class_b: str
    expected_difficulty: str


@dataclass(frozen=True, slots=True)
class ExecutableTrajectory:
    trajectory_id: str
    class_pair_id: str
    class_a: str
    class_b: str
    true_class: str
    scenario_id: str
    seed: int
    times: tuple[float, ...]
    measurements: tuple[float, ...]
    true_position: tuple[float, ...]
    true_velocity: tuple[float, ...]
    true_acceleration: tuple[float, ...]
    measurement_dim: int = 1
    coordinate_frame: str = "scalar_line"


@dataclass(frozen=True, slots=True)
class CommonExperimentSummary:
    experiment_name: str
    study_adapter_id: str
    executable_class_pairs: tuple[str, ...]
    trajectories_per_case: int
    num_pair_trajectories: int
    num_pair_predictions: int


@dataclass(frozen=True, slots=True)
class CommonExperimentConfig:
    experiment_name: str
    study_adapter_id: str
    config_path: Path
    output_dir_name: str
    dataset_generator_id: str
    declared_class_pairs: tuple[tuple[str, str], ...]
    output_filenames: dict[str, str]
    feature_sets_path: Path
    class_pair_manifest_path: Path
    classifier_manifest_path: Path


@dataclass(frozen=True, slots=True)
class CommonStudyAdapter:
    study_id: str
    description: str
    pair_spec_builder: "PairSpecBuilder"
    trajectory_generator: "TrajectoryGenerator"


@dataclass(frozen=True, slots=True)
class FamilyScoringContext:
    pair_spec: ExecutablePairSpec
    trajectory: ExecutableTrajectory
    truncated: ExecutableTrajectory
    times: tuple[float, ...]
    prior_weights: dict[str, float]
    feature_manifest: dict[str, dict[str, Any]]
    reference_builder: "ReferenceBuilder"
    feature_extractor: "FeatureExtractor"
    feature_sigma: "FeatureSigma"
    gaussian_logpdf: "GaussianLogPdf"
    safe_log: "SafeLog"
    measurement_sigma: "MeasurementSigma"


@dataclass(frozen=True, slots=True)
class CommonExperimentResult:
    config: CommonExperimentConfig
    summary: CommonExperimentSummary
    comparison: CommonComparisonResult
    pair_prediction_rows: tuple[dict[str, Any], ...]
    posterior_history_rows: tuple[dict[str, Any], ...]
    likelihood_history_rows: tuple[dict[str, Any], ...]
    feature_rows: tuple[dict[str, Any], ...]
    metrics_by_classifier_rows: tuple[MetricsByClassifierRow, ...]
    metrics_by_sensor_regime_rows: tuple[MetricsBySensorRegimeRow, ...]
    metrics_by_classifier_and_feature_set_rows: tuple[dict[str, Any], ...]
    metrics_by_class_pair_rows: tuple[dict[str, Any], ...]
    prior_sensitivity_rows: tuple[dict[str, Any], ...]
    feature_set_comparison_rows: tuple[dict[str, Any], ...]
    irregular_window_rows: tuple[dict[str, Any], ...]
    class_pair_duration_rows: tuple[ClassPairDurationSummaryRow, ...]
    class_pair_scenario_rows: tuple[ClassPairScenarioSummaryRow, ...]
    covariate_rows: tuple[CovariateAuditRow, ...]
    feature_excitation_rows: tuple[FeatureExcitationSummaryRow, ...]
    identifiability_rows: tuple[IdentifiabilitySummaryRow, ...]
    oracle_rows: tuple[OracleSummaryRow, ...]


@dataclass(frozen=True, slots=True)
class PairPredictionRow:
    run_id: str
    classifier_id: str
    feature_set_id: str
    sensor_regime_id: str
    measurement_dim: int
    coordinate_frame: str
    class_pair_id: str
    class_a: str
    class_b: str
    trajectory_id: str
    scenario_id: str
    scenario_family: str
    dataset_tier: str
    time: float
    true_class: str
    predicted_class: str
    confidence: float
    posterior_class_a: float
    posterior_class_b: float


@dataclass(frozen=True, slots=True)
class PosteriorHistoryRow:
    run_id: str
    classifier_id: str
    feature_set_id: str
    sensor_regime_id: str
    class_pair_id: str
    class_a: str
    class_b: str
    trajectory_id: str
    scenario_id: str
    scenario_family: str
    dataset_tier: str
    time: float
    true_class: str
    posterior_class_a: float
    posterior_class_b: float


@dataclass(frozen=True, slots=True)
class LikelihoodHistoryRow:
    run_id: str
    classifier_id: str
    feature_set_id: str
    sensor_regime_id: str
    class_pair_id: str
    trajectory_id: str
    scenario_id: str
    scenario_family: str
    dataset_tier: str
    time: float
    score_type: str
    class_a: str
    class_b: str
    log_likelihood_class_a: float
    log_likelihood_class_b: float


@dataclass(frozen=True, slots=True)
class FeatureTableRow(FeatureValueMappingMixin):
    trajectory_id: str
    class_pair_id: str
    scenario_id: str
    scenario_family: str
    dataset_tier: str
    true_class: str
    feature_set_id: str
    feature_values: dict[str, float]

    def as_flat_dict(self) -> dict[str, object]:
        return {
            "trajectory_id": self.trajectory_id,
            "class_pair_id": self.class_pair_id,
            "scenario_id": self.scenario_id,
            "scenario_family": self.scenario_family,
            "dataset_tier": self.dataset_tier,
            "true_class": self.true_class,
            "feature_set_id": self.feature_set_id,
            **self.feature_values,
        }


@dataclass(frozen=True, slots=True)
class CovariateRow:
    class_pair_id: str
    dataset_tier: str
    scenario_family: str
    true_class: str
    num_trajectories: int
    mean_duration: float
    mean_sample_count: float
    mean_dt: float
    std_dt: float
    max_dt: float
    sampling_irregularity: float
    measurement_std: float
    outlier_fraction: float
    max_covariate_delta_name: str
    max_covariate_delta_ratio: float
    status: str


@dataclass(frozen=True, slots=True)
class FeatureExcitationRow:
    class_pair_id: str
    dataset_tier: str
    scenario_family: str
    feature_set_id: str
    num_rows: int
    feature_means: dict[str, float]
    feature_stds: dict[str, float]


@dataclass(frozen=True, slots=True)
class ClassPairDurationRow:
    classifier_id: str
    class_pair_id: str
    time: float
    num_prefixes: int
    prefix_accuracy: float
    mean_confidence: float
    posterior_margin: float


@dataclass(frozen=True, slots=True)
class ClassPairScenarioRow:
    classifier_id: str
    class_pair_id: str
    scenario_id: str
    scenario_family: str
    overall_accuracy: float
    mean_confidence: float
    num_predictions: int


@dataclass(frozen=True, slots=True)
class CommonExperimentArtifacts:
    run_dir: Path
    config_path: Path
    dataset_manifest_path: Path
    class_definitions_path: Path
    feature_manifest_path: Path
    feature_sets_path: Path
    class_pair_manifest_path: Path
    classifier_manifest_path: Path
    sensor_regimes_path: Path
    predictions_path: Path
    posterior_history_path: Path
    likelihood_history_path: Path
    method_evaluation_summary_path: Path
    feature_matrix_path: Path
    metrics_by_classifier_path: Path
    metrics_by_sensor_regime_path: Path
    metrics_by_classifier_and_feature_set_path: Path
    metrics_by_class_pair_path: Path
    prior_sensitivity_by_class_pair_path: Path
    feature_set_comparison_path: Path
    irregular_window_comparison_path: Path
    class_pair_duration_study_path: Path
    class_pair_scenario_study_path: Path
    covariate_leakage_audit_path: Path
    feature_excitation_matrix_path: Path
    identifiability_matrix_path: Path
    oracle_classifier_results_path: Path
    report_path: Path
    canonical_report_path: Path
    plots_dir: Path
