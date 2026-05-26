from __future__ import annotations

from typing import TypedDict


class MetricsByClassifierRow(TypedDict):
    classifier_id: str
    overall_accuracy: float
    num_predictions: int


class MetricsBySensorRegimeRow(TypedDict):
    sensor_regime_id: str
    same_sensor_fairness_bucket: str
    overall_accuracy: float
    mean_confidence: float
    num_predictions: int
    num_classifiers: int
    measurement_dims: str
    coordinate_frames: str


class CovariateAuditRow(TypedDict):
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


class FeatureExcitationSummaryRow(TypedDict):
    class_pair_id: str
    dataset_tier: str
    scenario_family: str
    feature_set_id: str
    num_rows: int
    feature_means: dict[str, float]
    feature_stds: dict[str, float]


class ClassPairDurationSummaryRow(TypedDict):
    classifier_id: str
    class_pair_id: str
    time: float
    num_prefixes: int
    prefix_accuracy: float
    mean_confidence: float
    posterior_margin: float


class ClassPairScenarioSummaryRow(TypedDict):
    classifier_id: str
    class_pair_id: str
    scenario_id: str
    scenario_family: str
    overall_accuracy: float
    mean_confidence: float
    num_predictions: int


class IdentifiabilitySummaryRow(TypedDict):
    class_pair_id: str
    feature_set_id: str
    history_behavior: str
    class_a: str
    class_b: str
    num_examples: int
    num_features: int
    mean_absolute_feature_distance: float
    mean_standardized_feature_distance: float
    overlap_estimate: float
    confusability_score: float
    identifiability_status: str


class OracleSummaryRow(TypedDict):
    class_pair_id: str
    feature_set_id: str
    oracle_accuracy: float
    mean_confidence: float
    mean_posterior_margin: float
    num_examples: int
    history_behavior: str
    best_feature_set_for_pair: str
    best_oracle_accuracy_for_pair: float
    is_best_feature_set: bool
