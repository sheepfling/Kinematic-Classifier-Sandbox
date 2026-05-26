from __future__ import annotations

from pydantic import BaseModel, Field


class ExperimentSectionModel(BaseModel):
    name: str
    study_adapter_id: str
    seed: int = 123
    output_dir: str


class DatasetSectionModel(BaseModel):
    generator: str
    tiers: list[str] = Field(default_factory=list)
    classes: list[str] = Field(default_factory=list)
    class_pairs: list[tuple[str, str]] = Field(default_factory=list)
    trajectories_per_class: int = 0


class ManifestPathSectionModel(BaseModel):
    manifest_path: str


class PriorSectionModel(BaseModel):
    id: str
    mode: str
    strength: float | None = None


class EvaluationsSectionModel(BaseModel):
    confidence_thresholds: list[float] = Field(default_factory=list)
    duration_sweeps: bool = False
    noise_sweeps: bool = False
    prior_sweeps: bool = False
    identifiability: bool = False
    pca: bool = False
    oracle_separability: bool = False
    leakage_audit: bool = False
    feature_excitation: bool = False


class OutputsSectionModel(BaseModel):
    predictions_path: str = "unified_predictions.csv"
    posterior_history_path: str = "unified_posterior_history.csv"
    likelihood_history_path: str = "unified_likelihood_history.csv"
    feature_matrix_path: str = "unified_feature_matrix.csv"
    metrics_by_classifier_path: str = "metrics_by_classifier.csv"
    metrics_by_sensor_regime_path: str = "metrics_by_sensor_regime.csv"
    metrics_by_classifier_and_feature_set_path: str = "metrics_by_classifier_and_feature_set.csv"
    metrics_by_class_pair_path: str = "metrics_by_class_pair.csv"
    prior_sensitivity_by_class_pair_path: str = "prior_sensitivity_by_class_pair.csv"
    feature_set_comparison_path: str = "feature_set_comparison.csv"
    irregular_window_comparison_path: str = "irregular_window_comparison.csv"
    class_pair_duration_study_path: str = "class_pair_duration_study.csv"
    class_pair_scenario_study_path: str = "class_pair_scenario_study.csv"
    covariate_leakage_audit_path: str = "covariate_leakage_audit.csv"
    feature_excitation_matrix_path: str = "feature_excitation_matrix.csv"
    identifiability_matrix_path: str = "identifiability_matrix.csv"
    oracle_classifier_results_path: str = "oracle_classifier_results.csv"
    report_path: str = "common_experiment_report.md"


class CommonExperimentConfigModel(BaseModel):
    experiment: ExperimentSectionModel
    dataset: DatasetSectionModel
    feature_sets: ManifestPathSectionModel
    class_pairs: ManifestPathSectionModel
    classifiers: ManifestPathSectionModel
    priors: list[PriorSectionModel] = Field(default_factory=list)
    evaluations: EvaluationsSectionModel = Field(default_factory=EvaluationsSectionModel)
    outputs: OutputsSectionModel = Field(default_factory=OutputsSectionModel)
