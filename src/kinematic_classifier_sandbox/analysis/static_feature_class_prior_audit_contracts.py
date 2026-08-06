from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class StaticAuditFeatureSchemaEntry:
    name: str
    provenance_tags: tuple[str, ...] = ()
    online_available: bool = True
    label_rule_overlap: bool = False
    semantic_group: str = ""
    units: str = ""
    aggregation: str = ""
    dimension: str = ""
    measurement_resolution: float | None = None
    threshold_operator: str = ""
    threshold_value: float | None = None


@dataclass(frozen=True, slots=True)
class StaticAuditSample:
    true_class: str
    feature_values: dict[str, float]
    sample_id: str = ""


@dataclass(frozen=True, slots=True)
class StaticAuditClassFeatureExpectation:
    class_name: str
    feature_name: str
    expected_mean: float | None = None
    expected_std: float | None = None
    source: str = ""


@dataclass(frozen=True, slots=True)
class StaticFeatureClassPriorAuditResult:
    study_name: str
    feature_names: tuple[str, ...]
    class_names: tuple[str, ...]
    priors: dict[str, float]
    class_pair_rows: tuple[dict[str, object], ...]
    class_feature_signature_rows: tuple[dict[str, object], ...]
    class_observability_rows: tuple[dict[str, object], ...]
    feature_relevance_rows: tuple[dict[str, object], ...]
    feature_redundancy_rows: tuple[dict[str, object], ...]
    feature_alias_rows: tuple[dict[str, object], ...]
    feature_synergy_rows: tuple[dict[str, object], ...]
    prior_pathology_rows: tuple[dict[str, object], ...]
    coverage_rows: tuple[dict[str, object], ...]
    leakage_rows: tuple[dict[str, object], ...]
    decision_card_rows: tuple[dict[str, object], ...]
    static_decision: dict[str, object]
    declared_dimension: str = ""


@dataclass(frozen=True, slots=True)
class StaticFeatureClassPriorAuditArtifacts:
    run_dir: Path
    report_path: Path
    decision_card_path: Path
    decision_card_png_path: Path
    class_confusability_png_path: Path
    feature_relevance_png_path: Path
    feature_redundancy_png_path: Path
    feature_synergy_png_path: Path
    prior_pathology_surface_png_path: Path
    prior_flip_thresholds_png_path: Path
    coverage_feasibility_png_path: Path
    leakage_provenance_png_path: Path
    action_router_png_path: Path
    class_confusability_matrix_path: Path
    class_pair_diagnostics_path: Path
    class_feature_signature_path: Path
    class_observability_path: Path
    feature_relevance_table_path: Path
    feature_redundancy_matrix_path: Path
    feature_alias_candidates_path: Path
    feature_synergy_candidates_path: Path
    prior_regime_path: Path
    prior_pathology_report_path: Path
    prior_flip_thresholds_path: Path
    coverage_static_report_path: Path
    coverage_feasibility_path: Path
    leakage_static_report_path: Path
    leakage_provenance_audit_path: Path
