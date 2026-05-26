from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
ARTIFACTS_ROOT = ROOT / "artifacts"
SHOWCASE_DOCS_DIR = ROOT / "docs" / "showcase"


@dataclass(frozen=True, slots=True)
class ShowcaseArtifacts:
    showcase_dir: Path
    index_path: Path
    proof_gallery_path: Path
    artifact_manifest_path: Path
    summary_metrics_path: Path
    reports_dir: Path
    plots_dir: Path
    tables_dir: Path
    run_cards_dir: Path
    team_packet_dir: Path
    zip_path: Path | None
    validation_path: Path


@dataclass(frozen=True, slots=True)
class ShowcaseValidationResult:
    overall_status: str
    required_reports_exist: bool
    proof_gallery_complete: bool
    manifest_complete: bool
    metrics_tables_exist: bool
    gallery_references_exist: bool
    proof_gallery_references_exist: bool
    gallery_annotations_complete: bool
    feature_taxonomy_complete: bool
    class_pair_identifiability_complete: bool
    advanced_filter_go_no_go_present: bool
    dimensional_status_present: bool
    errors: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class ShowcaseManifestEntry:
    kind: str
    relative_path: str
    section: str | None = None
    source_path: str | None = None
    title: str | None = None
    plot_id: str | None = None
    caption: str | None = None
    interpretation: str | None = None
    limitations: str | None = None

    def to_dict(self) -> dict[str, object]:
        return {key: value for key, value in asdict(self).items() if value is not None}


@dataclass(frozen=True, slots=True)
class ShowcasePlotDefinition:
    plot_id: str
    source: str
    filename: str
    section: str
    caption: str
    interpretation: str
    limitations: str


@dataclass(frozen=True, slots=True)
class ShowcaseDerivedPlotArtifact:
    plot_id: str
    section: str
    relative_path: str
    source_path: str
    caption: str
    interpretation: str
    limitations: str


@dataclass(frozen=True, slots=True)
class ShowcaseTableSpec:
    filename: str
    source: Path
    section: str


@dataclass(frozen=True, slots=True)
class ShowcaseSourceDocSpec:
    source: Path
    relative_destination: str


@dataclass(frozen=True, slots=True)
class RunCardBody:
    heading: str
    bullets: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class RunCardSpec:
    filename: str
    title: str
    body: RunCardBody


@dataclass(frozen=True, slots=True)
class ShowcaseTopResult:
    identifier: str
    overall_accuracy: float


@dataclass(frozen=True, slots=True)
class ShowcaseCorpusAdequacySummary:
    overall_status: str
    feature_status: str
    class_pair_status: str
    covariate_status: str

    def to_dict(self) -> dict[str, str]:
        return {
            "overall_status": self.overall_status,
            "feature_status": self.feature_status,
            "class_pair_status": self.class_pair_status,
            "covariate_status": self.covariate_status,
        }


@dataclass(frozen=True, slots=True)
class ShowcaseAdvancedFilterSummary:
    imm_justified: bool
    particle_filter_justified: bool
    rbpf_justified: bool
    method_rows: tuple[dict[str, object], ...]
    primary_artifact: str


@dataclass(frozen=True, slots=True)
class ShowcaseAlgorithmReportData:
    metrics_by_classifier: tuple[dict[str, str], ...]
    common_dataset_rows: tuple[dict[str, str], ...]


@dataclass(frozen=True, slots=True)
class ShowcaseFeatureReportData:
    taxonomy_rows: tuple[dict[str, object], ...]
    identifiability_rows: tuple[dict[str, str], ...]
    oracle_rows: tuple[dict[str, str], ...]


@dataclass(frozen=True, slots=True)
class ShowcaseFilteringReportData:
    advanced_summary: ShowcaseAdvancedFilterSummary
    transition_rows: tuple[dict[str, str], ...]


@dataclass(frozen=True, slots=True)
class ShowcaseCorpusReportData:
    summary: ShowcaseCorpusAdequacySummary
    class_pair_rows: tuple[dict[str, str], ...]
    leakage_rows: tuple[dict[str, str], ...]


@dataclass(frozen=True, slots=True)
class ShowcaseDimensionalLiftReportData:
    dimension_rows: tuple[dict[str, str], ...]


@dataclass(frozen=True, slots=True)
class ShowcaseOpenRisksData:
    corpus_summary: ShowcaseCorpusAdequacySummary
    recommendations: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class ShowcaseHeadlineSummary:
    best_common_study_classifier: ShowcaseTopResult
    best_common_dataset_method: ShowcaseTopResult
    corpus_adequacy: ShowcaseCorpusAdequacySummary
    advanced_filters: ShowcaseAdvancedFilterSummary
    dimensional_status_counts: dict[str, int]

    def to_dict(self) -> dict[str, object]:
        return {
            "best_common_study_classifier": {
                "classifier_id": self.best_common_study_classifier.identifier,
                "overall_accuracy": self.best_common_study_classifier.overall_accuracy,
            },
            "best_common_dataset_method": {
                "method_name": self.best_common_dataset_method.identifier,
                "overall_accuracy": self.best_common_dataset_method.overall_accuracy,
            },
            "corpus_adequacy": self.corpus_adequacy.to_dict(),
            "advanced_filters": {
                "imm_justified": self.advanced_filters.imm_justified,
                "particle_filter_justified": self.advanced_filters.particle_filter_justified,
                "rbpf_justified": self.advanced_filters.rbpf_justified,
                "method_rows": list(self.advanced_filters.method_rows),
                "primary_artifact": self.advanced_filters.primary_artifact,
            },
            "dimensional_status_counts": dict(self.dimensional_status_counts),
        }
