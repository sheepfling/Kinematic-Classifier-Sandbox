from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class DimensionalLiftAuditResult:
    module_rows: tuple[dict[str, object], ...]
    scalar_assumption_rows: tuple[dict[str, object], ...]
    required_adapter_markdown: str
    audit_markdown: str
    vector_predictions_rows: tuple[dict[str, object], ...]
    vector_posterior_rows: tuple[dict[str, object], ...]
    vector_feature_rows: tuple[dict[str, object], ...]
    validation_results: dict[str, object]


@dataclass(frozen=True, slots=True)
class DimensionalLiftAuditArtifacts:
    run_dir: Path
    audit_report_path: Path
    module_status_path: Path
    scalar_assumption_inventory_path: Path
    required_adapters_path: Path
    vector_predictions_path: Path
    vector_posterior_history_path: Path
    vector_feature_matrix_path: Path
    validation_results_path: Path
