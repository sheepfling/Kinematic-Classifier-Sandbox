from __future__ import annotations

import json
from pathlib import Path

from kinematic_classifier_sandbox.utils.io import write_csv

from .dimensional_lift_audit_contracts import (
    DimensionalLiftAuditArtifacts,
    DimensionalLiftAuditResult,
)


def write_dimensional_lift_audit_artifacts(
    output_dir: str | Path,
    *,
    result: DimensionalLiftAuditResult | None = None,
) -> DimensionalLiftAuditArtifacts:
    from .dimensional_lift_audit import analyze_dimensional_lift_audit

    audit = result or analyze_dimensional_lift_audit()
    run_dir = Path(output_dir) / "dimensional_lift_audit"
    run_dir.mkdir(parents=True, exist_ok=True)

    audit_report_path = run_dir / "dimensional_lift_audit.md"
    module_status_path = run_dir / "module_dimension_status.csv"
    scalar_assumption_inventory_path = run_dir / "scalar_assumption_inventory.csv"
    dimensional_summary_path = run_dir / "dimensional_lift_summary.json"
    required_adapters_path = run_dir / "required_3d_adapters.md"
    vector_predictions_path = run_dir / "vector_proof_predictions.csv"
    vector_posterior_history_path = run_dir / "vector_proof_posterior_history.csv"
    vector_feature_matrix_path = run_dir / "vector_proof_feature_matrix.csv"
    validation_results_path = run_dir / "validation_results.json"

    audit_report_path.write_text(audit.audit_markdown, encoding="utf-8")
    dimensional_summary_path.write_text(json.dumps(audit.dimensional_summary, indent=2), encoding="utf-8")
    required_adapters_path.write_text(audit.required_adapter_markdown, encoding="utf-8")
    validation_results_path.write_text(json.dumps(audit.validation_results, indent=2), encoding="utf-8")
    write_csv(
        module_status_path,
        list(audit.module_rows),
        ["module", "layer", "dimensional_status", "reason", "required_3d_action"],
    )
    write_csv(
        scalar_assumption_inventory_path,
        list(audit.scalar_assumption_rows),
        ["module", "assumption_id", "severity", "blocking_for_3d", "current_assumption", "3d_requirement"],
    )
    write_csv(
        vector_predictions_path,
        list(audit.vector_predictions_rows),
        [
            "run_id",
            "classifier_id",
            "sensor_regime_id",
            "trajectory_id",
            "scenario_id",
            "time",
            "true_class",
            "predicted_class",
            "confidence",
            "posterior_slow_linear",
            "posterior_fast_linear",
            "measurement_dim",
            "coordinate_frame",
        ],
    )
    write_csv(
        vector_posterior_history_path,
        list(audit.vector_posterior_rows),
        [
            "run_id",
            "classifier_id",
            "sensor_regime_id",
            "trajectory_id",
            "scenario_id",
            "time",
            "true_class",
            "posterior_slow_linear",
            "posterior_fast_linear",
            "measurement_dim",
            "coordinate_frame",
        ],
    )
    write_csv(
        vector_feature_matrix_path,
        list(audit.vector_feature_rows),
        [
            "trajectory_id",
            "scenario_id",
            "true_class",
            "measurement_dim",
            "coordinate_frame",
            "duration",
            "path_length",
            "displacement_norm",
            "mean_dt",
        ],
    )

    return DimensionalLiftAuditArtifacts(
        run_dir=run_dir,
        audit_report_path=audit_report_path,
        module_status_path=module_status_path,
        scalar_assumption_inventory_path=scalar_assumption_inventory_path,
        dimensional_summary_path=dimensional_summary_path,
        required_adapters_path=required_adapters_path,
        vector_predictions_path=vector_predictions_path,
        vector_posterior_history_path=vector_posterior_history_path,
        vector_feature_matrix_path=vector_feature_matrix_path,
        validation_results_path=validation_results_path,
    )
