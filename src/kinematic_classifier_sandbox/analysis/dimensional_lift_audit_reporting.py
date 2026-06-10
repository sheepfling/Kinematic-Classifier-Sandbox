from __future__ import annotations

from ..markdown_builder import MarkdownDocument
from .dimensional_lift_audit_contracts import DimensionalLiftAuditResult


def render_dimensional_lift_audit_report(result: DimensionalLiftAuditResult) -> str:
    doc = MarkdownDocument("Dimensional Lift Audit")
    doc.paragraph(
        "This artifact audits which current modules are dimension-agnostic, adapter-compatible, or rewrite-required, and proves that a fake vector-valued corpus can still reach the standard methodology artifact surface."
    )

    doc.heading("Validation Summary", level=2)
    doc.bullet_list(
        [
            f"Overall status: `{result.validation_results['overall_status']}`",
            f"All modules labeled: `{result.validation_results['all_modules_labeled']}`",
            f"Scalar assumptions listed: `{result.validation_results['scalar_assumptions_listed']}`",
            f"Vector corpus loaded: `{result.validation_results['vector_corpus_loaded']}`",
            f"Vector features emitted: `{result.validation_results['vector_features_emitted']}`",
            f"Vector predictions emitted: `{result.validation_results['vector_predictions_emitted']}`",
            f"Vector posterior rows emitted: `{result.validation_results['vector_posteriors_emitted']}`",
        ]
    )

    doc.heading("3D-Ready Decision Boundary", level=2)
    doc.bullet_list(
        [
            f"Dimension-agnostic modules: `{result.dimensional_summary['dimension_agnostic_modules']}`",
            f"Adapter-compatible modules: `{result.dimensional_summary['adapter_compatible_modules']}`",
            f"Rewrite-required modules: `{result.dimensional_summary['rewrite_required_modules']}`",
            f"Blocking scalar assumptions: `{result.dimensional_summary['blocking_scalar_assumptions']}`",
            "3D-ready remains a future claim until vector corpus, vector feature, and vector contract proofs all exist together.",
        ]
    )

    doc.heading("Module Status", level=2)
    doc.table(
        ["module", "layer", "dimensional_status", "required_3d_action"],
        [
            (row["module"], row["layer"], row["dimensional_status"], row["required_3d_action"])
            for row in result.module_rows
        ],
    )

    doc.heading("Scalar Assumption Inventory", level=2)
    doc.table(
        ["module", "assumption_id", "severity", "blocking_for_3d"],
        [
            (row["module"], row["assumption_id"], row["severity"], str(row["blocking_for_3d"]))
            for row in result.scalar_assumption_rows
        ],
    )

    doc.heading("Fake Vector Proof", level=2)
    doc.bullet_list(
        [
            "The fake vector corpus uses `measurement_dim=3`, `measurement_axes=(x,y,z)`, and `coordinate_frame=enu`.",
            "It emits a standard feature table, prediction table, and posterior table without relying on full 3D dynamics or a full 3D Kalman bank.",
        ]
    )

    doc.heading("Required Adapters", level=2)
    doc.paragraph(result.required_adapter_markdown)

    doc.heading("Evidence Required Before Calling The Methodology 3D-Ready", level=2)
    doc.bullet_list([str(item) for item in result.dimensional_summary["three_d_ready_evidence_required"]])
    return doc.text()
