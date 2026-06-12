from __future__ import annotations

import json

from ..corpus.coverage_contracts import CoverageReportArtifacts
from ..utils.categorical import status_score
from ..utils.io import read_csv_rows
from .feature_analysis import FeatureAnalysisArtifacts
from .inspection_bundle_artifact_io import write_abstract_inspection_artifacts
from .inspection_bundle_contracts import AbstractInspectionArtifacts


def recommend_feature_set(summary_payload: dict[str, object]) -> dict[str, object]:
    candidates = list(summary_payload.get("feature_set_summary", []))
    if not candidates:
        raise ValueError("abstract inspection summary does not contain feature_set_summary")
    ranked = sorted(
        candidates,
        key=lambda row: (
            -status_score(str(row.get("feature_set_status", "")), yellow=0.6, red=0.2),
            -float(row.get("min_pairwise_auc", 0.0)),
            -float(row.get("avg_pairwise_auc", 0.0)),
            float(row.get("max_overlap", 1.0)),
            float(row.get("avg_overlap", 0.0)),
            -float(row.get("avg_moderate_or_strong_fraction", 0.0)),
        ),
    )
    return dict(ranked[0])


def recommend_hardest_class_pair(summary_payload: dict[str, object]) -> dict[str, object]:
    candidates = list(summary_payload.get("hardest_class_pairs", []))
    if not candidates:
        raise ValueError("abstract inspection summary does not contain hardest_class_pairs")
    ranked = sorted(
        candidates,
        key=lambda row: (
            float(row.get("pairwise_auc", 1.0)),
            -float(row.get("overlap_estimate", 0.0)),
            float(row.get("mahalanobis_distance", 0.0)),
        ),
    )
    return dict(ranked[0])

def _summary_rows(
    *,
    feature_analysis_runs: tuple[FeatureAnalysisArtifacts, ...],
    coverage_report: CoverageReportArtifacts,
) -> list[dict[str, object]]:
    coverage_summary = json.loads(coverage_report.summary_path.read_text(encoding="utf-8"))
    feature_set_rows = {
        str(row["feature_set"]): row
        for row in read_csv_rows(coverage_report.feature_set_summary_path)
    }
    rows: list[dict[str, object]] = []
    for artifacts in feature_analysis_runs:
        summary = json.loads((artifacts.run_dir / "feature_excitation_summary.json").read_text(encoding="utf-8"))
        feature_set_name = str(summary["feature_set_name"])
        identifiability_rows = read_csv_rows(artifacts.identifiability_matrix_path)
        filtered_auc_values = [float(row["pairwise_auc"]) for row in identifiability_rows]
        avg_pairwise_auc = sum(filtered_auc_values) / max(len(filtered_auc_values), 1)
        min_pairwise_auc = min(filtered_auc_values) if filtered_auc_values else 0.0
        filtered_overlap_values = [float(row["overlap_estimate"]) for row in identifiability_rows]
        avg_overlap = sum(filtered_overlap_values) / max(len(filtered_overlap_values), 1)
        max_overlap = max(filtered_overlap_values) if filtered_overlap_values else 0.0

        feature_set_summary = feature_set_rows[feature_set_name]
        rows.append(
            {
                "feature_set": feature_set_name,
                "feature_count": len(summary["feature_names"]),
                "avg_pairwise_auc": avg_pairwise_auc,
                "min_pairwise_auc": min_pairwise_auc,
                "avg_overlap": avg_overlap,
                "max_overlap": max_overlap,
                "top_features": " ".join(summary["top_features"][:3]),
                "feature_set_status": str(feature_set_summary["status"]),
                "avg_moderate_or_strong_fraction": float(feature_set_summary["avg_moderate_or_strong_fraction"]),
                "corpus_overall_status": str(coverage_summary["corpus_adequacy_summary"]["overall_status"]),
            }
        )
    return sorted(rows, key=lambda row: str(row["feature_set"]))

def _class_pair_summary_rows(
    *,
    baseline_feature_analysis: FeatureAnalysisArtifacts,
    limit: int = 10,
) -> list[dict[str, object]]:
    rows = read_csv_rows(baseline_feature_analysis.identifiability_matrix_path)
    scored = [
        {
            "class_pair": f"{row['class_a']} vs {row['class_b']}",
            "class_a": row["class_a"],
            "class_b": row["class_b"],
            "pairwise_auc": float(row["pairwise_auc"]),
            "overlap_estimate": float(row["overlap_estimate"]),
            "mahalanobis_distance": float(row["mahalanobis_distance"]),
            "pairwise_classifier_accuracy": float(row["pairwise_classifier_accuracy"]),
        }
        for row in rows
    ]
    scored.sort(key=lambda row: (row["pairwise_auc"], -row["overlap_estimate"], row["mahalanobis_distance"]))
    return scored[:limit]


__all__ = [
    "AbstractInspectionArtifacts",
    "recommend_feature_set",
    "recommend_hardest_class_pair",
    "write_abstract_inspection_artifacts",
]
