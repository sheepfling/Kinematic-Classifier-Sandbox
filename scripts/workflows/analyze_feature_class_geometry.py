#!/usr/bin/env python3
from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

from new_study_workflow_common import (
    copy_file,
    ensure_declaration_artifacts,
    filter_common_rows,
    load_study_config,
    phase_dir,
    requested_feature_names,
    write_common_experiment_artifacts,
    write_feature_analysis_artifacts,
    write_feature_redundancy_matrix,
    write_text,
)


def run_phase(study_path: str | Path, output_dir: str | Path) -> Path:
    study = load_study_config(study_path)
    ensure_declaration_artifacts(study_path, output_dir)
    output_path = phase_dir(output_dir, study, "01_feature_class_analysis")

    with tempfile.TemporaryDirectory() as temp_dir:
        feature_artifacts = write_feature_analysis_artifacts(
            temp_dir,
            seed=int(study.get("seed", 7)),
            trajectories_per_class=int(study.get("trajectories_per_class", 5)),
        )
        common_artifacts = write_common_experiment_artifacts(
            temp_dir,
            config_path=study.get("common_experiment_config"),
            seed=int(study.get("seed", 7)),
            trajectories_per_case=int(study.get("trajectories_per_case", 8)),
        )

        copy_file(feature_artifacts.pairwise_auc_matrix_path, output_path / "pairwise_auc.csv")
        copy_file(feature_artifacts.pairwise_overlap_matrix_path, output_path / "pairwise_overlap.csv")
        copy_file(feature_artifacts.feature_separation_scores_path, output_path / "feature_importance.csv")
        copy_file(feature_artifacts.plot_confusability_png_path, output_path / "class_confusability_graph.png")
        copy_file(common_artifacts.oracle_classifier_results_path, output_path / "oracle_separability.csv")
        copy_file(common_artifacts.prior_sensitivity_by_class_pair_path, output_path / "prior_fragility_preview.csv")
        write_feature_redundancy_matrix(feature_artifacts.feature_matrix_path, study, output_path / "feature_redundancy_matrix.csv")

    feature_names = requested_feature_names(study)
    hierarchy_lines = [
        "# Class Hierarchy Proposal",
        "",
        f"- Study pair(s): `{', '.join(study.get('class_pairs', []))}`",
        f"- Requested feature sets: `{', '.join(study.get('feature_sets', []))}`",
        f"- Requested features resolved from the manifest: `{', '.join(feature_names)}`",
        "- Current recommendation: keep the declared pair flat and compare rung behavior before introducing a hierarchy.",
        "- Promotion to a hierarchical split should only happen if overlap stays high after corpus and prior review.",
    ]
    write_text(output_path / "class_hierarchy_proposal.md", "\n".join(hierarchy_lines) + "\n")
    report_lines = [
        "# Feature/Class Analysis Report",
        "",
        f"- Study: `{study['study_id']}`",
        f"- Class pairs: `{', '.join(study.get('class_pairs', []))}`",
        f"- Feature sets: `{', '.join(study.get('feature_sets', []))}`",
        "- This phase packages static separability, overlap, redundancy, prior-fragility preview, and oracle preview artifacts.",
    ]
    write_text(output_path / "feature_class_analysis_report.md", "\n".join(report_lines) + "\n")
    return output_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the feature/class geometry workflow phase.")
    parser.add_argument("--study", required=True, help="Path to the study YAML.")
    parser.add_argument("--output-dir", default="artifacts", help="Workflow artifact root.")
    args = parser.parse_args()
    path = run_phase(args.study, args.output_dir)
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

