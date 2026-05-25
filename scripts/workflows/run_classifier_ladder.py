#!/usr/bin/env python3
from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

from new_study_workflow_common import (
    copy_file,
    derive_confusion_rows,
    derive_method_metrics,
    ensure_declaration_artifacts,
    filter_common_rows,
    load_study_config,
    phase_dir,
    read_csv,
    requested_classifiers,
    rung_id_for_classifier,
    write_csv,
    write_common_experiment_artifacts,
    write_rung_sufficiency_artifacts,
    write_text,
)


def run_phase(study_path: str | Path, output_dir: str | Path) -> Path:
    study = load_study_config(study_path)
    ensure_declaration_artifacts(study_path, output_dir)
    output_path = phase_dir(output_dir, study, "04_ladder_evaluation")

    with tempfile.TemporaryDirectory() as temp_dir:
        common_artifacts = write_common_experiment_artifacts(
            temp_dir,
            config_path=study.get("common_experiment_config"),
            seed=int(study.get("seed", 7)),
            trajectories_per_case=int(study.get("trajectories_per_case", 8)),
        )
        rung_artifacts = write_rung_sufficiency_artifacts(
            temp_dir,
            seed=int(study.get("seed", 7)),
            trajectories_per_case=int(study.get("trajectories_per_case", 8)),
        )

        posterior_rows = filter_common_rows(read_csv(common_artifacts.posterior_history_path), study)
        write_csv(output_path / "posterior_history_by_method.csv", posterior_rows)
        derive_method_metrics(common_artifacts.predictions_path, study, output_path / "method_metrics.csv")
        derive_confusion_rows(common_artifacts.predictions_path, study, output_path / "confusion_by_method.csv")
        prior_rows = filter_common_rows(read_csv(common_artifacts.prior_sensitivity_by_class_pair_path), study, include_feature_sets=False)
        write_csv(output_path / "prior_sensitivity_by_method.csv", prior_rows)

        allowed_rungs = {rung_id_for_classifier(value) for value in requested_classifiers(study)}
        promotion_rows = [
            row
            for row in read_csv(rung_artifacts.promotion_matrix_path)
            if str(row.get("current_rung_id", "")) in allowed_rungs or str(row.get("candidate_next_rung_id", "")) in allowed_rungs
        ]
        failure_rows = [
            row
            for row in read_csv(rung_artifacts.failure_mode_path)
            if str(row.get("current_rung_id", "")) in allowed_rungs or str(row.get("candidate_next_rung_id", "")) in allowed_rungs
        ]
        write_csv(output_path / "sufficiency_matrix.csv", promotion_rows)
        write_csv(output_path / "insufficiency_matrix.csv", failure_rows)
        copy_file(rung_artifacts.promotion_decision_plot_path, output_path / "promotion_decision_matrix.png")
        copy_file(rung_artifacts.posterior_quality_plot_path, output_path / "posterior_quality_by_rung.png")

    report_lines = [
        "# Ladder Evaluation Report",
        "",
        f"- Study: `{study['study_id']}`",
        f"- Requested classifiers: `{', '.join(study.get('classifiers', []))}`",
        "- This phase filters the common experiment and rung-sufficiency outputs down to the declared study pair, feature sets, and method set.",
    ]
    write_text(output_path / "ladder_evaluation_report.md", "\n".join(report_lines) + "\n")
    return output_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the classifier ladder workflow phase.")
    parser.add_argument("--study", required=True, help="Path to the study YAML.")
    parser.add_argument("--output-dir", default="artifacts", help="Workflow artifact root.")
    args = parser.parse_args()
    path = run_phase(args.study, args.output_dir)
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

