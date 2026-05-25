#!/usr/bin/env python3
from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

from new_study_workflow_common import (
    copy_file,
    derive_corpus_decision_gate,
    ensure_declaration_artifacts,
    load_study_config,
    phase_dir,
    write_corpus_adequacy_artifacts,
    write_selected_generated_corpus_artifacts,
)


def run_phase(study_path: str | Path, output_dir: str | Path) -> Path:
    study = load_study_config(study_path)
    ensure_declaration_artifacts(study_path, output_dir)
    output_path = phase_dir(output_dir, study, "03_corpus_audit")

    with tempfile.TemporaryDirectory() as temp_dir:
        selected_corpus_artifacts = write_selected_generated_corpus_artifacts(temp_dir)
        adequacy_artifacts = write_corpus_adequacy_artifacts(
            temp_dir,
            seed=int(study.get("seed", 7)),
            trajectories_per_class=int(study.get("trajectories_per_class", 5)),
        )

        copy_file(selected_corpus_artifacts.class_validity_scores_path, output_path / "class_validity_scores.csv")
        copy_file(selected_corpus_artifacts.validity_plot_path, output_path / "label_status_distribution.png")
        copy_file(adequacy_artifacts.covariate_leakage_path, output_path / "leakage_audit.csv")
        copy_file(adequacy_artifacts.report_path, output_path / "corpus_adequacy_report.md")
        copy_file(adequacy_artifacts.summary_path, output_path / "corpus_adequacy_summary.json")
        derive_corpus_decision_gate(
            adequacy_artifacts.summary_path,
            selected_corpus_artifacts.class_validity_scores_path,
            output_path / "corpus_decision_gate.json",
        )

    return output_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the selected-corpus audit workflow phase.")
    parser.add_argument("--study", required=True, help="Path to the study YAML.")
    parser.add_argument("--output-dir", default="artifacts", help="Workflow artifact root.")
    args = parser.parse_args()
    path = run_phase(args.study, args.output_dir)
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

