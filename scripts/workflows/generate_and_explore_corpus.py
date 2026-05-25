#!/usr/bin/env python3
from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

from new_study_workflow_common import (
    copy_file,
    ensure_declaration_artifacts,
    load_study_config,
    phase_dir,
    write_candidate_generation_artifacts,
    write_corpus_autodevelopment_artifacts,
    write_corpus_policy_tuning_artifacts,
    write_generated_corpus_feature_artifacts,
    write_generic_corpus_exploration_weight_sweep_artifacts,
    write_selected_generated_corpus_artifacts,
    write_text,
)


def run_phase(study_path: str | Path, output_dir: str | Path) -> Path:
    study = load_study_config(study_path)
    ensure_declaration_artifacts(study_path, output_dir)
    output_path = phase_dir(output_dir, study, "02_corpus_generation")

    with tempfile.TemporaryDirectory() as temp_dir:
        candidate_artifacts = write_candidate_generation_artifacts(temp_dir)
        autodevelopment_artifacts = write_corpus_autodevelopment_artifacts(temp_dir)
        feature_artifacts = write_generated_corpus_feature_artifacts(temp_dir)
        selected_corpus_artifacts = write_selected_generated_corpus_artifacts(temp_dir)
        sweep_artifacts = write_generic_corpus_exploration_weight_sweep_artifacts(
            temp_dir,
            seed=int(study.get("seed", 7)),
        )
        policy_artifacts = write_corpus_policy_tuning_artifacts(
            temp_dir,
            seed=int(study.get("seed", 7)) + 4,
        )

        copy_file(candidate_artifacts.generated_candidates_path, output_path / "generated_candidates.csv")
        copy_file(candidate_artifacts.candidate_coverage_png_path, output_path / "generated_candidate_coverage.png")
        copy_file(autodevelopment_artifacts.candidate_scores_path, output_path / "candidate_scores.csv")
        copy_file(autodevelopment_artifacts.selected_manifest_path, output_path / "selected_corpus_policy_manifest.json")
        copy_file(autodevelopment_artifacts.corpus_score_pareto_path, output_path / "corpus_score_pareto.png")
        copy_file(selected_corpus_artifacts.manifest_path, output_path / "selected_corpus_manifest.json")
        copy_file(feature_artifacts.excitation_scores_path, output_path / "feature_excitation_matrix.csv")
        copy_file(sweep_artifacts.rows_path, output_path / "corpus_policy_summary.csv")
        copy_file(policy_artifacts.recommended_policy_path, output_path / "recommended_corpus_policy.yaml")

    report_lines = [
        "# Corpus Generation Report",
        "",
        f"- Study: `{study['study_id']}`",
        f"- Objective difficulty: `{study.get('corpus_objective', {}).get('difficulty', 'unknown')}`",
        "- This phase packages generated candidate rows, corpus candidate scores, selected-corpus manifests, feature excitation, and policy/explorer summaries.",
    ]
    write_text(output_path / "corpus_generation_report.md", "\n".join(report_lines) + "\n")
    return output_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the corpus generation and exploration workflow phase.")
    parser.add_argument("--study", required=True, help="Path to the study YAML.")
    parser.add_argument("--output-dir", default="artifacts", help="Workflow artifact root.")
    args = parser.parse_args()
    path = run_phase(args.study, args.output_dir)
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

