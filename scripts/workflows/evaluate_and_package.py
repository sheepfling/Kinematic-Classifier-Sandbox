#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from analyze_feature_class_geometry import run_phase as run_feature_phase
from audit_selected_corpus import run_phase as run_audit_phase
from generate_and_explore_corpus import run_phase as run_generation_phase
from new_study_workflow_common import (
    derive_decision_card,
    ensure_declaration_artifacts,
    load_study_config,
    workflow_root,
    write_text,
    write_visual_gallery,
)
from run_classifier_ladder import run_phase as run_ladder_phase
from score_study_confidence import run_phase as run_confidence_phase


def run_workflow(study_path: str | Path, output_dir: str | Path) -> Path:
    study = load_study_config(study_path)
    ensure_declaration_artifacts(study_path, output_dir)
    run_feature_phase(study_path, output_dir)
    run_generation_phase(study_path, output_dir)
    run_audit_phase(study_path, output_dir)
    run_ladder_phase(study_path, output_dir)
    run_confidence_phase(study_path, output_dir)

    root = workflow_root(output_dir, study)
    report_dir = root / "05_report"
    report_dir.mkdir(parents=True, exist_ok=True)

    derive_decision_card(
        root / "03_corpus_audit" / "corpus_decision_gate.json",
        root / "04_ladder_evaluation" / "sufficiency_matrix.csv",
        root / "04b_confidence" / "study_confidence_summary.json",
        report_dir / "decision_card.md",
        study,
    )
    write_visual_gallery(output_dir, study)

    study_report_lines = [
        f"# Study Report: {study['title']}",
        "",
        f"- Study ID: `{study['study_id']}`",
        f"- Hypothesis: {study['hypothesis']}",
        f"- Class pairs: `{', '.join(study.get('class_pairs', []))}`",
        f"- Feature sets: `{', '.join(study.get('feature_sets', []))}`",
        f"- Classifiers: `{', '.join(study.get('classifiers', []))}`",
        "",
        "## Phase Outputs",
        "",
        f"- [Study Declaration]({root / '00_study_declaration'})",
        f"- [Feature/Class Analysis]({root / '01_feature_class_analysis'})",
        f"- [Corpus Generation]({root / '02_corpus_generation'})",
        f"- [Corpus Audit]({root / '03_corpus_audit'})",
        f"- [Ladder Evaluation]({root / '04_ladder_evaluation'})",
        f"- [Confidence]({root / '04b_confidence'})",
        f"- [Decision Card]({report_dir / 'decision_card.md'})",
        f"- [Visual Gallery]({report_dir / 'visual_gallery.md'})",
    ]
    write_text(report_dir / "study_report.md", "\n".join(str(line) for line in study_report_lines) + "\n")

    index_lines = [
        f"# Workflow Index: {study['study_id']}",
        "",
        f"- [Study Report]({report_dir / 'study_report.md'})",
        f"- [Decision Card]({report_dir / 'decision_card.md'})",
        f"- [Visual Gallery]({report_dir / 'visual_gallery.md'})",
    ]
    write_text(root / "index.md", "\n".join(str(line) for line in index_lines) + "\n")
    return root


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the full new-study workflow and package the result.")
    parser.add_argument("--study", required=True, help="Path to the study YAML.")
    parser.add_argument("--output-dir", default="artifacts", help="Workflow artifact root.")
    args = parser.parse_args()
    root = run_workflow(args.study, args.output_dir)
    print(root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
