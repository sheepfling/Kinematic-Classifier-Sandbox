from __future__ import annotations

import re
from pathlib import Path

from kinematic_classifier_sandbox.utils.io import _read_csv, _read_json

from ..story.repo_story import CLAIMS as REPO_STORY_CLAIMS
from .contracts import ROOT, ShowcaseValidationResult


def extract_markdown_relative_targets(markdown: str) -> tuple[str, ...]:
    matches = re.findall(r"!\[[^\]]*\]\(([^)]+)\)|\[[^\]]+\]\(([^)]+)\)", markdown)
    targets: list[str] = []
    for image_target, link_target in matches:
        target = image_target or link_target
        if target and "://" not in target and not target.startswith("#"):
            targets.append(target)
    return tuple(targets)


def required_report_names() -> tuple[str, ...]:
    return (
        "00_executive_summary.md",
        "01_problem_framing.md",
        "02_methodology_overview.md",
        "03_algorithm_ladder.md",
        "04_feature_taxonomy.md",
        "05_filtering_taxonomy.md",
        "06_study_suite.md",
        "07_visualization_gallery.md",
        "08_results_summary.md",
        "09_3d_transition_plan.md",
        "10_open_risks_and_next_steps.md",
    )


def validate_showcase_artifacts(showcase_dir: str | Path) -> ShowcaseValidationResult:
    root = Path(showcase_dir)
    errors: list[str] = []
    reports_dir = root / "reports"
    plots_dir = root / "plots"
    tables_dir = root / "tables"
    run_cards_dir = root / "run_cards"
    proof_gallery_path = root / "proof_gallery.md"
    manifest_path = root / "artifact_manifest.json"

    required_reports_exist = True
    for report_name in required_report_names():
        path = reports_dir / report_name
        if not path.exists() or not path.read_text(encoding="utf-8").strip():
            required_reports_exist = False
            errors.append(f"missing or empty report: {report_name}")

    proof_gallery_complete = True
    proof_gallery_references_exist = True
    required_claim_headings = tuple(
        f"## Claim {index}: {claim.claim}" for index, claim in enumerate(REPO_STORY_CLAIMS, start=1)
    )
    if not proof_gallery_path.exists() or not proof_gallery_path.read_text(encoding="utf-8").strip():
        proof_gallery_complete = False
        proof_gallery_references_exist = False
        errors.append("proof_gallery.md is missing or empty")
        proof_gallery_text = ""
    else:
        proof_gallery_text = proof_gallery_path.read_text(encoding="utf-8")
        for heading in required_claim_headings:
            if heading not in proof_gallery_text:
                proof_gallery_complete = False
                errors.append(f"proof gallery missing claim section: {heading}")
        for target in extract_markdown_relative_targets(proof_gallery_text):
            if not (root / target).exists():
                proof_gallery_references_exist = False
                errors.append(f"proof gallery missing referenced artifact: {target}")

    manifest_complete = manifest_path.exists()
    manifest_items = _read_json(manifest_path)["items"] if manifest_complete else []
    if not manifest_complete:
        errors.append("artifact_manifest.json is missing")

    metrics_tables_exist = True
    for filename in (
        "metrics_by_classifier.csv",
        "metrics_by_class_pair.csv",
        "feature_set_comparison.csv",
        "identifiability_matrix.csv",
    ):
        path = tables_dir / filename
        if not path.exists() or len(path.read_text(encoding="utf-8").splitlines()) <= 1:
            metrics_tables_exist = False
            errors.append(f"missing or empty metrics table: {filename}")

    plot_entries = [entry for entry in manifest_items if entry.get("kind") == "plot"]
    gallery_references_exist = True
    gallery_annotations_complete = True
    for entry in plot_entries:
        relative_path = root / str(entry["relative_path"])
        if not relative_path.exists():
            gallery_references_exist = False
            errors.append(f"missing gallery file: {entry['relative_path']}")
        if not entry.get("caption") or not entry.get("interpretation"):
            gallery_annotations_complete = False
            errors.append(f"incomplete gallery annotation: {entry['relative_path']}")

    feature_taxonomy_complete = True
    taxonomy_path = tables_dir / "feature_taxonomy.json"
    if taxonomy_path.exists():
        taxonomy_rows = _read_json(taxonomy_path)
        required_feature_keys = {
            "name",
            "role",
            "history_behavior",
            "geometry_assumption",
            "dimensional_transfer",
            "dependency_tags",
            "sensitivity_tags",
        }
        for row in taxonomy_rows:
            missing = required_feature_keys.difference(row)
            if missing:
                feature_taxonomy_complete = False
                errors.append(
                    f"feature taxonomy missing keys for {row.get('name', 'unknown')}: {sorted(missing)}"
                )
                break
    else:
        feature_taxonomy_complete = False
        errors.append("feature_taxonomy.json is missing")

    class_pair_identifiability_complete = True
    identifiability_path = tables_dir / "identifiability_matrix.csv"
    if identifiability_path.exists():
        rows = _read_csv(identifiability_path)
        pair_ids = {row["class_pair_id"] for row in rows}
        manifest_pairs = _read_json(
            ROOT / "experiments" / "common_1d_classifier_study" / "class_pair_manifest.json"
        )["class_pairs"]
        expected = {"_vs_".join(row["pair"]) for row in manifest_pairs}
        if not expected.issubset(pair_ids):
            class_pair_identifiability_complete = False
            errors.append("not every declared class pair has an identifiability row in the packet")
    else:
        class_pair_identifiability_complete = False
        errors.append("identifiability_matrix.csv is missing from packet tables")

    advanced_filter_go_no_go_present = True
    filtering_report = reports_dir / "05_filtering_taxonomy.md"
    if filtering_report.exists():
        text = filtering_report.read_text(encoding="utf-8")
        if "IMM justified now" not in text or "Particle filter justified now" not in text:
            advanced_filter_go_no_go_present = False
            errors.append("advanced-method go/no-go status missing from filtering report")
    else:
        advanced_filter_go_no_go_present = False
        errors.append("filtering report is missing")

    dimensional_status_present = True
    transition_report = reports_dir / "09_3d_transition_plan.md"
    if transition_report.exists():
        text = transition_report.read_text(encoding="utf-8")
        if (
            "dimension_agnostic" not in text
            or "adapter_compatible" not in text
            or "rewrite_required" not in text
        ):
            dimensional_status_present = False
            errors.append("3D transition report does not name dimensional status categories")
    else:
        dimensional_status_present = False
        errors.append("3D transition report is missing")

    if not plots_dir.exists():
        gallery_references_exist = False
        errors.append("plots directory is missing")
    if not tables_dir.exists():
        metrics_tables_exist = False
        errors.append("tables directory is missing")
    if not run_cards_dir.exists():
        errors.append("run_cards directory is missing")

    overall_status = "pass" if not errors else "fail"
    return ShowcaseValidationResult(
        overall_status=overall_status,
        required_reports_exist=required_reports_exist,
        proof_gallery_complete=proof_gallery_complete,
        manifest_complete=manifest_complete,
        metrics_tables_exist=metrics_tables_exist,
        gallery_references_exist=gallery_references_exist,
        proof_gallery_references_exist=proof_gallery_references_exist,
        gallery_annotations_complete=gallery_annotations_complete,
        feature_taxonomy_complete=feature_taxonomy_complete,
        class_pair_identifiability_complete=class_pair_identifiability_complete,
        advanced_filter_go_no_go_present=advanced_filter_go_no_go_present,
        dimensional_status_present=dimensional_status_present,
        errors=tuple(errors),
    )
