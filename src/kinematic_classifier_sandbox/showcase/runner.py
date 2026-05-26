from __future__ import annotations

import shutil
import subprocess
import zipfile
from dataclasses import asdict
from pathlib import Path

from kinematic_classifier_sandbox.markdown_builder import MarkdownDocument
from kinematic_classifier_sandbox.utils.io import _write_json, _write_text

from ..story.repo_story import (
    render_story_index as render_repo_story_index,
)
from ..story.repo_story import (
    render_team_packet_index as render_repo_story_team_packet_index,
)
from .assets import (
    _algorithm_report_data,
    _build_run_cards,
    _copy_showcase_plots,
    _copy_showcase_sources,
    _copy_showcase_tables,
    _corpus_report_data,
    _dimensional_lift_report_data,
    _feature_report_data,
    _filtering_report_data,
    _generate_showcase_derived_plots,
    _headline_summary,
    _open_risks_data,
    _render_proof_gallery,
)
from .contracts import ARTIFACTS_ROOT, ROOT, ShowcaseArtifacts
from .reporting import (
    _render_3d_transition_report,
    _render_algorithm_ladder_report,
    _render_executive_report,
    _render_feature_report,
    _render_filtering_report,
    _render_gallery_report,
    _render_methodology_report,
    _render_open_risks_report,
    _render_problem_framing_report,
    _render_results_summary_report,
    _render_study_suite_report,
)
from .validation import required_report_names, validate_showcase_artifacts


def build_showcase_artifacts(
    output_dir: str | Path = ARTIFACTS_ROOT,
    *,
    refresh: bool = False,
    create_zip: bool = False,
) -> ShowcaseArtifacts:
    if refresh:
        subprocess.run(["python3", "scripts/export_artifacts.py"], cwd=ROOT, check=True)

    output_root = Path(output_dir)
    showcase_dir = output_root / "showcase"
    reports_dir = showcase_dir / "reports"
    plots_dir = showcase_dir / "plots"
    tables_dir = showcase_dir / "tables"
    run_cards_dir = showcase_dir / "run_cards"
    proof_gallery_path = showcase_dir / "proof_gallery.md"
    team_packet_dir = output_root / "team_packet"
    zip_path = output_root / "kinematic_classifier_team_packet.zip" if create_zip else None
    validation_path = showcase_dir / "validation_results.json"

    if showcase_dir.exists():
        shutil.rmtree(showcase_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)
    run_cards_dir.mkdir(parents=True, exist_ok=True)

    summary = _headline_summary()
    algorithm_data = _algorithm_report_data()
    feature_data = _feature_report_data()
    filtering_data = _filtering_report_data()
    _corpus_report_data()
    dimensional_lift_data = _dimensional_lift_report_data()
    open_risks_data = _open_risks_data()

    report_payloads = {
        "00_executive_summary.md": _render_executive_report(summary),
        "01_problem_framing.md": _render_problem_framing_report(),
        "02_methodology_overview.md": _render_methodology_report(),
        "03_algorithm_ladder.md": _render_algorithm_ladder_report(algorithm_data),
        "04_feature_taxonomy.md": _render_feature_report(feature_data),
        "05_filtering_taxonomy.md": _render_filtering_report(filtering_data),
        "06_study_suite.md": _render_study_suite_report(),
        "07_visualization_gallery.md": "",
        "08_results_summary.md": _render_results_summary_report(summary),
        "09_3d_transition_plan.md": _render_3d_transition_report(dimensional_lift_data),
        "10_open_risks_and_next_steps.md": _render_open_risks_report(open_risks_data),
    }

    manifest_entries: list[dict[str, object]] = []
    manifest_entries.extend(_copy_showcase_sources(reports_dir))
    plot_entries = _copy_showcase_plots(plots_dir)
    plot_entries.extend(_generate_showcase_derived_plots(plots_dir))
    manifest_entries.extend(plot_entries)
    manifest_entries.extend(_copy_showcase_tables(tables_dir))
    manifest_entries.extend(_build_run_cards(run_cards_dir))

    report_payloads["07_visualization_gallery.md"] = _render_gallery_report(plot_entries)

    for filename, body in report_payloads.items():
        path = reports_dir / filename
        _write_text(path, body)
        manifest_entries.append(
            {
                "kind": "report",
                "relative_path": str(path.relative_to(showcase_dir)),
                "title": filename.removesuffix(".md"),
            }
        )

    summary_metrics_path = showcase_dir / "summary_metrics.json"
    _write_json(summary_metrics_path, summary.to_dict())
    manifest_entries.append(
        {
            "kind": "summary_metrics",
            "relative_path": str(summary_metrics_path.relative_to(showcase_dir)),
        }
    )

    index_report = MarkdownDocument("Team-Facing Methodology Showcase")
    index_report.paragraph(
        "This index is the team-facing packet entrypoint for the kinematic-classifier methodology stack."
    )
    index_report.heading("Claim-Oriented Entry Point", level=2)
    index_report.bullet_list(["[proof_gallery.md](proof_gallery.md)"])
    index_report.heading("Reports", level=2)
    index_report.bullet_list(
        [f"[reports/{report_name}](reports/{report_name})" for report_name in required_report_names()]
    )
    index_report.heading("Run Cards", level=2)
    index_report.bullet_list(
        [f"[run_cards/{card_path.name}](run_cards/{card_path.name})" for card_path in sorted(run_cards_dir.glob("*.md"))]
    )
    index_report.heading("Tables", level=2)
    index_report.bullet_list(
        [f"[tables/{table_path.name}](tables/{table_path.name})" for table_path in sorted(tables_dir.iterdir()) if table_path.is_file()]
    )
    index_report.heading("Plots", level=2)
    index_report.bullet_list(
        [f"[plots/{plot_path.name}](plots/{plot_path.name})" for plot_path in sorted(plots_dir.iterdir()) if plot_path.is_file()]
    )
    index_report.heading("Rerun Flow", level=2)
    index_report.bullet_list(
        [
            "Refresh source artifacts: `python3 scripts/export_artifacts.py`",
            "Rebuild showcase: `python3 scripts/build_showcase.py`",
            "Export team packet: `python3 scripts/export_team_packet.py --zip`",
        ]
    )
    index_path = showcase_dir / "index.md"
    _write_text(index_path, index_report.text())
    _write_text(proof_gallery_path, _render_proof_gallery())
    story_index_path = showcase_dir / "story_index.md"
    _write_text(story_index_path, render_repo_story_index())

    artifact_manifest_path = showcase_dir / "artifact_manifest.json"
    manifest_entries.append({"kind": "index", "relative_path": str(index_path.relative_to(showcase_dir))})
    manifest_entries.append(
        {"kind": "proof_gallery", "relative_path": str(proof_gallery_path.relative_to(showcase_dir))}
    )
    manifest_entries.append(
        {"kind": "story_index", "relative_path": str(story_index_path.relative_to(showcase_dir))}
    )

    _write_json(artifact_manifest_path, {"items": manifest_entries})
    validation = validate_showcase_artifacts(showcase_dir)
    _write_json(validation_path, asdict(validation))
    manifest_entries.append(
        {"kind": "validation", "relative_path": str(validation_path.relative_to(showcase_dir))}
    )
    _write_json(artifact_manifest_path, {"items": manifest_entries})

    if team_packet_dir.exists():
        shutil.rmtree(team_packet_dir)
    shutil.copytree(showcase_dir, team_packet_dir)
    _write_text(team_packet_dir / "index.md", render_repo_story_team_packet_index())

    if create_zip and zip_path is not None:
        if zip_path.exists():
            zip_path.unlink()
        with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for path in sorted(team_packet_dir.rglob("*")):
                if path.is_file():
                    archive.write(path, arcname=str(Path("team_packet") / path.relative_to(team_packet_dir)))

    return ShowcaseArtifacts(
        showcase_dir=showcase_dir,
        index_path=index_path,
        proof_gallery_path=proof_gallery_path,
        artifact_manifest_path=artifact_manifest_path,
        summary_metrics_path=summary_metrics_path,
        reports_dir=reports_dir,
        plots_dir=plots_dir,
        tables_dir=tables_dir,
        run_cards_dir=run_cards_dir,
        team_packet_dir=team_packet_dir,
        zip_path=zip_path,
        validation_path=validation_path,
    )
