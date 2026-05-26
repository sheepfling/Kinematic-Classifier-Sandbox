from __future__ import annotations

import json
from pathlib import Path

from kinematic_classifier_sandbox.utils.io import write_csv

from ..corpus.adequacy_audit import write_corpus_adequacy_artifacts
from ..corpus.coverage_report import write_coverage_report_artifacts
from ..utils.plotting import _figure_to_png
from .feature_analysis import load_feature_set_manifest, write_feature_analysis_artifacts
from .inspection_bundle_contracts import AbstractInspectionArtifacts
from .inspection_bundle_reporting import (
    render_abstract_inspection_index,
    render_class_pair_summary_chart,
    render_feature_set_summary_chart,
)
from .pca_analysis import write_pca_analysis_artifacts


def write_abstract_inspection_artifacts(
    output_dir: str | Path,
    *,
    seed: int = 7,
    trajectories_per_class: int = 5,
    n_components: int = 3,
    feature_sets: tuple[str, ...] | list[str] | None = None,
) -> AbstractInspectionArtifacts:
    from .inspection_bundle import _class_pair_summary_rows, _summary_rows

    output_root = Path(output_dir)
    run_dir = output_root / "abstract_inspection_v1"
    run_dir.mkdir(parents=True, exist_ok=True)

    manifest = load_feature_set_manifest()
    selected_feature_sets = tuple(feature_sets or tuple(manifest))

    feature_analysis_runs = [
        write_feature_analysis_artifacts(
            output_root,
            seed=seed,
            trajectories_per_class=trajectories_per_class,
            feature_set=feature_set,
        )
        for feature_set in selected_feature_sets
    ]
    pca_runs = [
        write_pca_analysis_artifacts(
            output_root,
            seed=seed,
            trajectories_per_class=trajectories_per_class,
            n_components=n_components,
            feature_set=feature_set,
        )
        for feature_set in selected_feature_sets
    ]
    corpus_adequacy = write_corpus_adequacy_artifacts(
        output_root,
        seed=seed,
        trajectories_per_class=trajectories_per_class,
    )
    coverage_report = write_coverage_report_artifacts(
        output_root,
        seed=seed,
        trajectories_per_class=trajectories_per_class,
    )

    index_path = run_dir / "abstract_inspection_index.md"
    manifest_path = run_dir / "abstract_inspection_manifest.json"
    machine_summary_path = run_dir / "abstract_inspection_summary.json"
    summary_table_path = run_dir / "feature_set_inspection_summary.csv"
    summary_chart_path = run_dir / "feature_set_inspection_summary.png"
    class_pair_summary_table_path = run_dir / "hardest_class_pairs.csv"
    class_pair_summary_chart_path = run_dir / "hardest_class_pairs.png"
    summary_rows = _summary_rows(
        feature_analysis_runs=tuple(feature_analysis_runs),
        coverage_report=coverage_report,
    )
    baseline_feature_analysis = next(
        artifacts for artifacts in feature_analysis_runs if artifacts.run_dir.name == "feature_analysis_v1"
    )
    class_pair_rows = _class_pair_summary_rows(
        baseline_feature_analysis=baseline_feature_analysis,
        limit=10,
    )
    write_csv(
        summary_table_path,
        summary_rows,
        [
            "feature_set",
            "feature_count",
            "avg_pairwise_auc",
            "min_pairwise_auc",
            "avg_overlap",
            "max_overlap",
            "top_features",
            "feature_set_status",
            "avg_moderate_or_strong_fraction",
            "corpus_overall_status",
        ],
    )
    summary_chart_path.write_bytes(_figure_to_png(render_feature_set_summary_chart(summary_rows)))
    write_csv(
        class_pair_summary_table_path,
        class_pair_rows,
        [
            "class_pair",
            "class_a",
            "class_b",
            "pairwise_auc",
            "overlap_estimate",
            "mahalanobis_distance",
            "pairwise_classifier_accuracy",
        ],
    )
    class_pair_summary_chart_path.write_bytes(_figure_to_png(render_class_pair_summary_chart(class_pair_rows)))
    machine_summary_path.write_text(
        json.dumps(
            {
                "feature_sets": list(selected_feature_sets),
                "feature_set_summary": summary_rows,
                "hardest_class_pairs": class_pair_rows,
                "corpus_adequacy": json.loads(corpus_adequacy.summary_path.read_text(encoding="utf-8")),
                "coverage_report": json.loads(coverage_report.summary_path.read_text(encoding="utf-8")),
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    index_path.write_text(
        render_abstract_inspection_index(
            feature_analysis_runs=tuple(feature_analysis_runs),
            pca_runs=tuple(pca_runs),
            corpus_adequacy=corpus_adequacy,
            coverage_report=coverage_report,
            summary_rows=summary_rows,
            summary_table_path=summary_table_path,
            summary_chart_path=summary_chart_path,
            class_pair_rows=class_pair_rows,
            class_pair_summary_table_path=class_pair_summary_table_path,
            class_pair_summary_chart_path=class_pair_summary_chart_path,
        ),
        encoding="utf-8",
    )
    manifest_path.write_text(
        json.dumps(
            {
                "feature_sets": list(selected_feature_sets),
                "feature_analysis_runs": [str(artifacts.run_dir.name) for artifacts in feature_analysis_runs],
                "pca_runs": [str(artifacts.run_dir.name) for artifacts in pca_runs],
                "corpus_adequacy_run": corpus_adequacy.run_dir.name,
                "coverage_report_run": coverage_report.run_dir.name,
                "machine_summary": machine_summary_path.name,
                "summary_table": summary_table_path.name,
                "summary_chart": summary_chart_path.name,
                "class_pair_summary_table": class_pair_summary_table_path.name,
                "class_pair_summary_chart": class_pair_summary_chart_path.name,
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    return AbstractInspectionArtifacts(
        run_dir=run_dir,
        index_path=index_path,
        manifest_path=manifest_path,
        machine_summary_path=machine_summary_path,
        summary_table_path=summary_table_path,
        summary_chart_path=summary_chart_path,
        class_pair_summary_table_path=class_pair_summary_table_path,
        class_pair_summary_chart_path=class_pair_summary_chart_path,
        feature_analysis_runs=tuple(feature_analysis_runs),
        pca_runs=tuple(pca_runs),
        corpus_adequacy=corpus_adequacy,
        coverage_report=coverage_report,
    )
