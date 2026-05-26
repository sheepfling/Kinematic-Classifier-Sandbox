from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path

from kinematic_classifier_sandbox.utils.io import write_csv

from ..corpus.adequacy_audit import CorpusAdequacyArtifacts, write_corpus_adequacy_artifacts
from ..corpus.coverage_report import CoverageReportArtifacts, write_coverage_report_artifacts
from ..markdown_builder import MarkdownDocument
from ..runtime_paths import prepare_matplotlib
from ..utils.plotting import _figure_to_png
from .feature_analysis import (
    FeatureAnalysisArtifacts,
    load_feature_set_manifest,
    write_feature_analysis_artifacts,
)
from .pca_analysis import PcaAnalysisArtifacts, write_pca_analysis_artifacts


@dataclass(frozen=True, slots=True)
class AbstractInspectionArtifacts:
    run_dir: Path
    index_path: Path
    manifest_path: Path
    machine_summary_path: Path
    summary_table_path: Path
    summary_chart_path: Path
    class_pair_summary_table_path: Path
    class_pair_summary_chart_path: Path
    feature_analysis_runs: tuple[FeatureAnalysisArtifacts, ...]
    pca_runs: tuple[PcaAnalysisArtifacts, ...]
    corpus_adequacy: CorpusAdequacyArtifacts
    coverage_report: CoverageReportArtifacts


def recommend_feature_set(summary_payload: dict[str, object]) -> dict[str, object]:
    candidates = list(summary_payload.get("feature_set_summary", []))
    if not candidates:
        raise ValueError("abstract inspection summary does not contain feature_set_summary")
    ranked = sorted(
        candidates,
        key=lambda row: (
            -_status_score(str(row.get("feature_set_status", ""))),
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


def _link(path: Path) -> str:
    return path.name


def _status_score(status: str) -> float:
    return {"green": 1.0, "yellow": 0.6, "red": 0.2}.get(status, 0.0)


def _summary_rows(
    *,
    feature_analysis_runs: tuple[FeatureAnalysisArtifacts, ...],
    coverage_report: CoverageReportArtifacts,
) -> list[dict[str, object]]:
    coverage_summary = json.loads(coverage_report.summary_path.read_text(encoding="utf-8"))
    feature_set_rows = {
        str(row["feature_set"]): row
        for row in _read_csv_rows(coverage_report.feature_set_summary_path)
    }
    rows: list[dict[str, object]] = []
    for artifacts in feature_analysis_runs:
        summary = json.loads((artifacts.run_dir / "feature_excitation_summary.json").read_text(encoding="utf-8"))
        feature_set_name = str(summary["feature_set_name"])
        identifiability_rows = _read_csv_rows(artifacts.identifiability_matrix_path)
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


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        return [dict(row) for row in reader]


def _render_summary_chart(rows: list[dict[str, object]]):
    plt = prepare_matplotlib()
    fig, axes = plt.subplots(2, 2, figsize=(13.2, 8.0))
    flat_axes = list(axes.flat)
    labels = [str(row["feature_set"]) for row in rows]
    avg_auc = [float(row["avg_pairwise_auc"]) for row in rows]
    min_auc = [float(row["min_pairwise_auc"]) for row in rows]
    avg_overlap = [float(row["avg_overlap"]) for row in rows]
    max_overlap = [float(row["max_overlap"]) for row in rows]
    adequacy = [_status_score(str(row["feature_set_status"])) for row in rows]

    flat_axes[0].bar(labels, avg_auc, color="#376996")
    flat_axes[0].set_title("Average Pairwise AUC")
    flat_axes[0].set_ylim(0.0, 1.0)

    flat_axes[1].bar(labels, min_auc, color="#254d73")
    flat_axes[1].set_title("Worst-Case Pairwise AUC")
    flat_axes[1].set_ylim(0.0, 1.0)

    flat_axes[2].bar(labels, avg_overlap, color="#c97a28")
    flat_axes[2].set_title("Average Overlap")
    flat_axes[2].set_ylim(0.0, max(avg_overlap + [1.0]))

    flat_axes[3].bar(labels, max_overlap, color="#a24b1d")
    flat_axes[3].set_title("Worst-Case Overlap")
    flat_axes[3].set_ylim(0.0, max(max_overlap + [1.0]))

    adequacy_ax = flat_axes[0].twinx()
    adequacy_ax.plot(labels, adequacy, color="#4a8f55", marker="o", linewidth=1.8)
    adequacy_ax.set_ylim(0.0, 1.05)
    adequacy_ax.set_yticks([0.2, 0.6, 1.0], labels=["red", "yellow", "green"])
    adequacy_ax.set_ylabel("Adequacy")

    for ax in flat_axes:
        ax.tick_params(axis="x", rotation=30)
        ax.grid(axis="y", alpha=0.25)
        ax.set_axisbelow(True)

    fig.suptitle("Feature-Set Inspection Summary")
    fig.tight_layout()
    return fig


def _class_pair_summary_rows(
    *,
    baseline_feature_analysis: FeatureAnalysisArtifacts,
    limit: int = 10,
) -> list[dict[str, object]]:
    rows = _read_csv_rows(baseline_feature_analysis.identifiability_matrix_path)
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


def _render_class_pair_chart(rows: list[dict[str, object]]):
    plt = prepare_matplotlib()
    fig, axes = plt.subplots(1, 2, figsize=(12.8, 5.4))
    labels = [str(row["class_pair"]) for row in rows]
    auc_values = [float(row["pairwise_auc"]) for row in rows]
    overlap_values = [float(row["overlap_estimate"]) for row in rows]

    axes[0].barh(labels, auc_values, color="#6b5fb5")
    axes[0].set_title("Hardest Class Pairs by AUC")
    axes[0].set_xlim(0.0, 1.0)
    axes[0].invert_yaxis()

    axes[1].barh(labels, overlap_values, color="#b55f5f")
    axes[1].set_title("Overlap Estimate")
    axes[1].set_xlim(0.0, max(overlap_values + [1.0]))
    axes[1].invert_yaxis()

    for ax in axes:
        ax.grid(axis="x", alpha=0.25)
        ax.set_axisbelow(True)

    fig.suptitle("Class-Pair Boundary Pressure")
    fig.tight_layout()
    return fig


def _render_index(
    *,
    feature_analysis_runs: tuple[FeatureAnalysisArtifacts, ...],
    pca_runs: tuple[PcaAnalysisArtifacts, ...],
    corpus_adequacy: CorpusAdequacyArtifacts,
    coverage_report: CoverageReportArtifacts,
    summary_rows: list[dict[str, object]],
    summary_table_path: Path,
    summary_chart_path: Path,
    class_pair_rows: list[dict[str, object]],
    class_pair_summary_table_path: Path,
    class_pair_summary_chart_path: Path,
) -> str:
    feature_run_lookup = {artifacts.run_dir.name: artifacts for artifacts in feature_analysis_runs}
    pca_run_lookup = {artifacts.run_dir.name: artifacts for artifacts in pca_runs}
    
    doc = MarkdownDocument("Abstract Inspection Bundle")
    doc.paragraph(
        "This bundle collects the abstract feature-space inspection artifacts that are intended to be rerun together."
    )
    
    doc.heading("Included Views", level=2)
    doc.bullet_list(
        [
            "Feature analysis by feature set",
            "PCA by feature set",
            "Corpus adequacy audit",
            "Coverage summary",
            "Cross-feature-set inspection summary",
        ]
    )

    doc.heading("Full-Set Baseline", level=2)
    baseline_feature = feature_run_lookup.get("feature_analysis_v1")
    baseline_pca = pca_run_lookup.get("pca_analysis_v1")
    baseline_links = []
    if baseline_feature is not None:
        baseline_links.extend(
            [
                f"Feature analysis report: `{_link(baseline_feature.report_path)}`",
                f"Feature ranking: `{_link(baseline_feature.plot_ranking_png_path)}`",
                f"Confusability heatmap: `{_link(baseline_feature.plot_confusability_png_path)}`",
            ]
        )
    if baseline_pca is not None:
        baseline_links.extend(
            [
                f"PCA report: `{_link(baseline_pca.report_path)}`",
                f"PCA scatter: `{_link(baseline_pca.plot_scatter_png_path)}`",
                f"PCA loadings: `{_link(baseline_pca.plot_loadings_png_path)}`",
            ]
        )
    if baseline_links:
        doc.bullet_list(baseline_links)

    doc.heading("Feature-Set Summary", level=2)
    doc.bullet_list(
        [
            f"Summary table: `{summary_table_path.name}`",
            f"Summary chart: `{summary_chart_path.name}`",
        ]
    )
    doc.table(
        ["Feature Set", "Avg Pairwise AUC", "Min Pairwise AUC", "Avg Overlap", "Max Overlap", "Adequacy", "Top Features"],
        [
            (
                f"`{row['feature_set']}`",
                f"`{float(row['avg_pairwise_auc']):.3f}`",
                f"`{float(row['min_pairwise_auc']):.3f}`",
                f"`{float(row['avg_overlap']):.3f}`",
                f"`{float(row['max_overlap']):.3f}`",
                f"`{row['feature_set_status']}`",
                f"`{row['top_features']}`",
            )
            for row in summary_rows
        ]
    )

    doc.heading("Hardest Class Boundaries", level=2)
    doc.bullet_list(
        [
            f"Boundary summary table: `{class_pair_summary_table_path.name}`",
            f"Boundary summary chart: `{class_pair_summary_chart_path.name}`",
        ]
    )
    doc.table(
        ["Class Pair", "Pairwise AUC", "Overlap", "Mahalanobis", "Accuracy"],
        [
            (
                f"`{row['class_pair']}`",
                f"`{float(row['pairwise_auc']):.3f}`",
                f"`{float(row['overlap_estimate']):.3f}`",
                f"`{float(row['mahalanobis_distance']):.3f}`",
                f"`{float(row['pairwise_classifier_accuracy']):.3f}`",
            )
            for row in class_pair_rows
        ]
    )

    doc.heading("Named Feature Sets", level=2)
    subset_names = sorted(
        {
            name.removeprefix("feature_analysis_").removesuffix("_v1")
            for name in feature_run_lookup
            if name != "feature_analysis_v1"
        }
    )
    named_sets_rows = []
    for feature_set_name in subset_names:
        feature_key = f"feature_analysis_{feature_set_name}_v1"
        pca_key = f"pca_analysis_{feature_set_name}_v1"
        feature_artifacts = feature_run_lookup[feature_key]
        pca_artifacts = pca_run_lookup[pca_key]
        named_sets_rows.append(
            (
                f"`{feature_set_name}`",
                f"`{feature_artifacts.report_path.relative_to(feature_artifacts.run_dir.parent)}`",
                f"`{pca_artifacts.report_path.relative_to(pca_artifacts.run_dir.parent)}`",
            )
        )
    doc.table(["Feature Set", "Feature Analysis", "PCA"], named_sets_rows)

    doc.heading("Corpus Stress And Coverage", level=2)
    doc.bullet_list(
        [
            f"Corpus adequacy report: `{corpus_adequacy.report_path.relative_to(corpus_adequacy.run_dir.parent)}`",
            f"Class-pair coverage heatmap: `{corpus_adequacy.pair_status_heatmap_path.relative_to(corpus_adequacy.run_dir.parent)}`",
            f"Coverage report: `{coverage_report.report_path.relative_to(coverage_report.run_dir.parent)}`",
        ]
    )

    doc.heading("Rerun Contract", level=2)
    doc.paragraph("Regenerate this bundle through the normal artifact exporter:")
    doc.fence(
        "PYTHONPYCACHEPREFIX=/Users/rick/LocalStorage/GIT_LOCAL/active/CACHE/kinematic-classifier-sandbox/.pycache python3 scripts/export_artifacts.py",
        language="bash"
    )
    
    return doc.text()


def write_abstract_inspection_artifacts(
    output_dir: str | Path,
    *,
    seed: int = 7,
    trajectories_per_class: int = 5,
    n_components: int = 3,
    feature_sets: tuple[str, ...] | list[str] | None = None,
) -> AbstractInspectionArtifacts:
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
    summary_chart_path.write_bytes(_figure_to_png(_render_summary_chart(summary_rows)))
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
    class_pair_summary_chart_path.write_bytes(_figure_to_png(_render_class_pair_chart(class_pair_rows)))
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
        _render_index(
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
