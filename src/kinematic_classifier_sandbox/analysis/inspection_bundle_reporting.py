from __future__ import annotations

from pathlib import Path

from ..corpus.adequacy_audit import CorpusAdequacyArtifacts
from ..corpus.coverage_report import CoverageReportArtifacts
from ..markdown_builder import MarkdownDocument
from ..utils.categorical import status_score
from ..utils.plotting import plt
from .feature_analysis import FeatureAnalysisArtifacts
from .pca_analysis import PcaAnalysisArtifacts


def render_feature_set_summary_chart(rows: list[dict[str, object]]):
    fig, axes = plt.subplots(2, 2, figsize=(13.2, 8.0))
    flat_axes = list(axes.flat)
    labels = [str(row["feature_set"]) for row in rows]
    avg_auc = [float(row["avg_pairwise_auc"]) for row in rows]
    min_auc = [float(row["min_pairwise_auc"]) for row in rows]
    avg_overlap = [float(row["avg_overlap"]) for row in rows]
    max_overlap = [float(row["max_overlap"]) for row in rows]
    adequacy = [status_score(str(row["feature_set_status"]), yellow=0.6, red=0.2) for row in rows]

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


def render_class_pair_summary_chart(rows: list[dict[str, object]]):
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


def render_abstract_inspection_index(
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
        ],
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
        ],
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
        language="bash",
    )
    return doc.text()


def _link(path: Path) -> str:
    return path.name
