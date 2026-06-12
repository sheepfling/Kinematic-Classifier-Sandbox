from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from kinematic_classifier_sandbox.reports.markdown import MarkdownDocument
from kinematic_classifier_sandbox.utils.io import write_csv

from ..utils.math import (
    centroid as _centroid,
)
from ..utils.math import (
    cluster_balance as _cluster_balance,
)
from ..utils.math import (
    cluster_purity as _cluster_purity,
)
from ..utils.math import (
    euclidean_distance as _euclidean,
)
from ..utils.math import (
    kmeans as _kmeans,
)
from ..utils.math import (
    mean as _mean,
)
from ..utils.math import (
    project_rows as _project_rows,
)
from ..utils.math import (
    reconstruct_rows as _reconstruct_rows,
)
from ..utils.math import (
    silhouette_score as _silhouette_score,
)
from ..utils.plotting import _figure_to_png, plt
from .pca_analysis import analyze_feature_pca


def _class_labels(coordinates: tuple[dict[str, object], ...]) -> tuple[str, ...]:
    return tuple(str(row["true_class"]) for row in coordinates)


def _unique_labels(labels: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(sorted(set(labels)))


@dataclass(frozen=True, slots=True)
class PcaDimensionalityRow:
    component_count: int
    cumulative_variance_ratio: float
    reconstruction_rmse: float
    reconstruction_mae: float
    class_centroid_separation: float
    kmeans_cluster_count: int
    kmeans_inertia: float
    kmeans_silhouette: float
    kmeans_cluster_purity: float
    kmeans_cluster_balance: float


@dataclass(frozen=True, slots=True)
class PcaDimensionalityResult:
    feature_set_name: str
    feature_names: tuple[str, ...]
    class_names: tuple[str, ...]
    sample_count: int
    component_rows: tuple[PcaDimensionalityRow, ...]
    recommendation: dict[str, object]
    pca_component_count: int


@dataclass(frozen=True, slots=True)
class PcaDimensionalityArtifacts:
    run_dir: Path
    report_path: Path
    summary_path: Path
    component_sweep_path: Path
    clusterability_path: Path
    plot_variance_path: Path
    plot_clusterability_path: Path
    plot_separation_path: Path


def analyze_pca_dimensionality(
    *,
    seed: int = 7,
    trajectories_per_class: int = 5,
    feature_set: str | None = None,
    feature_names: tuple[str, ...] | list[str] | None = None,
    max_components: int = 6,
) -> PcaDimensionalityResult:
    pca_result = analyze_feature_pca(
        seed=seed,
        trajectories_per_class=trajectories_per_class,
        n_components=max_components,
        feature_set=feature_set,
        feature_names=feature_names,
    )
    selected_feature_names = pca_result.feature_names
    class_names = tuple(sorted({row["true_class"] for row in pca_result.coordinates}))
    raw_rows = [
        [float(getattr(row, feature_name)) for feature_name in selected_feature_names]
        for row in pca_result.feature_analysis.feature_rows
    ]
    truth = _class_labels(pca_result.coordinates)
    component_vectors = [
        [component.loadings[feature_name] for feature_name in selected_feature_names]
        for component in pca_result.components
    ]
    component_rows: list[PcaDimensionalityRow] = []
    for component_count in range(1, len(component_vectors) + 1):
        projected_rows = _project_rows(raw_rows, component_vectors, component_count)
        reconstructed_rows = _reconstruct_rows(projected_rows, component_vectors, component_count)
        squared_errors = [
            (row[index] - reconstructed[index]) ** 2
            for row, reconstructed in zip(raw_rows, reconstructed_rows)
            for index in range(len(selected_feature_names))
        ]
        absolute_errors = [
            abs(row[index] - reconstructed[index])
            for row, reconstructed in zip(raw_rows, reconstructed_rows)
            for index in range(len(selected_feature_names))
        ]
        class_centroids = {
            class_name: _centroid([row for row, label in zip(projected_rows, truth) if label == class_name])
            for class_name in class_names
        }
        centroid_distances = []
        for index, class_a in enumerate(class_names):
            for class_b in class_names[index + 1 :]:
                centroid_distances.append(_euclidean(class_centroids[class_a], class_centroids[class_b]))
        cluster_count = len(class_names)
        labels, _, inertia = _kmeans(projected_rows, cluster_count)
        component_rows.append(
            PcaDimensionalityRow(
                component_count=component_count,
                cumulative_variance_ratio=sum(component.explained_variance_ratio for component in pca_result.components[:component_count]),
                reconstruction_rmse=(sum(squared_errors) / max(len(squared_errors), 1)) ** 0.5,
                reconstruction_mae=_mean(absolute_errors),
                class_centroid_separation=_mean(centroid_distances),
                kmeans_cluster_count=cluster_count,
                kmeans_inertia=inertia,
                kmeans_silhouette=_silhouette_score(projected_rows, labels),
                kmeans_cluster_purity=_cluster_purity(labels, truth),
                kmeans_cluster_balance=_cluster_balance(labels),
            )
        )
    best_variance = next((row.component_count for row in component_rows if row.cumulative_variance_ratio >= 0.95), component_rows[-1].component_count if component_rows else 1)
    best_clusterability = max(component_rows, key=lambda row: (row.kmeans_silhouette, row.kmeans_cluster_purity, row.cumulative_variance_ratio)) if component_rows else None
    recommendation = {
        "recommended_components_for_95pct_variance": best_variance,
        "best_clusterability_component_count": best_clusterability.component_count if best_clusterability else 0,
        "best_clusterability_silhouette": best_clusterability.kmeans_silhouette if best_clusterability else 0.0,
        "best_clusterability_purity": best_clusterability.kmeans_cluster_purity if best_clusterability else 0.0,
        "feature_count": len(selected_feature_names),
        "class_count": len(class_names),
    }
    return PcaDimensionalityResult(
        feature_set_name=pca_result.feature_set_name,
        feature_names=selected_feature_names,
        class_names=class_names,
        sample_count=len(raw_rows),
        component_rows=tuple(component_rows),
        recommendation=recommendation,
        pca_component_count=len(pca_result.components),
    )


def _render_report(result: PcaDimensionalityResult) -> str:
    best_clusterability = (
        max(result.component_rows, key=lambda row: (row.kmeans_silhouette, row.kmeans_cluster_purity, row.cumulative_variance_ratio))
        if result.component_rows
        else None
    )
    report = MarkdownDocument("PCA Dimensionality Audit")
    report.paragraph(
        "This audit answers whether the current feature set is sufficiently low-dimensional for clustering and whether 2D is enough for class separation."
    )
    report.bullet_list(
        [
            f"Feature set: {result.feature_set_name}",
            f"Feature count: {len(result.feature_names)}",
            f"Class count: {len(result.class_names)}",
            f"Samples: {result.sample_count}",
        ]
    )

    report.heading("Recommendation", level=2)
    report.bullet_list(
        [
            f"95% variance needs at least `{result.recommendation['recommended_components_for_95pct_variance']}` principal components.",
            f"Best clusterability appears at `{result.recommendation['best_clusterability_component_count']}` principal components.",
            f"Best silhouette: `{float(result.recommendation['best_clusterability_silhouette']):.3f}`",
            f"Best cluster purity: `{float(result.recommendation['best_clusterability_purity']):.3f}`",
        ]
    )

    report.heading("Component Sweep", level=2)
    report.table(
        [
            "components",
            "cumulative_variance_ratio",
            "reconstruction_rmse",
            "reconstruction_mae",
            "centroid_separation",
            "kmeans_silhouette",
            "kmeans_cluster_purity",
        ],
        [
            (
                row.component_count,
                f"{row.cumulative_variance_ratio:.3f}",
                f"{row.reconstruction_rmse:.3f}",
                f"{row.reconstruction_mae:.3f}",
                f"{row.class_centroid_separation:.3f}",
                f"{row.kmeans_silhouette:.3f}",
                f"{row.kmeans_cluster_purity:.3f}",
            )
            for row in result.component_rows
        ],
    )

    report.heading("Clusterability", level=2)
    report.paragraph(
        "This section uses a deterministic k-means sweep on the PCA projections to ask whether the class structure is recoverable from low-dimensional embeddings."
    )

    report.heading("Interpretation", level=2)
    report.bullet_list(
        [
            f"`PC{best_clusterability.component_count}` currently gives the strongest combined silhouette and purity."
            if best_clusterability
            else "No PCA components were available for a clusterability recommendation.",
            "If silhouette stays low while cumulative variance is already high, the feature set is compressible but not cluster-separable.",
            "If purity improves materially as components increase, the 2D picture is hiding structure that is useful for clustering.",
            "Reconstruction error is reported in standardized feature units, so it is comparable across feature sets.",
        ]
    )
    return report.text()


def _render_variance_plot(result: PcaDimensionalityResult):
    fig, ax = plt.subplots(figsize=(8.0, 4.8))
    xs = [row.component_count for row in result.component_rows]
    variance = [row.cumulative_variance_ratio for row in result.component_rows]
    error = [row.reconstruction_rmse for row in result.component_rows]
    ax.plot(xs, variance, marker="o", color="#2563eb", label="cumulative variance ratio")
    ax.plot(xs, error, marker="s", color="#dc2626", label="reconstruction RMSE")
    ax.axhline(0.95, color="#16a34a", linestyle="--", linewidth=1.1, label="95% variance target")
    ax.set_xlabel("components")
    ax.set_ylabel("score")
    ax.set_title("Variance vs Reconstruction Error", loc="left", fontweight="bold")
    ax.grid(True, alpha=0.2)
    ax.legend(frameon=False)
    fig.tight_layout()
    return fig


def _render_clusterability_plot(result: PcaDimensionalityResult):
    fig, ax = plt.subplots(figsize=(8.0, 4.8))
    xs = [row.component_count for row in result.component_rows]
    silhouette = [row.kmeans_silhouette for row in result.component_rows]
    purity = [row.kmeans_cluster_purity for row in result.component_rows]
    ax.plot(xs, silhouette, marker="o", color="#7c3aed", label="silhouette")
    ax.plot(xs, purity, marker="s", color="#0f766e", label="cluster purity")
    ax.set_xlabel("components")
    ax.set_ylabel("score")
    ax.set_ylim(0.0, 1.05)
    ax.set_title("Clusterability on PCA Projection", loc="left", fontweight="bold")
    ax.grid(True, alpha=0.2)
    ax.legend(frameon=False)
    fig.tight_layout()
    return fig


def _render_separation_plot(result: PcaDimensionalityResult):
    fig, ax = plt.subplots(figsize=(8.0, 4.8))
    xs = [row.component_count for row in result.component_rows]
    centroid_separation = [row.class_centroid_separation for row in result.component_rows]
    balance = [row.kmeans_cluster_balance for row in result.component_rows]
    ax.plot(xs, centroid_separation, marker="o", color="#b45309", label="mean centroid separation")
    ax.plot(xs, balance, marker="s", color="#db2777", label="cluster balance")
    ax.set_xlabel("components")
    ax.set_ylabel("score")
    ax.set_title("Class Separation and Cluster Balance", loc="left", fontweight="bold")
    ax.grid(True, alpha=0.2)
    ax.legend(frameon=False)
    fig.tight_layout()
    return fig


def write_pca_dimensionality_audit_artifacts(
    output_dir: str | Path,
    *,
    seed: int = 7,
    trajectories_per_class: int = 5,
    feature_set: str | None = None,
    feature_names: tuple[str, ...] | list[str] | None = None,
    max_components: int = 6,
) -> PcaDimensionalityArtifacts:
    result = analyze_pca_dimensionality(
        seed=seed,
        trajectories_per_class=trajectories_per_class,
        feature_set=feature_set,
        feature_names=feature_names,
        max_components=max_components,
    )
    output_root = Path(output_dir)
    run_dir = output_root / (
        "pca_dimensionality_audit_v1"
        if result.feature_set_name == "all_engineered"
        else f"pca_dimensionality_audit_{result.feature_set_name}_v1"
    )
    run_dir.mkdir(parents=True, exist_ok=True)

    report_path = run_dir / "pca_dimensionality_report.md"
    summary_path = run_dir / "pca_dimensionality_summary.json"
    component_sweep_path = run_dir / "pca_component_sweep.csv"
    clusterability_path = run_dir / "pca_clusterability.csv"
    plot_variance_path = run_dir / "variance_vs_error.png"
    plot_clusterability_path = run_dir / "clusterability.png"
    plot_separation_path = run_dir / "separation.png"

    report_path.write_text(_render_report(result), encoding="utf-8")
    summary_path.write_text(
        json.dumps(
            {
                "feature_set_name": result.feature_set_name,
                "feature_names": list(result.feature_names),
                "class_names": list(result.class_names),
                "sample_count": result.sample_count,
                "recommendation": result.recommendation,
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    write_csv(
        component_sweep_path,
        [
            {
                "component_count": row.component_count,
                "cumulative_variance_ratio": row.cumulative_variance_ratio,
                "reconstruction_rmse": row.reconstruction_rmse,
                "reconstruction_mae": row.reconstruction_mae,
                "class_centroid_separation": row.class_centroid_separation,
                "kmeans_cluster_count": row.kmeans_cluster_count,
                "kmeans_inertia": row.kmeans_inertia,
                "kmeans_silhouette": row.kmeans_silhouette,
                "kmeans_cluster_purity": row.kmeans_cluster_purity,
                "kmeans_cluster_balance": row.kmeans_cluster_balance,
            }
            for row in result.component_rows
        ],
        [
            "component_count",
            "cumulative_variance_ratio",
            "reconstruction_rmse",
            "reconstruction_mae",
            "class_centroid_separation",
            "kmeans_cluster_count",
            "kmeans_inertia",
            "kmeans_silhouette",
            "kmeans_cluster_purity",
            "kmeans_cluster_balance",
        ],
    )
    write_csv(
        clusterability_path,
        [
            {
                "component_count": row.component_count,
                "kmeans_cluster_count": row.kmeans_cluster_count,
                "kmeans_inertia": row.kmeans_inertia,
                "kmeans_silhouette": row.kmeans_silhouette,
                "kmeans_cluster_purity": row.kmeans_cluster_purity,
                "kmeans_cluster_balance": row.kmeans_cluster_balance,
            }
            for row in result.component_rows
        ],
        [
            "component_count",
            "kmeans_cluster_count",
            "kmeans_inertia",
            "kmeans_silhouette",
            "kmeans_cluster_purity",
            "kmeans_cluster_balance",
        ],
    )
    plot_variance_path.write_bytes(_figure_to_png(_render_variance_plot(result)))
    plot_clusterability_path.write_bytes(_figure_to_png(_render_clusterability_plot(result)))
    plot_separation_path.write_bytes(_figure_to_png(_render_separation_plot(result)))

    return PcaDimensionalityArtifacts(
        run_dir=run_dir,
        report_path=report_path,
        summary_path=summary_path,
        component_sweep_path=component_sweep_path,
        clusterability_path=clusterability_path,
        plot_variance_path=plot_variance_path,
        plot_clusterability_path=plot_clusterability_path,
        plot_separation_path=plot_separation_path,
    )
