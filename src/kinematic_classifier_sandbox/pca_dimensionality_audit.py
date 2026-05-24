from __future__ import annotations

from dataclasses import dataclass
import csv
import io
import json
import os
from pathlib import Path

from .pca_analysis import analyze_feature_pca


def _prepare_matplotlib():
    os.environ.setdefault("MPLCONFIGDIR", str(Path("/private/tmp/kinematic-classifier-sandbox-mpl")))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def _write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _mean(values: list[float]) -> float:
    return sum(values) / max(len(values), 1)


def _std(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    mean_value = _mean(values)
    return (sum((value - mean_value) ** 2 for value in values) / (len(values) - 1)) ** 0.5


def _dot(left: list[float], right: list[float]) -> float:
    return sum(left[index] * right[index] for index in range(len(left)))


def _euclidean(left: list[float], right: list[float]) -> float:
    return sum((left[index] - right[index]) ** 2 for index in range(len(left))) ** 0.5


def _project_rows(rows: list[list[float]], vectors: list[list[float]], k: int) -> list[list[float]]:
    active = vectors[:k]
    return [[_dot(row, vector) for vector in active] for row in rows]


def _reconstruct_rows(projected_rows: list[list[float]], vectors: list[list[float]], k: int) -> list[list[float]]:
    active = vectors[:k]
    if not active:
        return [[0.0 for _ in range(len(vectors[0]))] for _ in projected_rows] if vectors else []
    reconstructed: list[list[float]] = []
    for projected in projected_rows:
        row = [0.0 for _ in range(len(active[0]))]
        for component_index, vector in enumerate(active):
            weight = projected[component_index]
            for feature_index, loading in enumerate(vector):
                row[feature_index] += weight * loading
        reconstructed.append(row)
    return reconstructed


def _row_mean(rows: list[list[float]]) -> list[float]:
    if not rows:
        return []
    dimension = len(rows[0])
    return [_mean([row[index] for row in rows]) for index in range(dimension)]


def _centroid(rows: list[list[float]]) -> list[float]:
    return _row_mean(rows)


def _class_labels(coordinates: tuple[dict[str, object], ...]) -> tuple[str, ...]:
    return tuple(str(row["true_class"]) for row in coordinates)


def _unique_labels(labels: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(sorted(set(labels)))


def _farthest_first_initialization(rows: list[list[float]], k: int) -> list[list[float]]:
    if not rows or k <= 0:
        return []
    ordered = [rows[index] for index in sorted(range(len(rows)), key=lambda index: tuple(rows[index]))]
    centroids = [ordered[0][:]]
    while len(centroids) < k and len(centroids) < len(ordered):
        next_point = max(
            ordered,
            key=lambda point: min(_euclidean(point, centroid) for centroid in centroids),
        )
        if any(all(abs(next_point[index] - centroid[index]) <= 1e-12 for index in range(len(next_point))) for centroid in centroids):
            break
        centroids.append(next_point[:])
    while len(centroids) < k:
        centroids.append(centroids[-1][:])
    return centroids


def _kmeans(rows: list[list[float]], k: int, max_iter: int = 100) -> tuple[list[int], list[list[float]], float]:
    if not rows or k <= 0:
        return [], [], 0.0
    centroids = _farthest_first_initialization(rows, k)
    labels = [0 for _ in rows]
    for _ in range(max_iter):
        changed = False
        for row_index, row in enumerate(rows):
            label = min(range(k), key=lambda centroid_index: _euclidean(row, centroids[centroid_index]))
            if labels[row_index] != label:
                labels[row_index] = label
                changed = True
        new_centroids: list[list[float]] = []
        for centroid_index in range(k):
            cluster_rows = [row for row, label in zip(rows, labels) if label == centroid_index]
            if cluster_rows:
                new_centroids.append(_centroid(cluster_rows))
            else:
                new_centroids.append(centroids[centroid_index][:])
        if not changed and all(
            _euclidean(old, new) <= 1e-9 for old, new in zip(centroids, new_centroids)
        ):
            centroids = new_centroids
            break
        centroids = new_centroids
    inertia = sum(_euclidean(row, centroids[label]) ** 2 for row, label in zip(rows, labels))
    return labels, centroids, inertia


def _silhouette_score(rows: list[list[float]], labels: list[int]) -> float:
    if len(rows) < 2 or len(set(labels)) < 2:
        return 0.0
    scores: list[float] = []
    for index, row in enumerate(rows):
        same_cluster = [other for other, label in zip(rows, labels) if label == labels[index] and other is not row]
        if same_cluster:
            a = _mean([_euclidean(row, other) for other in same_cluster])
        else:
            a = 0.0
        b_values = []
        for cluster_id in sorted(set(labels)):
            if cluster_id == labels[index]:
                continue
            cluster_rows = [other for other, label in zip(rows, labels) if label == cluster_id]
            if cluster_rows:
                b_values.append(_mean([_euclidean(row, other) for other in cluster_rows]))
        if not b_values:
            continue
        b = min(b_values)
        scores.append((b - a) / max(a, b, 1e-12))
    return _mean(scores)


def _cluster_purity(labels: list[int], truth: list[str]) -> float:
    clusters: dict[int, list[str]] = {}
    for cluster_id, label in zip(labels, truth):
        clusters.setdefault(cluster_id, []).append(label)
    correct = 0
    for cluster_labels in clusters.values():
        counts: dict[str, int] = {}
        for label in cluster_labels:
            counts[label] = counts.get(label, 0) + 1
        correct += max(counts.values())
    return correct / max(len(truth), 1)


def _cluster_balance(labels: list[int]) -> float:
    counts = [labels.count(cluster_id) for cluster_id in sorted(set(labels))]
    if not counts:
        return 0.0
    return min(counts) / max(counts)


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
    total_variance = sum(component.explained_variance for component in pca_result.components) or 1.0
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
        cluster_rows = [dict(labels=labels[index], truth=truth[index]) for index in range(len(labels))]
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
    lines = [
        "# PCA Dimensionality Audit",
        "",
        "This audit answers whether the current feature set is sufficiently low-dimensional for clustering and whether 2D is enough for class separation.",
        "",
        f"- Feature set: {result.feature_set_name}",
        f"- Feature count: {len(result.feature_names)}",
        f"- Class count: {len(result.class_names)}",
        f"- Samples: {result.sample_count}",
        "",
        "## Recommendation",
        "",
        f"- 95% variance needs at least `{result.recommendation['recommended_components_for_95pct_variance']}` principal components.",
        f"- Best clusterability appears at `{result.recommendation['best_clusterability_component_count']}` principal components.",
        f"- Best silhouette: `{float(result.recommendation['best_clusterability_silhouette']):.3f}`",
        f"- Best cluster purity: `{float(result.recommendation['best_clusterability_purity']):.3f}`",
        "",
        "## Component Sweep",
        "",
        "| components | cumulative_variance_ratio | reconstruction_rmse | reconstruction_mae | centroid_separation | kmeans_silhouette | kmeans_cluster_purity |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in result.component_rows:
        lines.append(
            f"| {row.component_count} | {row.cumulative_variance_ratio:.3f} | {row.reconstruction_rmse:.3f} | {row.reconstruction_mae:.3f} | {row.class_centroid_separation:.3f} | {row.kmeans_silhouette:.3f} | {row.kmeans_cluster_purity:.3f} |"
        )
    lines.extend(
        [
            "",
            "## Clusterability",
            "",
            "This section uses a deterministic k-means sweep on the PCA projections to ask whether the class structure is recoverable from low-dimensional embeddings.",
            "",
            "## Interpretation",
            "",
            f"- `PC{best_clusterability.component_count}` currently gives the strongest combined silhouette and purity."
            if best_clusterability
            else "- No PCA components were available for a clusterability recommendation.",
            "- If silhouette stays low while cumulative variance is already high, the feature set is compressible but not cluster-separable.",
            "- If purity improves materially as components increase, the 2D picture is hiding structure that is useful for clustering.",
            "- Reconstruction error is reported in standardized feature units, so it is comparable across feature sets.",
        ]
    )
    return "\n".join(lines)


def _render_variance_plot(result: PcaDimensionalityResult):
    plt = _prepare_matplotlib()
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
    plt = _prepare_matplotlib()
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
    plt = _prepare_matplotlib()
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


def _figure_to_png(fig) -> bytes:
    plt = _prepare_matplotlib()
    buffer = io.BytesIO()
    try:
        fig.savefig(buffer, format="png", dpi=160, bbox_inches="tight")
        return buffer.getvalue()
    finally:
        plt.close(fig)


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
    _write_csv(
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
    _write_csv(
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
