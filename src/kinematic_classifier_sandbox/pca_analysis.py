from __future__ import annotations

from dataclasses import dataclass, asdict
from math import sqrt
import csv
import io
import json
import os
from pathlib import Path

from .feature_analysis import FeatureAnalysisResult, analyze_feature_datasets


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
    return sqrt(sum((value - mean_value) ** 2 for value in values) / (len(values) - 1))


def _dot(left: list[float], right: list[float]) -> float:
    return sum(left[index] * right[index] for index in range(len(left)))


def _matvec(matrix: list[list[float]], vector: list[float]) -> list[float]:
    return [sum(row[index] * vector[index] for index in range(len(vector))) for row in matrix]


def _vector_norm(vector: list[float]) -> float:
    return sqrt(sum(value * value for value in vector))


def _normalize(vector: list[float]) -> list[float]:
    norm = _vector_norm(vector)
    if norm <= 1e-12:
        return [0.0 for _ in vector]
    return [value / norm for value in vector]


def _outer(vector: list[float]) -> list[list[float]]:
    return [[vector[row] * vector[col] for col in range(len(vector))] for row in range(len(vector))]


def _deflate(matrix: list[list[float]], eigenvalue: float, eigenvector: list[float]) -> list[list[float]]:
    outer = _outer(eigenvector)
    return [
        [matrix[row][col] - eigenvalue * outer[row][col] for col in range(len(matrix[row]))]
        for row in range(len(matrix))
    ]


def _covariance_matrix(rows: list[list[float]]) -> list[list[float]]:
    if not rows:
        return []
    dimension = len(rows[0])
    means = [_mean([row[index] for row in rows]) for index in range(dimension)]
    covariance = [[0.0 for _ in range(dimension)] for _ in range(dimension)]
    for row in rows:
        centered = [row[index] - means[index] for index in range(dimension)]
        for i in range(dimension):
            for j in range(dimension):
                covariance[i][j] += centered[i] * centered[j]
    denom = max(len(rows) - 1, 1)
    for i in range(dimension):
        for j in range(dimension):
            covariance[i][j] /= denom
    return covariance


def _power_iteration(matrix: list[list[float]], max_iter: int = 200, tolerance: float = 1e-9) -> tuple[float, list[float]]:
    if not matrix:
        return 0.0, []
    dimension = len(matrix)
    vector = _normalize([1.0 + index for index in range(dimension)])
    previous_value = 0.0
    for _ in range(max_iter):
        next_vector = _matvec(matrix, vector)
        vector = _normalize(next_vector)
        value = _dot(vector, _matvec(matrix, vector))
        if abs(value - previous_value) <= tolerance:
            return value, vector
        previous_value = value
    return previous_value, vector


def _standardize_matrix(rows: list[dict[str, float]], feature_names: tuple[str, ...]) -> tuple[list[list[float]], dict[str, float], dict[str, float]]:
    means = {name: _mean([row[name] for row in rows]) for name in feature_names}
    stds = {name: max(_std([row[name] for row in rows]), 1e-9) for name in feature_names}
    matrix = [[(row[name] - means[name]) / stds[name] for name in feature_names] for row in rows]
    return matrix, means, stds


@dataclass(frozen=True, slots=True)
class PcaComponent:
    component_index: int
    explained_variance: float
    explained_variance_ratio: float
    loadings: dict[str, float]


@dataclass(frozen=True, slots=True)
class PcaAnalysisResult:
    feature_analysis: FeatureAnalysisResult
    feature_set_name: str
    feature_names: tuple[str, ...]
    coordinates: tuple[dict[str, object], ...]
    components: tuple[PcaComponent, ...]
    standardization_means: dict[str, float]
    standardization_stds: dict[str, float]


@dataclass(frozen=True, slots=True)
class PcaAnalysisArtifacts:
    run_dir: Path
    report_path: Path
    coordinates_path: Path
    loadings_path: Path
    explained_variance_path: Path
    config_path: Path
    plot_scatter_png_path: Path
    plot_variance_png_path: Path
    plot_loadings_png_path: Path


def analyze_feature_pca(
    *,
    seed: int = 7,
    trajectories_per_class: int = 5,
    n_components: int = 3,
    feature_set: str | None = None,
    feature_names: tuple[str, ...] | list[str] | None = None,
) -> PcaAnalysisResult:
    feature_analysis = analyze_feature_datasets(
        seed=seed,
        trajectories_per_class=trajectories_per_class,
        feature_set=feature_set,
        feature_names=feature_names,
    )
    selected_feature_names = feature_analysis.summary.feature_names
    numeric_rows = []
    metadata_rows = []
    for row in feature_analysis.feature_rows:
        numeric_rows.append({name: float(getattr(row, name)) for name in selected_feature_names})
        metadata_rows.append(
            {
                "trajectory_id": row.trajectory_id,
                "tier": row.tier,
                "scenario_id": row.scenario_id,
                "true_class": row.true_class,
                "seed": row.seed,
            }
        )
    matrix, means, stds = _standardize_matrix(numeric_rows, selected_feature_names)
    covariance = _covariance_matrix(matrix)
    total_variance = sum(covariance[index][index] for index in range(len(covariance))) if covariance else 0.0

    components: list[PcaComponent] = []
    working = [row[:] for row in covariance]
    component_vectors: list[list[float]] = []
    component_count = min(n_components, len(selected_feature_names))
    for component_index in range(component_count):
        eigenvalue, eigenvector = _power_iteration(working)
        if not eigenvector:
            break
        component_vectors.append(eigenvector)
        components.append(
            PcaComponent(
                component_index=component_index + 1,
                explained_variance=eigenvalue,
                explained_variance_ratio=(eigenvalue / total_variance) if total_variance > 0.0 else 0.0,
                loadings={feature_name: eigenvector[index] for index, feature_name in enumerate(selected_feature_names)},
            )
        )
        working = _deflate(working, eigenvalue, eigenvector)

    coordinates: list[dict[str, object]] = []
    for row_index, row in enumerate(matrix):
        coordinate_row = dict(metadata_rows[row_index])
        for component_index, vector in enumerate(component_vectors, start=1):
            coordinate_row[f"pc{component_index}"] = _dot(row, vector)
        coordinates.append(coordinate_row)

    return PcaAnalysisResult(
        feature_analysis=feature_analysis,
        feature_set_name=feature_analysis.summary.feature_set_name,
        feature_names=selected_feature_names,
        coordinates=tuple(coordinates),
        components=tuple(components),
        standardization_means=means,
        standardization_stds=stds,
    )


def _render_scatter(result: PcaAnalysisResult):
    plt = _prepare_matplotlib()
    fig, ax = plt.subplots(figsize=(8.6, 6.2))
    class_names = sorted({row["true_class"] for row in result.coordinates})
    colors = {
        name: color
        for name, color in zip(
            class_names,
            ("#2563eb", "#16a34a", "#7c3aed", "#d97706", "#db2777", "#0f766e", "#dc2626"),
        )
    }
    for class_name in class_names:
        rows = [row for row in result.coordinates if row["true_class"] == class_name]
        ax.scatter(
            [float(row.get("pc1", 0.0)) for row in rows],
            [float(row.get("pc2", 0.0)) for row in rows],
            label=class_name,
            color=colors[class_name],
            alpha=0.8,
            s=28,
        )
    ax.set_title("PCA Feature Scatter", loc="left", fontweight="bold")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.grid(True, alpha=0.2)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    return fig


def _render_variance(result: PcaAnalysisResult):
    plt = _prepare_matplotlib()
    fig, ax = plt.subplots(figsize=(7.4, 4.8))
    indices = [component.component_index for component in result.components]
    ratios = [component.explained_variance_ratio for component in result.components]
    cumulative = []
    running = 0.0
    for ratio in ratios:
        running += ratio
        cumulative.append(running)
    ax.bar(indices, ratios, color="#2563eb", alpha=0.85, label="explained")
    ax.plot(indices, cumulative, color="#dc2626", linewidth=2.0, marker="o", label="cumulative")
    ax.set_title("Explained Variance", loc="left", fontweight="bold")
    ax.set_xlabel("component")
    ax.set_ylabel("variance ratio")
    ax.set_ylim(0.0, 1.05)
    ax.grid(True, alpha=0.2, axis="y")
    ax.legend(frameon=False)
    fig.tight_layout()
    return fig


def _render_loadings(result: PcaAnalysisResult):
    plt = _prepare_matplotlib()
    component_labels = [f"PC{component.component_index}" for component in result.components]
    matrix = [
        [component.loadings[feature_name] for feature_name in result.feature_names]
        for component in result.components
    ]
    fig, ax = plt.subplots(figsize=(11.0, 4.8))
    image = ax.imshow(matrix, cmap="coolwarm", vmin=-1.0, vmax=1.0)
    ax.set_title("PCA Loadings", loc="left", fontweight="bold")
    ax.set_xticks(range(len(result.feature_names)))
    ax.set_xticklabels(result.feature_names, rotation=35, ha="right")
    ax.set_yticks(range(len(component_labels)))
    ax.set_yticklabels(component_labels)
    for row_index, row in enumerate(matrix):
        for col_index, value in enumerate(row):
            ax.text(col_index, row_index, f"{value:.2f}", ha="center", va="center", fontsize=7)
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    return fig


def _figure_to_svg(fig) -> str:
    plt = _prepare_matplotlib()
    buffer = io.StringIO()
    try:
        fig.savefig(buffer, format="svg", bbox_inches="tight")
        return buffer.getvalue()
    finally:
        plt.close(fig)


def _figure_to_png(fig) -> bytes:
    plt = _prepare_matplotlib()
    buffer = io.BytesIO()
    try:
        fig.savefig(buffer, format="png", dpi=160, bbox_inches="tight")
        return buffer.getvalue()
    finally:
        plt.close(fig)


def _render_report(result: PcaAnalysisResult) -> str:
    lines = [
        "# PCA Feature Analysis",
        "",
        "This PCA is a diagnostic over the engineered feature matrix, not a replacement classifier.",
        "",
        f"- Feature set: {result.feature_set_name}",
        f"- Active features: {', '.join(result.feature_names)}",
        "",
        "## Explained Variance",
        "",
        "| component | explained_variance | explained_variance_ratio |",
        "| --- | ---: | ---: |",
    ]
    for component in result.components:
        lines.append(
            f"| PC{component.component_index} | {component.explained_variance:.4f} | {component.explained_variance_ratio:.4f} |"
        )
    lines.extend(["", "## Dominant Loadings", ""])
    for component in result.components[:3]:
        top_loadings = sorted(component.loadings.items(), key=lambda item: abs(item[1]), reverse=True)[:5]
        lines.append(f"### PC{component.component_index}")
        lines.append("")
        for feature_name, value in top_loadings:
            lines.append(f"- `{feature_name}`: {value:.4f}")
        lines.append("")
    lines.extend(
        [
            "## Notes",
            "",
            "- PCA is run on standardized engineered features.",
            "- Class-colored scatter plots help check whether major variance directions align with class identity.",
            "- Strong loadings identify which feature groups dominate the first principal directions.",
        ]
    )
    return "\n".join(lines)


def write_pca_analysis_artifacts(
    output_dir: str | Path,
    *,
    seed: int = 7,
    trajectories_per_class: int = 5,
    n_components: int = 3,
    feature_set: str | None = None,
    feature_names: tuple[str, ...] | list[str] | None = None,
) -> PcaAnalysisArtifacts:
    result = analyze_feature_pca(
        seed=seed,
        trajectories_per_class=trajectories_per_class,
        n_components=n_components,
        feature_set=feature_set,
        feature_names=feature_names,
    )
    output_root = Path(output_dir)
    run_dir = output_root / (
        "pca_analysis_v1" if result.feature_set_name == "all_engineered" else f"pca_analysis_{result.feature_set_name}_v1"
    )
    run_dir.mkdir(parents=True, exist_ok=True)

    report_path = run_dir / "pca_report.md"
    coordinates_path = run_dir / "pca_coordinates.csv"
    loadings_path = run_dir / "pca_loadings.csv"
    explained_variance_path = run_dir / "pca_explained_variance.csv"
    config_path = run_dir / "pca_config.yaml"
    plot_scatter_png_path = run_dir / "pc1_pc2_by_class.png"
    plot_variance_png_path = run_dir / "explained_variance.png"
    plot_loadings_png_path = run_dir / "loadings_heatmap.png"

    report_path.write_text(_render_report(result), encoding="utf-8")
    config_path.write_text(
        "\n".join(
            [
                "experiment:",
                f"  name: {run_dir.name}",
                f"  seed: {seed}",
                f"  trajectories_per_class: {trajectories_per_class}",
                f"  n_components: {n_components}",
                f"  feature_set: {result.feature_set_name}",
                f"  feature_names: [{', '.join(result.feature_names)}]",
                "",
            ]
        ),
        encoding="utf-8",
    )
    coordinate_fields = ["trajectory_id", "tier", "scenario_id", "true_class", "seed", *[f"pc{index}" for index in range(1, len(result.components) + 1)]]
    _write_csv(coordinates_path, [dict(row) for row in result.coordinates], coordinate_fields)
    _write_csv(
        loadings_path,
        [
            {"component": f"PC{component.component_index}", **component.loadings}
            for component in result.components
        ],
        ["component", *result.feature_names],
    )
    _write_csv(
        explained_variance_path,
        [
            {
                "component": f"PC{component.component_index}",
                "explained_variance": component.explained_variance,
                "explained_variance_ratio": component.explained_variance_ratio,
            }
            for component in result.components
        ],
        ["component", "explained_variance", "explained_variance_ratio"],
    )
    plot_scatter_png_path.write_bytes(_figure_to_png(_render_scatter(result)))
    plot_variance_png_path.write_bytes(_figure_to_png(_render_variance(result)))
    plot_loadings_png_path.write_bytes(_figure_to_png(_render_loadings(result)))
    return PcaAnalysisArtifacts(
        run_dir=run_dir,
        report_path=report_path,
        coordinates_path=coordinates_path,
        loadings_path=loadings_path,
        explained_variance_path=explained_variance_path,
        config_path=config_path,
        plot_scatter_png_path=plot_scatter_png_path,
        plot_variance_png_path=plot_variance_png_path,
        plot_loadings_png_path=plot_loadings_png_path,
    )
