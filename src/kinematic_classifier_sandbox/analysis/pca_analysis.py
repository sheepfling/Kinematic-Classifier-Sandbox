from __future__ import annotations

from kinematic_classifier_sandbox.utils.math import (
    covariance_matrix as _covariance_matrix,
)
from kinematic_classifier_sandbox.utils.math import (
    dot_product as _dot,
)
from kinematic_classifier_sandbox.utils.math import (
    matrix_deflation as _deflate,
)
from kinematic_classifier_sandbox.utils.math import (
    mean as _mean,
)
from kinematic_classifier_sandbox.utils.math import (
    power_iteration as _power_iteration,
)
from kinematic_classifier_sandbox.utils.math import (
    std as _std,
)

from .feature_analysis import FeatureAnalysisResult, analyze_feature_datasets
from .pca_analysis_artifact_io import write_pca_analysis_artifacts
from .pca_analysis_contracts import PcaAnalysisArtifacts, PcaAnalysisResult, PcaComponent
from .pca_analysis_reporting import (
    render_pca_analysis_report,
    render_pca_loadings,
    render_pca_scatter,
    render_pca_variance,
)


def _standardize_matrix(rows: list[dict[str, float]], feature_names: tuple[str, ...]) -> tuple[list[list[float]], dict[str, float], dict[str, float]]:
    means = {name: _mean([row[name] for row in rows]) for name in feature_names}
    stds = {name: max(_std([row[name] for row in rows]), 1e-9) for name in feature_names}
    matrix = [[(row[name] - means[name]) / stds[name] for name in feature_names] for row in rows]
    return matrix, means, stds


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
