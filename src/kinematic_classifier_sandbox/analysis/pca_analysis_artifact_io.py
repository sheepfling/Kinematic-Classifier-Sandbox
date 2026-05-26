from __future__ import annotations

from pathlib import Path

from kinematic_classifier_sandbox.utils.io import write_csv

from ..utils.plotting import _figure_to_png
from .pca_analysis_contracts import PcaAnalysisArtifacts
from .pca_analysis_reporting import (
    render_pca_analysis_report,
    render_pca_loadings,
    render_pca_scatter,
    render_pca_variance,
)


def write_pca_analysis_artifacts(
    output_dir: str | Path,
    *,
    seed: int = 7,
    trajectories_per_class: int = 5,
    n_components: int = 3,
    feature_set: str | None = None,
    feature_names: tuple[str, ...] | list[str] | None = None,
) -> PcaAnalysisArtifacts:
    from .pca_analysis import analyze_feature_pca

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

    report_path.write_text(render_pca_analysis_report(result), encoding="utf-8")
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
    write_csv(coordinates_path, [dict(row) for row in result.coordinates], coordinate_fields)
    write_csv(
        loadings_path,
        [{"component": f"PC{component.component_index}", **component.loadings} for component in result.components],
        ["component", *result.feature_names],
    )
    write_csv(
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
    plot_scatter_png_path.write_bytes(_figure_to_png(render_pca_scatter(result)))
    plot_variance_png_path.write_bytes(_figure_to_png(render_pca_variance(result)))
    plot_loadings_png_path.write_bytes(_figure_to_png(render_pca_loadings(result)))
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
