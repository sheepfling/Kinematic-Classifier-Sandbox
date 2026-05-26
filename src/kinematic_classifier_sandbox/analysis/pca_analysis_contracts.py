from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class PcaComponent:
    component_index: int
    explained_variance: float
    explained_variance_ratio: float
    loadings: dict[str, float]


@dataclass(frozen=True, slots=True)
class PcaAnalysisResult:
    feature_analysis: object
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
