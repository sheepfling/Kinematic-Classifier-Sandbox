from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class FeatureAnalysisArtifacts:
    run_dir: Path
    report_path: Path
    feature_matrix_path: Path
    feature_summary_path: Path
    feature_excitation_path: Path
    feature_excitation_summary_path: Path
    feature_caveats_path: Path
    feature_separation_scores_path: Path
    identifiability_matrix_path: Path
    pairwise_distance_matrix_path: Path
    pairwise_overlap_matrix_path: Path
    pairwise_auc_matrix_path: Path
    plot_excitation_png_path: Path
    plot_distance_png_path: Path
    plot_overlap_png_path: Path
    plot_scatter_png_path: Path
    plot_confusability_png_path: Path
    plot_ranking_png_path: Path


__all__ = ["FeatureAnalysisArtifacts"]
