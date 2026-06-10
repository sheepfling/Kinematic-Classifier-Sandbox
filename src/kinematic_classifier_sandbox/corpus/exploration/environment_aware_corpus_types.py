from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class EnvironmentAwareCorpusResult:
    environment_manifest: dict[str, Any]
    environment_coverage_rows: tuple[dict[str, Any], ...]
    environment_leakage_rows: tuple[dict[str, Any], ...]
    report_markdown: str


@dataclass(frozen=True, slots=True)
class EnvironmentAwareCorpusArtifacts:
    run_dir: Path
    environment_manifest_path: Path
    environment_coverage_path: Path
    environment_leakage_audit_path: Path
    report_path: Path
    coverage_heatmap_png_path: Path
    leakage_plot_png_path: Path
    trajectory_gallery_png_path: Path
