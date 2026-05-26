from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class ClassValidityResult:
    class_definition_schema: dict[str, Any]
    score_rows: tuple[dict[str, Any], ...]
    report_markdown: str


@dataclass(frozen=True, slots=True)
class ClassValidityArtifacts:
    run_dir: Path
    class_definition_schema_path: Path
    class_validity_scores_path: Path
    report_path: Path
    confusion_png_path: Path
    status_distribution_png_path: Path
    alternate_similarity_png_path: Path


__all__ = ["ClassValidityArtifacts", "ClassValidityResult"]
