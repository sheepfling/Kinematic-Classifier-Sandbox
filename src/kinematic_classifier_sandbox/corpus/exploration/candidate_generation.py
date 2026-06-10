from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from . import candidate_generation_core as _core
from .candidate_generation_rendering import (
    write_candidate_generation_artifacts as _write_candidate_generation_artifacts,
)

if TYPE_CHECKING:
    from ..objectives import CorpusObjectiveSpec
    from .backend_adapter_proof_types import BackendCandidateSpec
    from .candidate_generation_types import CandidateGenerationArtifacts, CandidateGenerationResult


def generate_candidates_from_objectives(
    objectives: tuple[CorpusObjectiveSpec, ...] | None = None,
) -> tuple[BackendCandidateSpec, ...]:
    return _core.generate_candidates_from_objectives(objectives)


def generate_candidates_from_objective_file(path: str | Path) -> tuple[BackendCandidateSpec, ...]:
    return _core.generate_candidates_from_objective_file(path)


def analyze_candidate_generation() -> CandidateGenerationResult:
    return _core.analyze_candidate_generation()


def write_candidate_generation_artifacts(
    output_dir: str | Path,
    *,
    result: CandidateGenerationResult | None = None,
) -> CandidateGenerationArtifacts:
    return _write_candidate_generation_artifacts(output_dir, result=result)
