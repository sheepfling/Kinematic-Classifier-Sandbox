from __future__ import annotations

import random
from pathlib import Path

from ..objectives import (
    CorpusObjectiveSpec,
    default_corpus_objectives,
    load_corpus_objectives_from_yaml,
)
from .backend_adapter_proof_types import BackendCandidateSpec
from .candidate_generation_types import CandidateGenerationResult
from .candidate_generation_utils import (
    _archive_mutation_sampler,
    _backend_constraints_for_objective,
    _boundary_mutation_sampler,
    _candidate_row,
    _grid_sampler,
    _lhs_sampler,
    _random_sampler,
    _stress_mutation_sampler,
)
from .capability_aware_search import analyze_capability_aware_search


def generate_candidates_from_objectives(
    objectives: tuple[CorpusObjectiveSpec, ...] | None = None,
) -> tuple[BackendCandidateSpec, ...]:
    objective_list = objectives or default_corpus_objectives()
    candidates: list[BackendCandidateSpec] = []
    for objective in objective_list:
        backend_ids = _backend_constraints_for_objective(objective)
        for backend_id in backend_ids:
            budget = max(12, len(objective.target_environment_regimes) * 4 if objective.target_environment_regimes else 12)
            candidates.extend(_random_sampler(objective, backend_id, budget, random.Random(objective.objective_id)))
            candidates.extend(_grid_sampler(objective, backend_id, budget))
            candidates.extend(_lhs_sampler(objective, backend_id, budget))
            candidates.extend(_boundary_mutation_sampler(objective, backend_id, budget))
            candidates.extend(_archive_mutation_sampler(objective, backend_id, budget))
            candidates.extend(_stress_mutation_sampler(objective, backend_id, budget))
    return tuple(candidates)


def generate_candidates_from_objective_file(path: str | Path) -> tuple[BackendCandidateSpec, ...]:
    return generate_candidates_from_objectives(load_corpus_objectives_from_yaml(path))


def analyze_candidate_generation() -> CandidateGenerationResult:
    objectives = default_corpus_objectives()
    candidates = generate_candidates_from_objectives(objectives)
    search_analysis = analyze_capability_aware_search()
    generated_candidate_rows = tuple(
        _candidate_row(candidate, candidate.provenance.get("sampler_name", ""), candidate.provenance.get("parent_candidate_id", ""))
        for candidate in candidates
    )
    sampler_names = sorted({str(row["sampler_name"]) for row in generated_candidate_rows})
    sampler_manifest = {
        "objective_count": len(objectives),
        "candidate_count": len(candidates),
        "samplers": sampler_names,
        "search_analysis": search_analysis.search_planner_rules,
    }
    report_markdown = "\n".join(
        [
            "# Candidate Generation",
            "",
            "This artifact establishes the candidate sampler surface used by corpus exploration.",
            "Candidate generation is now objective-driven and routed through the exploration helper stack.",
            "",
            f"- Objective count: `{len(objectives)}`",
            f"- Candidate count: `{len(candidates)}`",
            "",
            "## Reading Notes",
            "",
            "- This layer feeds search and corpus exploration tools.",
            "- The shared helper logic now lives in `candidate_generation_utils.py`.",
            "- Rendering and artifact output now live in `candidate_generation_rendering.py`.",
        ]
    )
    return CandidateGenerationResult(
        sampler_manifest=sampler_manifest,
        generated_candidate_rows=generated_candidate_rows,
        report_markdown=report_markdown,
    )
