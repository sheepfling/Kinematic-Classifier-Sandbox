from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

from ..common_experiment.contracts import CommonExperimentResult
from ..corpus.autodevelopment_types import CorpusAutodevelopmentResult
from ..corpus.policy import CorpusPolicySpec, corpus_policy_to_dict
from ..study_candidate_generation import StudyCandidateGenerationResult
from ..study_candidate_protocol import (
    StudyCandidateProtocolResult,
    analyze_study_candidate_protocol,
)
from ..validation.validation_ladder_contracts import ValidationLadderResult
from .cached_analysis import (
    cached_common_experiment_analysis,
    cached_corpus_autodevelopment_analysis,
    cached_study_candidate_generation_analysis,
    cached_validation_ladder_analysis,
)


@dataclass(frozen=True, slots=True)
class MethodologyExecutionContext:
    protocol_result: StudyCandidateProtocolResult
    common_result: CommonExperimentResult
    corpus_result: CorpusAutodevelopmentResult
    study_generation_result: StudyCandidateGenerationResult
    validation_result: ValidationLadderResult


_METHODOLOGY_CONTEXT_CACHE: dict[str, MethodologyExecutionContext] = {}


def _context_cache_key(
    *,
    seed: int,
    trajectories_per_case: int,
    config_path: str | Path | None,
    policy: CorpusPolicySpec | None,
    use_cache: bool,
) -> str:
    payload = {
        "seed": seed,
        "trajectories_per_case": trajectories_per_case,
        "config_path": str(Path(config_path).resolve()) if config_path is not None else None,
        "policy": corpus_policy_to_dict(policy) if policy is not None else None,
        "use_cache": use_cache,
    }
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def build_methodology_execution_context(
    *,
    seed: int = 7,
    trajectories_per_case: int = 6,
    config_path: str | Path | None = None,
    policy: CorpusPolicySpec | None = None,
    use_cache: bool = True,
) -> MethodologyExecutionContext:
    cache_key = None
    if use_cache:
        cache_key = _context_cache_key(
            seed=seed,
            trajectories_per_case=trajectories_per_case,
            config_path=config_path,
            policy=policy,
            use_cache=use_cache,
        )
        cached = _METHODOLOGY_CONTEXT_CACHE.get(cache_key)
        if cached is not None:
            return cached
    protocol = analyze_study_candidate_protocol()
    common = cached_common_experiment_analysis(
        config_path=config_path,
        seed=seed,
        trajectories_per_case=trajectories_per_case,
        use_cache=use_cache,
    )
    corpus = cached_corpus_autodevelopment_analysis(
        seed=seed,
        policy=policy,
        use_cache=use_cache,
    )
    study_generation = cached_study_candidate_generation_analysis(
        seed=seed,
        trajectories_per_case=trajectories_per_case,
        policy=policy,
        protocol_result=protocol,
        common_result=common,
        corpus_result=corpus,
        use_cache=use_cache,
    )
    validation = cached_validation_ladder_analysis(
        seed=seed,
        trajectories_per_case=trajectories_per_case,
        protocol_result=protocol,
        common_result=common,
        corpus_result=corpus,
        study_generation_result=study_generation,
        use_cache=use_cache,
    )
    context = MethodologyExecutionContext(
        protocol_result=protocol,
        common_result=common,
        corpus_result=corpus,
        study_generation_result=study_generation,
        validation_result=validation,
    )
    if cache_key is not None:
        _METHODOLOGY_CONTEXT_CACHE[cache_key] = context
    return context


__all__ = ["MethodologyExecutionContext", "build_methodology_execution_context"]
