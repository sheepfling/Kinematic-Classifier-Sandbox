from __future__ import annotations

from pathlib import Path

from ..common_experiment.config import load_common_experiment_config
from ..common_experiment.contracts import CommonExperimentResult
from ..common_experiment.runner import analyze_common_experiment
from ..corpus.autodevelopment import analyze_corpus_autodevelopment
from ..corpus.autodevelopment_types import CorpusAutodevelopmentResult
from ..corpus.policy import CorpusPolicySpec, corpus_policy_to_dict, load_corpus_policy_spec
from ..study_candidate_generation import (
    StudyCandidateGenerationResult,
    analyze_study_candidate_generation,
)
from ..study_candidate_protocol import (
    StudyCandidateProtocolResult,
    analyze_study_candidate_protocol,
)
from ..utils.analysis_cache import (
    file_fingerprint,
    load_or_compute_pickled,
    stable_cache_key,
)
from ..validation.validation_ladder import analyze_validation_ladder
from ..validation.validation_ladder_contracts import ValidationLadderResult


def common_experiment_cache_key(
    *,
    config_path: str | Path | None = None,
    seed: int = 7,
    trajectories_per_case: int = 8,
) -> str:
    config = load_common_experiment_config(config_path)
    payload = {
        "seed": seed,
        "trajectories_per_case": trajectories_per_case,
        "config": file_fingerprint(config.config_path),
        "feature_sets": file_fingerprint(config.feature_sets_path),
        "class_pairs": file_fingerprint(config.class_pair_manifest_path),
        "classifier_manifest": file_fingerprint(config.classifier_manifest_path),
    }
    return stable_cache_key("common_experiment", payload)


def corpus_autodevelopment_cache_key(
    *,
    seed: int = 7,
    policy: CorpusPolicySpec | None = None,
) -> str:
    resolved_policy = policy or load_corpus_policy_spec()
    payload = {
        "seed": seed,
        "policy": corpus_policy_to_dict(resolved_policy),
    }
    return stable_cache_key("corpus_autodevelopment", payload)


def study_candidate_generation_cache_key(
    *,
    seed: int = 7,
    trajectories_per_case: int = 8,
    policy: CorpusPolicySpec | None = None,
) -> str:
    config = load_common_experiment_config()
    resolved_policy = policy or load_corpus_policy_spec()
    payload = {
        "seed": seed,
        "trajectories_per_case": trajectories_per_case,
        "policy": corpus_policy_to_dict(resolved_policy),
        "config": file_fingerprint(config.config_path),
        "feature_sets": file_fingerprint(config.feature_sets_path),
        "class_pairs": file_fingerprint(config.class_pair_manifest_path),
        "classifier_manifest": file_fingerprint(config.classifier_manifest_path),
    }
    return stable_cache_key("study_candidate_generation", payload)


def validation_ladder_cache_key(
    *,
    seed: int = 7,
    trajectories_per_case: int = 6,
) -> str:
    config = load_common_experiment_config()
    payload = {
        "seed": seed,
        "trajectories_per_case": trajectories_per_case,
        "policy": corpus_policy_to_dict(load_corpus_policy_spec()),
        "config": file_fingerprint(config.config_path),
        "feature_sets": file_fingerprint(config.feature_sets_path),
        "class_pairs": file_fingerprint(config.class_pair_manifest_path),
        "classifier_manifest": file_fingerprint(config.classifier_manifest_path),
    }
    return stable_cache_key("validation_ladder", payload)


def cached_common_experiment_analysis(
    *,
    config_path: str | Path | None = None,
    seed: int = 7,
    trajectories_per_case: int = 8,
    use_cache: bool = True,
) -> CommonExperimentResult:
    key = common_experiment_cache_key(
        config_path=config_path,
        seed=seed,
        trajectories_per_case=trajectories_per_case,
    )
    return load_or_compute_pickled(
        namespace="common_experiment",
        cache_key=key,
        enabled=use_cache,
        metadata={
            "config_path": str(Path(config_path).resolve()) if config_path is not None else "default",
            "seed": seed,
            "trajectories_per_case": trajectories_per_case,
        },
        compute=lambda: analyze_common_experiment(
            config_path=config_path,
            seed=seed,
            trajectories_per_case=trajectories_per_case,
        ),
    )


def cached_corpus_autodevelopment_analysis(
    *,
    seed: int = 7,
    policy: CorpusPolicySpec | None = None,
    use_cache: bool = True,
) -> CorpusAutodevelopmentResult:
    resolved_policy = policy or load_corpus_policy_spec()
    key = corpus_autodevelopment_cache_key(seed=seed, policy=resolved_policy)
    return load_or_compute_pickled(
        namespace="corpus_autodevelopment",
        cache_key=key,
        enabled=use_cache,
        metadata={
            "seed": seed,
            "policy": corpus_policy_to_dict(resolved_policy),
        },
        compute=lambda: analyze_corpus_autodevelopment(seed=seed, policy=resolved_policy),
    )


def cached_study_candidate_generation_analysis(
    *,
    seed: int = 7,
    trajectories_per_case: int = 8,
    policy: CorpusPolicySpec | None = None,
    protocol_result: StudyCandidateProtocolResult | None = None,
    common_result: CommonExperimentResult | None = None,
    corpus_result: CorpusAutodevelopmentResult | None = None,
    use_cache: bool = True,
) -> StudyCandidateGenerationResult:
    key = study_candidate_generation_cache_key(
        seed=seed,
        trajectories_per_case=trajectories_per_case,
        policy=policy,
    )
    return load_or_compute_pickled(
        namespace="study_candidate_generation",
        cache_key=key,
        enabled=use_cache,
        metadata={
            "seed": seed,
            "trajectories_per_case": trajectories_per_case,
            "policy": corpus_policy_to_dict(policy) if policy is not None else None,
        },
        compute=lambda: analyze_study_candidate_generation(
            seed=seed,
            trajectories_per_case=trajectories_per_case,
            policy=policy,
            protocol_result=protocol_result,
            common_result=common_result,
            corpus_result=corpus_result,
        ),
    )


def cached_validation_ladder_analysis(
    *,
    seed: int = 7,
    trajectories_per_case: int = 6,
    protocol_result: StudyCandidateProtocolResult | None = None,
    common_result: CommonExperimentResult | None = None,
    corpus_result: CorpusAutodevelopmentResult | None = None,
    study_generation_result: StudyCandidateGenerationResult | None = None,
    use_cache: bool = True,
) -> ValidationLadderResult:
    key = validation_ladder_cache_key(seed=seed, trajectories_per_case=trajectories_per_case)
    return load_or_compute_pickled(
        namespace="validation_ladder",
        cache_key=key,
        enabled=use_cache,
        metadata={
            "seed": seed,
            "trajectories_per_case": trajectories_per_case,
        },
        compute=lambda: analyze_validation_ladder(
            seed=seed,
            trajectories_per_case=trajectories_per_case,
            protocol_result=protocol_result,
            common_result=common_result,
            corpus_result=corpus_result,
            study_generation_result=study_generation_result,
        ),
    )


__all__ = [
    "cached_common_experiment_analysis",
    "cached_corpus_autodevelopment_analysis",
    "cached_study_candidate_generation_analysis",
    "cached_validation_ladder_analysis",
    "common_experiment_cache_key",
    "corpus_autodevelopment_cache_key",
    "study_candidate_generation_cache_key",
    "validation_ladder_cache_key",
]
