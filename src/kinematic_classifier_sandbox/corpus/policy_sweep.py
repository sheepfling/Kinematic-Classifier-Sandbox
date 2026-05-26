from __future__ import annotations

from pathlib import Path

from .policy import (
    CorpusPolicySpec,
    load_corpus_policy_spec,
    write_default_policy_artifacts,
)
from .policy_sweep_rendering import (
    write_corpus_policy_tuning_artifacts as _write_policy_sweep_artifacts,
)
from .policy_sweep_types import CorpusPolicyTuningArtifacts, PolicyEvaluationRow
from .policy_sweep_utils import (
    _ablation_rows,
    _dev_holdout_rows,
    _evaluate_policy,
    _gate_threshold_rows,
    _jaccard_rows,
    _local_perturbation_rows,
    _pareto_rows,
    _policy_variants,
    _rank_stability_rows,
    _recommended_policy,
    _sampler_budget_rows,
)


def write_corpus_policy_tuning_artifacts(
    output_dir: str | Path,
    *,
    policy: CorpusPolicySpec | None = None,
    seed: int = 11,
) -> CorpusPolicyTuningArtifacts:
    run_dir = Path(output_dir) / "corpus_hyperparameter_tuning_v1"
    run_dir.mkdir(parents=True, exist_ok=True)
    default_policy = policy or load_corpus_policy_spec()
    write_default_policy_artifacts(Path(output_dir), default_policy)

    baseline = _evaluate_policy(default_policy, seed=seed, objective_id="dev_boundary_switching").as_row_dict()
    variants = _policy_variants(default_policy)
    sweep_rows = [_evaluate_policy(variant, seed=seed, objective_id="dev_boundary_switching").as_row_dict() for variant in variants]
    ablation_rows = _ablation_rows(default_policy, baseline, seed)
    perturbation_rows = _local_perturbation_rows(default_policy, baseline, seed)
    jaccard_rows = _jaccard_rows([baseline, *sweep_rows, *perturbation_rows])
    rank_rows = _rank_stability_rows([baseline, *sweep_rows, *perturbation_rows], baseline)
    sampler_rows = _sampler_budget_rows(default_policy)
    gate_rows = _gate_threshold_rows(default_policy, baseline)
    dev_holdout_rows = _dev_holdout_rows(default_policy, variants, seed)
    pareto_rows = _pareto_rows([baseline, *sweep_rows, *perturbation_rows])
    recommended = _recommended_policy([baseline, *sweep_rows, *perturbation_rows], default_policy, variants)
    return _write_policy_sweep_artifacts(
        run_dir=run_dir,
        baseline=baseline,
        sweep_rows=sweep_rows,
        ablation_rows=ablation_rows,
        perturbation_rows=perturbation_rows,
        jaccard_rows=jaccard_rows,
        rank_rows=rank_rows,
        sampler_rows=sampler_rows,
        gate_rows=gate_rows,
        dev_holdout_rows=dev_holdout_rows,
        pareto_rows=pareto_rows,
        recommended=recommended,
        seed=seed,
    )
