from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class RlBackendDecisionResult:
    rl_justified: bool
    state_space: tuple[str, ...]
    action_space: tuple[str, ...]
    reward_components: tuple[str, ...]
    episode_definition: str
    baseline_to_beat: dict[str, float]
    success_metric: str
    search_selected_mean_utility: float
    qd_final_coverage_fraction: float
    qd_best_feature_excitation: float
    stress_resolved_modes: int
    stress_total_modes: int
    stress_improved_modes: tuple[str, ...]
    offpolicy_mean_best_policy_minus_best_baseline: float
    offpolicy_seed_promotion_rate: float
    offpolicy_best_policy_backend: str
    decision_rows: tuple["RlBackendDecisionGateRow", ...]


@dataclass(frozen=True, slots=True)
class RlBackendDecisionGateRow:
    criterion: str
    status: str
    value: str | float
    note: str

    def __getitem__(self, key: str) -> str | float:
        return getattr(self, key)


@dataclass(frozen=True, slots=True)
class RlBackendDecisionArtifacts:
    run_dir: Path
    report_path: Path
    summary_path: Path
    evidence_path: Path
