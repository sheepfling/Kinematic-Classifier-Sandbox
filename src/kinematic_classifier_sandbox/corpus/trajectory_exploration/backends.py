from __future__ import annotations

import random

from ..gym_types import CorpusGymAction
from ..search_baseline_utils import _doe_actions, _random_action
from .contracts import (
    TrajectoryExplorationBackend,
    TrajectoryExplorationEvaluation,
    TrajectoryExplorationObjective,
    TrajectoryExplorationProposal,
)

_LOW = (0.65, 0.75, 0.65, 0.65, 0.75)
_HIGH = (1.45, 1.55, 1.90, 2.15, 1.25)


def _clip(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _action_from_vector(seed: int, tier_name: str, vector: tuple[float, float, float, float, float]) -> CorpusGymAction:
    return CorpusGymAction(
        seed=seed,
        tier_name=tier_name,
        duration_scale=_clip(vector[0], _LOW[0], _HIGH[0]),
        measurement_scale=_clip(vector[1], _LOW[1], _HIGH[1]),
        irregularity_scale=_clip(vector[2], _LOW[2], _HIGH[2]),
        outlier_scale=_clip(vector[3], _LOW[3], _HIGH[3]),
        step_scale=_clip(vector[4], _LOW[4], _HIGH[4]),
    )


def _vector_from_action(action: CorpusGymAction) -> tuple[float, float, float, float, float]:
    return (
        action.duration_scale,
        action.measurement_scale,
        action.irregularity_scale,
        action.outlier_scale,
        action.step_scale,
    )


def _seed_action_for_objective(rng: random.Random, objective: TrajectoryExplorationObjective, seed: int) -> CorpusGymAction:
    target = objective.target
    if target.target_failure_mode:
        mode = target.target_failure_mode
        if mode == "raw_extrema_failure":
            return CorpusGymAction(seed=seed, tier_name=target.target_tier or "adversarial_v1", duration_scale=0.95, measurement_scale=1.12, irregularity_scale=1.28, outlier_scale=1.82, step_scale=0.96)
        if mode == "irregular_window_failure":
            return CorpusGymAction(seed=seed, tier_name=target.target_tier or "boundary_v1", duration_scale=1.08, measurement_scale=1.05, irregularity_scale=1.65, outlier_scale=1.00, step_scale=1.00)
        if mode == "kalman_mismatch":
            return CorpusGymAction(seed=seed, tier_name=target.target_tier or "stress_v1", duration_scale=0.82, measurement_scale=1.36, irregularity_scale=1.08, outlier_scale=1.18, step_scale=0.86)
    if objective.geometry_target == "maximize_prior_flip_witness":
        return CorpusGymAction(seed=seed, tier_name=target.target_tier or "boundary_v1", duration_scale=0.82, measurement_scale=1.28, irregularity_scale=1.04, outlier_scale=0.96, step_scale=0.88)
    if objective.geometry_target == "fill_underexcited_feature_cells":
        return CorpusGymAction(seed=seed, tier_name=target.target_tier or "adversarial_v1", duration_scale=1.05, measurement_scale=1.10, irregularity_scale=1.16, outlier_scale=1.30, step_scale=0.92)
    if objective.geometry_target == "create_ambiguous_boundary_trajectories":
        return CorpusGymAction(seed=seed, tier_name=target.target_tier or "boundary_v1", duration_scale=0.88, measurement_scale=1.20, irregularity_scale=1.08, outlier_scale=1.02, step_scale=0.90)
    return _random_action(rng, target, seed=seed)


class _BaseBackend(TrajectoryExplorationBackend):
    backend_id = "base"

    def __init__(self) -> None:
        self._rng = random.Random(0)
        self._objective: TrajectoryExplorationObjective | None = None
        self._proposal_counter = 0
        self._iteration = 0

    def initialize(self, objective: TrajectoryExplorationObjective, seed: int) -> None:
        self._objective = objective
        self._rng = random.Random(seed)
        self._proposal_counter = 0
        self._iteration = 0

    def _next_seed(self) -> int:
        self._proposal_counter += 1
        return self._rng.randint(1, 10_000_000)

    def _proposal(self, action: CorpusGymAction, *, parent_id: str | None = None, metadata: dict[str, object] | None = None) -> TrajectoryExplorationProposal:
        return TrajectoryExplorationProposal(
            proposal_id=f"{self.backend_id}_p{self._proposal_counter}",
            backend_id=self.backend_id,
            iteration=self._iteration,
            candidate_index=self._proposal_counter - 1,
            action=action,
            parent_id=parent_id,
            metadata=metadata or {},
        )

    def state_summary(self) -> dict[str, object]:
        return {"backend_id": self.backend_id, "iteration": self._iteration}

    def diagnostics(self) -> dict[str, object]:
        return {}


class HeuristicSearchBackend(_BaseBackend):
    backend_id = "heuristic_search"

    def propose_batch(self, batch_size: int) -> tuple[TrajectoryExplorationProposal, ...]:
        assert self._objective is not None
        target = self._objective.target
        proposals: list[TrajectoryExplorationProposal] = []
        for action in _doe_actions(target, base_seed=self._next_seed()):
            proposals.append(self._proposal(action, metadata={"search_method": "doe_grid"}))
        while len(proposals) < batch_size:
            action = _random_action(self._rng, target, seed=self._next_seed())
            proposals.append(self._proposal(action, metadata={"search_method": "random"}))
        self._iteration += 1
        return tuple(proposals[:batch_size])

    def observe(self, evaluations: tuple[TrajectoryExplorationEvaluation, ...]) -> None:
        return None


class BlackBoxOptimizerBackend(_BaseBackend):
    backend_id = "blackbox_optimizer"

    def __init__(self, *, elite_fraction: float = 0.35) -> None:
        super().__init__()
        self._mean = (1.0, 1.0, 1.0, 1.0, 1.0)
        self._std = (0.18, 0.18, 0.20, 0.22, 0.12)
        self._elite_fraction = elite_fraction
        self._trace_rows: list[dict[str, object]] = []

    def initialize(self, objective: TrajectoryExplorationObjective, seed: int) -> None:
        super().initialize(objective, seed)
        seed_action = _seed_action_for_objective(self._rng, objective, seed)
        self._mean = _vector_from_action(seed_action)
        self._std = (0.16, 0.16, 0.18, 0.18, 0.10)
        self._trace_rows = []

    def propose_batch(self, batch_size: int) -> tuple[TrajectoryExplorationProposal, ...]:
        assert self._objective is not None
        target = self._objective.target
        proposals: list[TrajectoryExplorationProposal] = []
        for _ in range(batch_size):
            vector = tuple(self._rng.gauss(self._mean[index], self._std[index]) for index in range(5))
            action = _action_from_vector(self._next_seed(), target.target_tier or "realistic_v1", vector)
            proposals.append(self._proposal(action, parent_id="cem_mean", metadata={"search_method": "cross_entropy"}))
        self._iteration += 1
        return tuple(proposals)

    def observe(self, evaluations: tuple[TrajectoryExplorationEvaluation, ...]) -> None:
        if not evaluations:
            return
        ranked = sorted(evaluations, key=lambda row: row.total_utility, reverse=True)
        elite_count = max(1, int(len(ranked) * self._elite_fraction))
        elites = ranked[:elite_count]
        elite_vectors = [
            (
                float(row.diagnostics["duration_scale"]),
                float(row.diagnostics["measurement_scale"]),
                float(row.diagnostics["irregularity_scale"]),
                float(row.diagnostics["outlier_scale"]),
                float(row.diagnostics["step_scale"]),
            )
            for row in elites
        ]
        self._mean = tuple(sum(vector[index] for vector in elite_vectors) / elite_count for index in range(5))
        self._std = tuple(max(0.05, (sum(abs(vector[index] - self._mean[index]) for vector in elite_vectors) / elite_count) * 0.90) for index in range(5))
        self._trace_rows.append(
            {
                "iteration": self._iteration,
                "elite_count": elite_count,
                "best_total_utility": ranked[0].total_utility,
                "mean_duration_scale": self._mean[0],
                "mean_measurement_scale": self._mean[1],
                "mean_irregularity_scale": self._mean[2],
                "mean_outlier_scale": self._mean[3],
                "mean_step_scale": self._mean[4],
            }
        )

    def diagnostics(self) -> dict[str, object]:
        return {"optimizer_trace_rows": tuple(self._trace_rows)}


class StatelessRlPolicyBackend(_BaseBackend):
    backend_id = "rl_policy"

    def __init__(self) -> None:
        super().__init__()
        self._policy_mean = (1.0, 1.0, 1.0, 1.0, 1.0)
        self._exploration = 0.22
        self._update_rate = 0.18
        self._trace_rows: list[dict[str, object]] = []

    def initialize(self, objective: TrajectoryExplorationObjective, seed: int) -> None:
        super().initialize(objective, seed)
        seed_action = _seed_action_for_objective(self._rng, objective, seed + 17)
        self._policy_mean = _vector_from_action(seed_action)
        self._exploration = 0.24
        self._trace_rows = []

    def propose_batch(self, batch_size: int) -> tuple[TrajectoryExplorationProposal, ...]:
        assert self._objective is not None
        proposals: list[TrajectoryExplorationProposal] = []
        target_tier = self._objective.target.target_tier or "realistic_v1"
        for _ in range(batch_size):
            vector = tuple(self._rng.gauss(self._policy_mean[index], self._exploration) for index in range(5))
            action = _action_from_vector(self._next_seed(), target_tier, vector)
            proposals.append(self._proposal(action, parent_id="policy_mean", metadata={"search_method": "rl_policy"}))
        self._iteration += 1
        return tuple(proposals)

    def observe(self, evaluations: tuple[TrajectoryExplorationEvaluation, ...]) -> None:
        if not evaluations:
            return
        ranked = sorted(evaluations, key=lambda row: row.total_utility, reverse=True)
        best = ranked[0]
        best_vector = (
            float(best.diagnostics["duration_scale"]),
            float(best.diagnostics["measurement_scale"]),
            float(best.diagnostics["irregularity_scale"]),
            float(best.diagnostics["outlier_scale"]),
            float(best.diagnostics["step_scale"]),
        )
        self._policy_mean = tuple(
            (1.0 - self._update_rate) * self._policy_mean[index] + self._update_rate * best_vector[index] for index in range(5)
        )
        self._exploration = max(0.08, self._exploration * 0.94)
        self._trace_rows.append(
            {
                "iteration": self._iteration,
                "best_total_utility": best.total_utility,
                "policy_duration_scale": self._policy_mean[0],
                "policy_measurement_scale": self._policy_mean[1],
                "policy_irregularity_scale": self._policy_mean[2],
                "policy_outlier_scale": self._policy_mean[3],
                "policy_step_scale": self._policy_mean[4],
                "sequential_mode": False,
            }
        )

    def state_summary(self) -> dict[str, object]:
        summary = super().state_summary()
        summary.update({"action_mode": "parameter_proposal", "sequential_mode": False})
        return summary

    def diagnostics(self) -> dict[str, object]:
        return {"policy_trace_rows": tuple(self._trace_rows)}
