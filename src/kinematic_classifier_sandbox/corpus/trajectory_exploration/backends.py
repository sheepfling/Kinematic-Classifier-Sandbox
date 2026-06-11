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
_METADATA_DEFAULTS: dict[str, object] = {
    "search_method": "",
    "map_strategy": "",
    "archive_size": 0,
    "acquisition_mode": "",
    "acquisition": 0.0,
    "predicted_mean": 0.0,
    "predicted_uncertainty": 0.0,
    "predicted_novelty": 0.0,
    "predicted_boundary": 0.0,
    "predicted_witness": 0.0,
}


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
        self._selection_policy = "utility_frontier"
        self._search_regime = "default"
        self._prefer_novelty = False

    def initialize(self, objective: TrajectoryExplorationObjective, seed: int) -> None:
        self._objective = objective
        self._rng = random.Random(seed)
        self._proposal_counter = 0
        self._iteration = 0
        self._selection_policy = str(objective.backend_constraints.get("selection_policy", "utility_frontier"))
        self._search_regime = str(objective.backend_constraints.get("search_regime", "default"))
        self._prefer_novelty = bool(objective.backend_constraints.get("prefer_novelty", False))

    def _dimension_bias(self) -> tuple[float, float, float, float, float]:
        if self._selection_policy == "coverage_first":
            return (1.00, 0.95, 1.20, 1.10, 0.95)
        if self._selection_policy == "adaptive_continuous":
            return (0.95, 1.10, 1.05, 1.00, 0.95)
        if self._selection_policy == "adversarial_witness":
            return (0.90, 1.20, 1.15, 1.20, 0.90)
        return (1.0, 1.0, 1.0, 1.0, 1.0)

    def _apply_dimension_bias(
        self,
        values: tuple[float, float, float, float, float],
    ) -> tuple[float, float, float, float, float]:
        bias = self._dimension_bias()
        return tuple(values[index] * bias[index] for index in range(5))

    def _next_seed(self) -> int:
        self._proposal_counter += 1
        return self._rng.randint(1, 10_000_000)

    def _proposal(self, action: CorpusGymAction, *, parent_id: str | None = None, metadata: dict[str, object] | None = None) -> TrajectoryExplorationProposal:
        payload = dict(_METADATA_DEFAULTS)
        payload.update(metadata or {})
        return TrajectoryExplorationProposal(
            proposal_id=f"{self.backend_id}_p{self._proposal_counter}",
            backend_id=self.backend_id,
            iteration=self._iteration,
            candidate_index=self._proposal_counter - 1,
            action=action,
            parent_id=parent_id,
            metadata=payload,
        )

    def state_summary(self) -> dict[str, object]:
        return {
            "backend_id": self.backend_id,
            "iteration": self._iteration,
            "selection_policy": self._selection_policy,
            "search_regime": self._search_regime,
            "prefer_novelty": self._prefer_novelty,
        }

    def diagnostics(self) -> dict[str, object]:
        return {}


class HeuristicSearchBackend(_BaseBackend):
    backend_id = "heuristic_search"

    def propose_batch(self, batch_size: int) -> tuple[TrajectoryExplorationProposal, ...]:
        assert self._objective is not None
        target = self._objective.target
        proposals: list[TrajectoryExplorationProposal] = []
        for action in _doe_actions(target, base_seed=self._next_seed()):
            base_vector = _vector_from_action(action)
            biased_action = _action_from_vector(
                action.seed,
                action.tier_name,
                self._apply_dimension_bias(base_vector),
            )
            proposals.append(self._proposal(biased_action, metadata={"search_method": "doe_grid"}))
        while len(proposals) < batch_size:
            action = _random_action(self._rng, target, seed=self._next_seed())
            biased_action = _action_from_vector(
                action.seed,
                action.tier_name,
                self._apply_dimension_bias(_vector_from_action(action)),
            )
            proposals.append(self._proposal(biased_action, metadata={"search_method": "random"}))
        self._iteration += 1
        return tuple(proposals[:batch_size])

    def observe(self, evaluations: tuple[TrajectoryExplorationEvaluation, ...]) -> None:
        return None


class LatinHypercubeBackend(_BaseBackend):
    backend_id = "latin_hypercube"

    def __init__(self) -> None:
        super().__init__()
        self._target_tier = "realistic_v1"
        self._trace_rows: list[dict[str, object]] = []

    def initialize(self, objective: TrajectoryExplorationObjective, seed: int) -> None:
        super().initialize(objective, seed)
        self._target_tier = objective.target.target_tier or "realistic_v1"
        self._trace_rows = []

    def _latin_hypercube_vector(self, slot: int, batch_size: int) -> tuple[float, float, float, float, float]:
        vector: list[float] = []
        bias = self._dimension_bias()
        for index, (low, high) in enumerate(zip(_LOW, _HIGH)):
            span = high - low
            # Offset each dimension deterministically so one batch covers the box more evenly.
            offset = 3 if bias[index] > 1.1 else 2 if bias[index] > 1.0 else 1
            shifted_slot = (slot + index * offset + self._iteration) % batch_size
            lower = shifted_slot / batch_size
            upper = (shifted_slot + 1) / batch_size
            unit = self._rng.uniform(lower, upper)
            vector.append(low + unit * span)
        return self._apply_dimension_bias(tuple(vector))  # type: ignore[return-value]

    def propose_batch(self, batch_size: int) -> tuple[TrajectoryExplorationProposal, ...]:
        proposals: list[TrajectoryExplorationProposal] = []
        for slot in range(batch_size):
            vector = self._latin_hypercube_vector(slot, batch_size)
            action = _action_from_vector(self._next_seed(), self._target_tier, vector)
            proposals.append(
                self._proposal(
                    action,
                    parent_id="lhs_batch",
                    metadata={"search_method": "latin_hypercube"},
                )
            )
        self._iteration += 1
        return tuple(proposals)

    def observe(self, evaluations: tuple[TrajectoryExplorationEvaluation, ...]) -> None:
        if not evaluations:
            return
        ranked = sorted(evaluations, key=lambda row: row.total_utility, reverse=True)
        self._trace_rows.append(
            {
                "iteration": self._iteration,
                "batch_size": len(evaluations),
                "best_total_utility": ranked[0].total_utility,
                "mean_total_utility": sum(float(row.total_utility) for row in evaluations) / len(evaluations),
            }
        )

    def diagnostics(self) -> dict[str, object]:
        return {"lhs_trace_rows": tuple(self._trace_rows)}


class MapElitesBackend(_BaseBackend):
    backend_id = "map_elites"

    def __init__(self) -> None:
        super().__init__()
        self._target_tier = "realistic_v1"
        self._archive: dict[tuple[str, str, str], tuple[tuple[float, float, float, float, float], float]] = {}
        self._trace_rows: list[dict[str, object]] = []
        self._cell_visit_counts: dict[tuple[str, str, str], int] = {}

    def initialize(self, objective: TrajectoryExplorationObjective, seed: int) -> None:
        super().initialize(objective, seed)
        self._target_tier = objective.target.target_tier or "realistic_v1"
        self._archive = {}
        self._trace_rows = []
        self._cell_visit_counts = {}

    def _cell_id(self, vector: tuple[float, float, float, float, float]) -> tuple[str, str, str]:
        duration_bucket = "low" if vector[0] < 0.95 else "mid" if vector[0] < 1.10 else "high"
        irregularity_bucket = "low" if vector[2] < 1.00 else "mid" if vector[2] < 1.30 else "high"
        outlier_bucket = "low" if vector[3] < 1.00 else "mid" if vector[3] < 1.35 else "high"
        return duration_bucket, irregularity_bucket, outlier_bucket

    def _mutate(
        self,
        parent: tuple[float, float, float, float, float],
    ) -> tuple[float, float, float, float, float]:
        if self._selection_policy == "coverage_first":
            spread = (0.12, 0.10, 0.18, 0.18, 0.08)
        elif self._selection_policy == "adversarial_witness":
            spread = (0.08, 0.14, 0.20, 0.20, 0.07)
        else:
            spread = (0.10, 0.12, 0.14, 0.14, 0.08)
        return tuple(self._rng.gauss(parent[index], spread[index]) for index in range(5))

    def _crossover(
        self,
        first: tuple[float, float, float, float, float],
        second: tuple[float, float, float, float, float],
    ) -> tuple[float, float, float, float, float]:
        mix = []
        for index in range(5):
            alpha = self._rng.uniform(0.30, 0.70)
            mix.append(alpha * first[index] + (1.0 - alpha) * second[index])
        return tuple(mix)

    def _sample_sparse_seed(self) -> tuple[float, float, float, float, float]:
        duration_buckets = ("low", "mid", "high")
        irregularity_buckets = ("low", "mid", "high")
        outlier_buckets = ("low", "mid", "high")
        all_cells = [
            (duration_bucket, irregularity_bucket, outlier_bucket)
            for duration_bucket in duration_buckets
            for irregularity_bucket in irregularity_buckets
            for outlier_bucket in outlier_buckets
        ]
        sparsest_cell = min(
            all_cells,
            key=lambda cell: (
                0 if cell in self._archive else 1,
                self._cell_visit_counts.get(cell, 0),
                self._rng.random(),
            ),
        )
        duration_center = {"low": 0.82, "mid": 1.02, "high": 1.22}[sparsest_cell[0]]
        irregularity_center = {"low": 0.85, "mid": 1.15, "high": 1.55}[sparsest_cell[1]]
        outlier_center = {"low": 0.85, "mid": 1.15, "high": 1.60}[sparsest_cell[2]]
        base = (
            duration_center,
            1.0,
            irregularity_center,
            outlier_center,
            1.0,
        )
        return self._mutate(base)

    def propose_batch(self, batch_size: int) -> tuple[TrajectoryExplorationProposal, ...]:
        proposals: list[TrajectoryExplorationProposal] = []
        archive_values = list(self._archive.values())
        for slot in range(batch_size):
            if not archive_values:
                base_action = _seed_action_for_objective(self._rng, self._objective, self._next_seed()) if self._objective is not None else None
                base_vector = _vector_from_action(base_action) if base_action is not None else (1.0, 1.0, 1.0, 1.0, 1.0)
                vector = self._apply_dimension_bias(self._mutate(base_vector))
                parent_id = "map_seed"
                strategy = "seed_mutation"
            elif self._selection_policy == "coverage_first" and slot < max(1, batch_size // 3):
                vector = self._apply_dimension_bias(self._sample_sparse_seed())
                parent_id = "map_sparse_cell"
                strategy = "sparse_cell_explore"
            elif len(archive_values) >= 2 and (self._prefer_novelty or slot % 3 == 0):
                first_vector, _ = archive_values[self._rng.randrange(len(archive_values))]
                second_vector, _ = archive_values[self._rng.randrange(len(archive_values))]
                crossed = self._crossover(first_vector, second_vector)
                vector = self._apply_dimension_bias(self._mutate(crossed))
                parent_id = "map_crossover"
                strategy = "elite_crossover"
            else:
                parent_vector, _score = archive_values[self._rng.randrange(len(archive_values))]
                vector = self._apply_dimension_bias(self._mutate(parent_vector))
                parent_id = "map_elite"
                strategy = "elite_mutation"
            action = _action_from_vector(self._next_seed(), self._target_tier, vector)
            self._cell_visit_counts[self._cell_id(vector)] = self._cell_visit_counts.get(self._cell_id(vector), 0) + 1
            proposals.append(
                self._proposal(
                    action,
                    parent_id=parent_id,
                    metadata={
                        "search_method": "map_elites",
                        "archive_size": len(self._archive),
                        "map_strategy": strategy,
                    },
                )
            )
        self._iteration += 1
        return tuple(proposals)

    def observe(self, evaluations: tuple[TrajectoryExplorationEvaluation, ...]) -> None:
        if not evaluations:
            return
        added = 0
        replaced = 0
        visited_cells = {self._cell_id((
            float(row.diagnostics["duration_scale"]),
            float(row.diagnostics["measurement_scale"]),
            float(row.diagnostics["irregularity_scale"]),
            float(row.diagnostics["outlier_scale"]),
            float(row.diagnostics["step_scale"]),
        )) for row in evaluations}
        for row in evaluations:
            vector = (
                float(row.diagnostics["duration_scale"]),
                float(row.diagnostics["measurement_scale"]),
                float(row.diagnostics["irregularity_scale"]),
                float(row.diagnostics["outlier_scale"]),
                float(row.diagnostics["step_scale"]),
            )
            cell_id = self._cell_id(vector)
            incumbent = self._archive.get(cell_id)
            if incumbent is None:
                self._archive[cell_id] = (vector, float(row.total_utility))
                added += 1
            elif float(row.total_utility) > incumbent[1]:
                self._archive[cell_id] = (vector, float(row.total_utility))
                replaced += 1
        best = max(float(row.total_utility) for row in evaluations)
        self._trace_rows.append(
            {
                "iteration": self._iteration,
                "archive_size": len(self._archive),
                "visited_cells": len(visited_cells),
                "cells_added": added,
                "cells_replaced": replaced,
                "best_total_utility": best,
            }
        )

    def diagnostics(self) -> dict[str, object]:
        return {"map_elites_trace_rows": tuple(self._trace_rows)}


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
        self._mean = self._apply_dimension_bias(_vector_from_action(seed_action))
        if self._selection_policy == "coverage_first":
            self._std = (0.18, 0.16, 0.22, 0.20, 0.10)
        elif self._selection_policy == "adversarial_witness":
            self._std = (0.14, 0.18, 0.22, 0.22, 0.09)
        else:
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


class CmaEsBackend(_BaseBackend):
    backend_id = "cmaes"

    def __init__(self, *, elite_fraction: float = 0.5, learning_rate: float = 0.32) -> None:
        super().__init__()
        self._mean = (1.0, 1.0, 1.0, 1.0, 1.0)
        self._sigma = (0.16, 0.16, 0.18, 0.18, 0.10)
        self._elite_fraction = elite_fraction
        self._learning_rate = learning_rate
        self._trace_rows: list[dict[str, object]] = []
        self._stagnation_count = 0
        self._restart_count = 0
        self._best_seen_utility = float("-inf")
        self._target_tier = "realistic_v1"

    def initialize(self, objective: TrajectoryExplorationObjective, seed: int) -> None:
        super().initialize(objective, seed)
        seed_action = _seed_action_for_objective(self._rng, objective, seed + 29)
        self._mean = self._apply_dimension_bias(_vector_from_action(seed_action))
        if self._selection_policy == "coverage_first":
            self._sigma = (0.16, 0.14, 0.18, 0.17, 0.09)
        elif self._selection_policy == "adversarial_witness":
            self._sigma = (0.12, 0.17, 0.20, 0.20, 0.08)
        else:
            self._sigma = (0.14, 0.14, 0.16, 0.16, 0.09)
        self._trace_rows = []
        self._stagnation_count = 0
        self._restart_count = 0
        self._best_seen_utility = float("-inf")
        self._target_tier = objective.target.target_tier or "realistic_v1"

    def _restart(self) -> None:
        seed_action = _seed_action_for_objective(self._rng, self._objective, self._next_seed()) if self._objective is not None else None
        base_vector = _vector_from_action(seed_action) if seed_action is not None else (1.0, 1.0, 1.0, 1.0, 1.0)
        self._mean = self._apply_dimension_bias(base_vector)
        if self._selection_policy == "coverage_first":
            self._sigma = (0.18, 0.16, 0.20, 0.20, 0.10)
        elif self._selection_policy == "adversarial_witness":
            self._sigma = (0.14, 0.18, 0.22, 0.22, 0.09)
        else:
            self._sigma = (0.16, 0.16, 0.18, 0.18, 0.10)
        self._restart_count += 1
        self._stagnation_count = 0

    def propose_batch(self, batch_size: int) -> tuple[TrajectoryExplorationProposal, ...]:
        target_tier = self._target_tier
        proposals: list[TrajectoryExplorationProposal] = []
        for _ in range(batch_size):
            vector = tuple(
                self._rng.gauss(self._mean[index], self._sigma[index])
                for index in range(5)
            )
            action = _action_from_vector(self._next_seed(), target_tier, vector)
            proposals.append(
                self._proposal(
                    action,
                    parent_id="cma_mean",
                    metadata={"search_method": "cmaes_diagonal"},
                )
            )
        self._iteration += 1
        return tuple(proposals)

    def observe(self, evaluations: tuple[TrajectoryExplorationEvaluation, ...]) -> None:
        if not evaluations:
            return
        ranked = sorted(evaluations, key=lambda row: row.total_utility, reverse=True)
        best_iteration_utility = float(ranked[0].total_utility)
        elite_count = max(2, int(len(ranked) * self._elite_fraction))
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
        elite_mean = tuple(
            sum(vector[index] for vector in elite_vectors) / elite_count
            for index in range(5)
        )
        elite_sigma = tuple(
            max(
                0.04,
                (
                    sum((vector[index] - elite_mean[index]) ** 2 for vector in elite_vectors)
                    / elite_count
                )
                ** 0.5,
            )
            for index in range(5)
        )
        self._mean = tuple(
            (1.0 - self._learning_rate) * self._mean[index] + self._learning_rate * elite_mean[index]
            for index in range(5)
        )
        self._sigma = tuple(
            max(
                0.035,
                min(
                    0.28,
                    (1.0 - self._learning_rate) * self._sigma[index] + self._learning_rate * elite_sigma[index],
                ),
                )
            for index in range(5)
        )
        if best_iteration_utility > self._best_seen_utility + 0.01:
            self._best_seen_utility = best_iteration_utility
            self._stagnation_count = 0
        else:
            self._stagnation_count += 1
        restarted = False
        if self._stagnation_count >= 2:
            self._restart()
            restarted = True
        self._trace_rows.append(
            {
                "iteration": self._iteration,
                "elite_count": elite_count,
                "best_total_utility": ranked[0].total_utility,
                "global_best_total_utility": self._best_seen_utility,
                "mean_duration_scale": self._mean[0],
                "mean_measurement_scale": self._mean[1],
                "mean_irregularity_scale": self._mean[2],
                "mean_outlier_scale": self._mean[3],
                "mean_step_scale": self._mean[4],
                "sigma_duration_scale": self._sigma[0],
                "sigma_measurement_scale": self._sigma[1],
                "sigma_irregularity_scale": self._sigma[2],
                "sigma_outlier_scale": self._sigma[3],
                "sigma_step_scale": self._sigma[4],
                "stagnation_count": self._stagnation_count,
                "restart_count": self._restart_count,
                "restarted": "yes" if restarted else "no",
            }
        )

    def diagnostics(self) -> dict[str, object]:
        return {"cmaes_trace_rows": tuple(self._trace_rows)}


class BayesianOptimizationBackend(_BaseBackend):
    backend_id = "bayesian_optimization"

    def __init__(
        self,
        *,
        candidate_pool_size: int = 24,
        exploration_weight: float = 0.14,
        novelty_weight: float = 0.08,
    ) -> None:
        super().__init__()
        self._candidate_pool_size = candidate_pool_size
        self._exploration_weight = exploration_weight
        self._novelty_weight = novelty_weight
        self._boundary_weight = 0.0
        self._witness_weight = 0.0
        self._acquisition_mode = "expected_utility"
        self._observed_vectors: list[tuple[float, float, float, float, float]] = []
        self._observed_scores: list[float] = []
        self._observed_boundary_scores: list[float] = []
        self._observed_witness_scores: list[float] = []
        self._trace_rows: list[dict[str, object]] = []
        self._target_tier = "realistic_v1"

    def initialize(self, objective: TrajectoryExplorationObjective, seed: int) -> None:
        super().initialize(objective, seed)
        seed_action = _seed_action_for_objective(self._rng, objective, seed + 43)
        self._observed_vectors = [self._apply_dimension_bias(_vector_from_action(seed_action))]
        self._observed_scores = []
        self._observed_boundary_scores = []
        self._observed_witness_scores = []
        self._trace_rows = []
        self._target_tier = objective.target.target_tier or "realistic_v1"
        if self._selection_policy == "coverage_first":
            self._candidate_pool_size = 28
            self._exploration_weight = 0.18
            self._novelty_weight = 0.04
            self._boundary_weight = 0.02
            self._witness_weight = 0.02
            self._acquisition_mode = "coverage_ucb"
        elif self._selection_policy == "adaptive_continuous":
            self._candidate_pool_size = 28
            self._exploration_weight = 0.12
            self._novelty_weight = 0.05
            self._boundary_weight = 0.18
            self._witness_weight = 0.04
            self._acquisition_mode = "boundary_ucb"
        elif self._selection_policy == "adversarial_witness":
            self._candidate_pool_size = 32
            self._exploration_weight = 0.18
            self._novelty_weight = 0.18
            self._boundary_weight = 0.06
            self._witness_weight = 0.20
            self._acquisition_mode = "witness_novelty_ucb"
        else:
            self._candidate_pool_size = 24
            self._exploration_weight = 0.14
            self._novelty_weight = 0.08
            self._boundary_weight = 0.04
            self._witness_weight = 0.04
            self._acquisition_mode = "expected_utility"

    def _sample_candidate_vector(self) -> tuple[float, float, float, float, float]:
        if not self._observed_scores:
            center = self._observed_vectors[0]
            spread = (0.18, 0.18, 0.20, 0.20, 0.12)
            return tuple(self._rng.gauss(center[index], spread[index]) for index in range(5))
        best_index = max(range(len(self._observed_scores)), key=self._observed_scores.__getitem__)
        center = self._observed_vectors[best_index]
        spread = (0.12, 0.12, 0.14, 0.14, 0.09)
        if self._prefer_novelty:
            spread = tuple(value * 1.20 for value in spread)
        return tuple(self._rng.gauss(center[index], spread[index]) for index in range(5))

    def _predict(self, vector: tuple[float, float, float, float, float]) -> tuple[float, float, float, float, float]:
        if not self._observed_scores:
            novelty = 1.0
            return 0.0, 1.0, novelty, 0.0, 0.0
        weighted_sum = 0.0
        weighted_boundary_sum = 0.0
        weighted_witness_sum = 0.0
        total_weight = 0.0
        min_distance = float("inf")
        for observed_vector, observed_score, boundary_score, witness_score in zip(
            self._observed_vectors,
            self._observed_scores,
            self._observed_boundary_scores,
            self._observed_witness_scores,
        ):
            distance = sum((vector[index] - observed_vector[index]) ** 2 for index in range(5)) ** 0.5
            min_distance = min(min_distance, distance)
            weight = 1.0 / (0.05 + distance)
            weighted_sum += weight * observed_score
            weighted_boundary_sum += weight * boundary_score
            weighted_witness_sum += weight * witness_score
            total_weight += weight
        mean_prediction = weighted_sum / max(total_weight, 1e-9)
        boundary_prediction = weighted_boundary_sum / max(total_weight, 1e-9)
        witness_prediction = weighted_witness_sum / max(total_weight, 1e-9)
        uncertainty = min(0.35, max(0.03, min_distance))
        novelty = min(0.45, max(0.0, min_distance))
        return mean_prediction, uncertainty, novelty, boundary_prediction, witness_prediction

    def propose_batch(self, batch_size: int) -> tuple[TrajectoryExplorationProposal, ...]:
        proposals: list[TrajectoryExplorationProposal] = []
        for _ in range(batch_size):
            ranked_candidates: list[tuple[float, tuple[float, float, float, float, float], float, float, float, float, float]] = []
            for _candidate_index in range(self._candidate_pool_size):
                vector = self._sample_candidate_vector()
                predicted_mean, uncertainty, novelty, boundary_prediction, witness_prediction = self._predict(vector)
                acquisition = (
                    predicted_mean
                    + self._exploration_weight * uncertainty
                    + self._novelty_weight * novelty
                    + self._boundary_weight * boundary_prediction
                    + self._witness_weight * witness_prediction
                )
                ranked_candidates.append(
                    (
                        acquisition,
                        vector,
                        predicted_mean,
                        uncertainty,
                        novelty,
                        boundary_prediction,
                        witness_prediction,
                    )
                )
            ranked_candidates.sort(key=lambda item: item[0], reverse=True)
            acquisition, vector, predicted_mean, uncertainty, novelty, boundary_prediction, witness_prediction = ranked_candidates[0]
            action = _action_from_vector(self._next_seed(), self._target_tier, vector)
            proposals.append(
                self._proposal(
                    action,
                    parent_id="bo_acquisition",
                    metadata={
                        "search_method": "bayesian_optimization",
                        "acquisition_mode": self._acquisition_mode,
                        "acquisition": acquisition,
                        "predicted_mean": predicted_mean,
                        "predicted_uncertainty": uncertainty,
                        "predicted_novelty": novelty,
                        "predicted_boundary": boundary_prediction,
                        "predicted_witness": witness_prediction,
                    },
                )
            )
        self._iteration += 1
        return tuple(proposals)

    def observe(self, evaluations: tuple[TrajectoryExplorationEvaluation, ...]) -> None:
        if not evaluations:
            return
        ranked = sorted(evaluations, key=lambda row: row.total_utility, reverse=True)
        for row in evaluations:
            self._observed_vectors.append(
                (
                    float(row.diagnostics["duration_scale"]),
                    float(row.diagnostics["measurement_scale"]),
                    float(row.diagnostics["irregularity_scale"]),
                    float(row.diagnostics["outlier_scale"]),
                    float(row.diagnostics["step_scale"]),
                )
            )
            self._observed_scores.append(float(row.total_utility))
            self._observed_boundary_scores.append(float(row.boundary_closeness))
            self._observed_witness_scores.append(float(row.confusion_witness_score))
        best_score = max(self._observed_scores)
        self._trace_rows.append(
            {
                "iteration": self._iteration,
                "observations": len(self._observed_scores),
                "best_total_utility": ranked[0].total_utility,
                "global_best_total_utility": best_score,
                "mean_observed_utility": sum(self._observed_scores) / len(self._observed_scores),
                "acquisition_mode": self._acquisition_mode,
            }
        )

    def diagnostics(self) -> dict[str, object]:
        return {"bayesopt_trace_rows": tuple(self._trace_rows)}


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
