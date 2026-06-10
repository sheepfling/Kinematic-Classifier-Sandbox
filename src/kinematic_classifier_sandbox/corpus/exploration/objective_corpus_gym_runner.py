from __future__ import annotations

from ..gym import CorpusGymEnvironment
from ..objectives import CorpusObjectiveSpec, default_corpus_objectives
from .candidate_generation_core import generate_candidates_from_objectives
from .objective_corpus_gym_runner_types import ObjectiveCorpusGymRecord
from .objective_corpus_gym_runner_utils import (
    _episode_to_execution,
    candidate_to_corpus_gym_action,
    objective_to_corpus_gym_target,
)


def execute_objective_candidates_via_corpus_gym(
    objectives: tuple[CorpusObjectiveSpec, ...] | None = None,
) -> tuple[ObjectiveCorpusGymRecord, ...]:
    objective_list = objectives or default_corpus_objectives()
    objective_lookup = {objective.objective_id: objective for objective in objective_list}
    target_lookup = {objective.objective_id: objective_to_corpus_gym_target(objective) for objective in objective_list}
    environment = CorpusGymEnvironment()
    records: list[ObjectiveCorpusGymRecord] = []
    for candidate in generate_candidates_from_objectives(objective_list):
        objective_id = str(candidate.provenance.get("objective_id", ""))
        objective = objective_lookup[objective_id]
        target = target_lookup[objective_id]
        action = candidate_to_corpus_gym_action(candidate)
        environment.reset(target)
        episode = environment.simulate(action)
        execution = _episode_to_execution(candidate, episode)
        records.append(
            ObjectiveCorpusGymRecord(
                objective=objective,
                candidate=candidate,
                target=target,
                action=action,
                execution=execution,
                reward=episode.reward,
                diagnostics=episode.diagnostics,
            )
        )
    return tuple(records)
