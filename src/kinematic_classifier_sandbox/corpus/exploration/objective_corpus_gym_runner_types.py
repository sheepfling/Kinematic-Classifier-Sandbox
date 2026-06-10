from __future__ import annotations

from dataclasses import dataclass

from ..gym import CorpusGymAction, CorpusGymReward, CorpusGymTarget
from ..objectives import CorpusObjectiveSpec
from .backend_adapter_proof_types import AdapterExecutionRecord, BackendCandidateSpec


@dataclass(frozen=True, slots=True)
class ObjectiveCorpusGymRecord:
    objective: CorpusObjectiveSpec
    candidate: BackendCandidateSpec
    target: CorpusGymTarget
    action: CorpusGymAction
    execution: AdapterExecutionRecord
    reward: CorpusGymReward
    diagnostics: dict[str, object]
