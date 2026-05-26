from __future__ import annotations

from dataclasses import dataclass
from typing import Generic, TypeVar

from ..witnesses.surface import WitnessSurface

ResultT = TypeVar("ResultT")
ArtifactsT = TypeVar("ArtifactsT")


@dataclass(frozen=True, slots=True)
class AdvancedFilterSurface(WitnessSurface[ResultT, ArtifactsT], Generic[ResultT, ArtifactsT]):
    metadata: dict[str, object]
