from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Generic, TypeVar

from ..witnesses.surface import WitnessSurface, run_surface, write_surface_artifacts


ResultT = TypeVar("ResultT")
ArtifactsT = TypeVar("ArtifactsT")


@dataclass(frozen=True, slots=True)
class AdvancedFilterSurface(WitnessSurface[ResultT, ArtifactsT], Generic[ResultT, ArtifactsT]):
    metadata: dict[str, object]
