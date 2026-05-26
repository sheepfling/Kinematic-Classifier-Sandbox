from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Generic, TypeVar

ResultT = TypeVar("ResultT")
ArtifactsT = TypeVar("ArtifactsT")


@dataclass(frozen=True, slots=True, kw_only=True)
class WitnessSurface(Generic[ResultT, ArtifactsT]):
    study_id: str
    run: Callable[..., ResultT]
    write_artifacts: Callable[[str | Path], ArtifactsT]
    describe_artifacts: Callable[[ArtifactsT], tuple[str, ...]]
    metadata: dict[str, object]


def run_surface(surface: WitnessSurface[ResultT, ArtifactsT], **run_kwargs: Any) -> ResultT:
    return surface.run(**run_kwargs)


def write_surface_artifacts(surface: WitnessSurface[ResultT, ArtifactsT], output_dir: str | Path) -> ArtifactsT:
    return surface.write_artifacts(output_dir)
