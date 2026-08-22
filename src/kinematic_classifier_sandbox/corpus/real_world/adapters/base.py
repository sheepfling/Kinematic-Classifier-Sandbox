from __future__ import annotations

from pathlib import Path
from typing import Protocol

from ..contracts import DatasetManifest, NormalizedTrack


class RealWorldTrackAdapter(Protocol):
    adapter_id: str
    adapter_version: str

    @property
    def manifest(self) -> DatasetManifest: ...

    def load_tracks(self, path: str | Path) -> tuple[NormalizedTrack, ...]: ...
####


__all__ = ["RealWorldTrackAdapter"]
