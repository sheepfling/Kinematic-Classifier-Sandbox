from __future__ import annotations

from kinematic_classifier_sandbox.corpus import real_world
from kinematic_classifier_sandbox.corpus.real_world import adapters
from kinematic_classifier_sandbox.corpus.real_world.adapters.tgsim import (
    TgsimFoggyBottomAdapter,
    load_tgsim_foggy_bottom_csv,
)


def test_real_world_package_initializers_are_passive() -> None:
    assert real_world.__all__ == []
    assert adapters.__all__ == []
    assert not hasattr(real_world, "NormalizedTrack")
    assert not hasattr(adapters, "TgsimFoggyBottomAdapter")
####


def test_concrete_tgsim_entrypoints_are_available() -> None:
    assert callable(TgsimFoggyBottomAdapter)
    assert callable(load_tgsim_foggy_bottom_csv)
####
