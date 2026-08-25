from __future__ import annotations

import os
from pathlib import Path

import pytest

from kinematic_classifier_sandbox.utils.runtime import configure_runtime_environment

configure_runtime_environment()

_PRODUCT2_DIRECTORIES = {
    "advanced_filters",
    "inference",
    "rung_sufficiency",
    "validation",
    "validation_packets",
}
_PRODUCT3_DIRECTORIES = {"corpus", "study_candidate_generation"}
_LAND_TEST_PREFIXES = ("test_tgsim", "test_road_vehicle")
_LANE_MARKERS = {
    "test_cmre_route_tracklets.py": "product4_sea_surface",
    "test_sea_subsurface_common_front.py": "product4_sea_subsurface",
    "test_sea_subsurface_research_fixtures.py": "product4_sea_subsurface",
    "test_sea_subsurface_selected_anchor.py": "product4_sea_subsurface",
    "test_air_common_front.py": "product4_air_atmospheric",
    "test_adsblol_readsb_trace.py": "product4_air_atmospheric",
    "test_land_common_front.py": "product4_land_surface",
    "test_space_orbital_common_front.py": "product4_space_orbital",
    "test_space_near_fixture_adapter.py": "product4_space_near",
    "test_space_orbital_oem.py": "product4_space_orbital",
    "test_space_orbital_sp3.py": "product4_space_orbital",
}


def pytest_configure(config) -> None:
    cache_dir = Path(os.environ["MPLCONFIGDIR"]).parent / "pytest-cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache = getattr(config, "cache", None)
    if cache is not None:
        cache._cachedir = cache_dir


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    root = Path(config.rootpath)
    for item in items:
        relative = Path(item.path).relative_to(root)
        parts = relative.parts
        if len(parts) < 2 or parts[0] != "tests":
            item.add_marker(pytest.mark.cross_product)
            continue

        top_level = parts[1]
        if len(parts) >= 3 and parts[1:3] == ("corpus", "real_world"):
            item.add_marker(pytest.mark.product4)
            filename = relative.name
            if filename == "test_portfolio.py":
                item.add_marker(pytest.mark.product4_cross_domain)
            else:
                marker_name = _LANE_MARKERS.get(filename)
                if marker_name is None and filename.startswith(_LAND_TEST_PREFIXES):
                    marker_name = "product4_land_surface"
                if marker_name is None:
                    marker_name = "product4_common"
                item.add_marker(getattr(pytest.mark, marker_name))
            continue

        if top_level == "static_admissibility":
            item.add_marker(pytest.mark.product1)
        elif top_level in _PRODUCT2_DIRECTORIES:
            item.add_marker(pytest.mark.product2)
        elif top_level in _PRODUCT3_DIRECTORIES:
            item.add_marker(pytest.mark.product3)
        else:
            item.add_marker(pytest.mark.cross_product)
