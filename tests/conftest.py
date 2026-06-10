from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from kinematic_classifier_sandbox.utils.runtime import configure_runtime_environment

configure_runtime_environment()


def pytest_configure(config) -> None:
    cache_dir = Path(os.environ["MPLCONFIGDIR"]).parent / "pytest-cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache = getattr(config, "cache", None)
    if cache is not None:
        cache._cachedir = cache_dir
