from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

os.environ.setdefault("PYTHONPYCACHEPREFIX", str(Path(tempfile.gettempdir()) / "kinematic-classifier-sandbox-pycache"))
os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "kinematic-classifier-sandbox-mpl"))

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def pytest_configure(config) -> None:
    cache_dir = Path(tempfile.gettempdir()) / "kinematic-classifier-sandbox-pytest-cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache = getattr(config, "cache", None)
    if cache is not None:
        cache._cachedir = cache_dir
