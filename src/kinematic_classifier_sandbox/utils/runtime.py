from __future__ import annotations

import os
import tempfile
from pathlib import Path

_RUNTIME_ROOT_NAME = "kinematic-classifier-sandbox"


def runtime_root() -> Path:
    override = os.environ.get("KINEMATIC_CLASSIFIER_RUNTIME_ROOT")
    if override:
        return Path(override)
    return Path(tempfile.gettempdir()) / _RUNTIME_ROOT_NAME


def pycache_prefix() -> Path:
    return runtime_root() / "pycache"


def mpl_config_dir() -> Path:
    return runtime_root() / "mplconfig"


def configure_runtime_environment() -> None:
    pycache_dir = pycache_prefix()
    mpl_dir = mpl_config_dir()
    pycache_dir.mkdir(parents=True, exist_ok=True)
    mpl_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("PYTHONPYCACHEPREFIX", str(pycache_dir))
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_dir))


def configure_matplotlib_environment() -> None:
    mpl_dir = mpl_config_dir()
    mpl_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_dir))


def repo_root() -> Path:
    return Path(__file__).resolve().parents[3]
