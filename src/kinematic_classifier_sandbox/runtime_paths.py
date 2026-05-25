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
    os.environ.setdefault("PYTHONPYCACHEPREFIX", str(pycache_prefix()))
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_config_dir()))
