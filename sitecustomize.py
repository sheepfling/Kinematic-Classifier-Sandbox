from __future__ import annotations

import os
import tempfile
from pathlib import Path


def _set_default(name: str, value: str) -> None:
    if not os.environ.get(name):
        os.environ[name] = value


_set_default("PYTHONPYCACHEPREFIX", str(Path(tempfile.gettempdir()) / "kinematic-classifier-sandbox-pycache"))
_set_default("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "kinematic-classifier-sandbox-mpl"))
