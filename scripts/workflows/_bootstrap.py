from __future__ import annotations

import importlib.util
from pathlib import Path


_ROOT_BOOTSTRAP_PATH = Path(__file__).resolve().parents[1] / "_bootstrap.py"
_SPEC = importlib.util.spec_from_file_location("_kcs_root_script_bootstrap", _ROOT_BOOTSTRAP_PATH)
if _SPEC is None or _SPEC.loader is None:
    raise RuntimeError(f"could not load root script bootstrap: {_ROOT_BOOTSTRAP_PATH}")
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)

bootstrap_repo = _MODULE.bootstrap_repo
check_environment = _MODULE.check_environment
repo_root = _MODULE.repo_root

__all__ = ["bootstrap_repo", "check_environment", "repo_root"]
