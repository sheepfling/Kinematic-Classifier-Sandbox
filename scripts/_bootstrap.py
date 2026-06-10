from __future__ import annotations

import os
from pathlib import Path
import sys


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def bootstrap_repo(*, configure_runtime: bool = False, configure_matplotlib: bool = False) -> Path:
    root = repo_root()
    src_path = str(root / "src")
    if src_path not in sys.path:
        sys.path.insert(0, src_path)
    if configure_runtime:
        from kinematic_classifier_sandbox.utils.runtime import configure_runtime_environment

        configure_runtime_environment()
    elif configure_matplotlib:
        from kinematic_classifier_sandbox.utils.runtime import configure_matplotlib_environment

        configure_matplotlib_environment()
    return root


def check_environment() -> tuple[Path, dict[str, str]]:
    root = bootstrap_repo()
    from kinematic_classifier_sandbox.utils.runtime import configure_runtime_environment, runtime_root

    configure_runtime_environment()
    tool_cache_dir = runtime_root() / "tool_cache"
    ruff_cache_dir = tool_cache_dir / "ruff"
    ruff_cache_dir.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["PYTHONPYCACHEPREFIX"] = os.environ["PYTHONPYCACHEPREFIX"]
    env["MPLCONFIGDIR"] = os.environ["MPLCONFIGDIR"]
    env["RUFF_CACHE_DIR"] = str(ruff_cache_dir)
    src_path = str(root / "src")
    env["PYTHONPATH"] = src_path if not env.get("PYTHONPATH") else f"{src_path}{os.pathsep}{env['PYTHONPATH']}"
    return root, env
