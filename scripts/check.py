from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


def run(command: list[str], *, env: dict[str, str]) -> int:
    print("$", " ".join(command))
    completed = subprocess.run(
        command,
        cwd=Path(__file__).resolve().parents[1],
        env=env,
    )
    return completed.returncode


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--fix",
        action="store_true",
        help="apply available Ruff fixes before running checks",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    src_path = str(root / "src")
    if src_path not in sys.path:
        sys.path.insert(0, src_path)

    from kinematic_classifier_sandbox.utils.runtime import configure_runtime_environment, runtime_root

    configure_runtime_environment()
    tool_cache_dir = runtime_root() / "tool_cache"
    ruff_cache_dir = tool_cache_dir / "ruff"
    ruff_cache_dir.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["PYTHONPYCACHEPREFIX"] = os.environ["PYTHONPYCACHEPREFIX"]
    env["MPLCONFIGDIR"] = os.environ["MPLCONFIGDIR"]
    env["RUFF_CACHE_DIR"] = str(ruff_cache_dir)
    env["PYTHONPATH"] = (
        src_path if not env.get("PYTHONPATH") else f"{src_path}{os.pathsep}{env['PYTHONPATH']}"
    )
    commands = [
        [sys.executable, "scripts/guard_numpy_imports.py"],
        [sys.executable, "scripts/audit/audit_repo_shape.py"],
        [sys.executable, "scripts/audit/audit_human_operability.py"],
        [sys.executable, "-m", "ruff", "check", *(["--fix"] if args.fix else []), "."],
        [sys.executable, "-m", "ruff", "format", *( [] if args.fix else ["--check"] ), "."],
        [sys.executable, "-m", "pyright"],
    ]
    for command in commands:
        result = run(command, env=env)
        if result != 0:
            return result
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
