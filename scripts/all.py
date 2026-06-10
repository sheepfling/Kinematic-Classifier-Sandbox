from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


def run(command: list[str], *, cwd: Path, env: dict[str, str]) -> int:
    print(f"+ {' '.join(command)}", flush=True)
    result = subprocess.run(command, cwd=cwd, env=env)
    return result.returncode


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    env = os.environ.copy()
    src_path = str(root / "src")
    env["PYTHONPATH"] = (
        src_path if not env.get("PYTHONPATH") else f"{src_path}{os.pathsep}{env['PYTHONPATH']}"
    )
    if src_path not in sys.path:
        sys.path.insert(0, src_path)

    from kinematic_classifier_sandbox.utils.runtime import configure_runtime_environment

    configure_runtime_environment()
    env.update(
        {
            "PYTHONPYCACHEPREFIX": os.environ["PYTHONPYCACHEPREFIX"],
            "MPLCONFIGDIR": os.environ["MPLCONFIGDIR"],
        }
    )

    commands = [
        [sys.executable, "scripts/check_env.py"],
        [sys.executable, "-m", "unittest", "discover", "-s", "tests", "-p", "test_*.py"],
        [sys.executable, "scripts/syntax_check.py"],
    ]
    for command in commands:
        code = run(command, cwd=root, env=env)
        if code != 0:
            return code
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
