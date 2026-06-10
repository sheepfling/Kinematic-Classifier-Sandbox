from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from _bootstrap import check_environment


def run(command: list[str], *, cwd: Path, env: dict[str, str]) -> int:
    print(f"+ {' '.join(command)}", flush=True)
    result = subprocess.run(command, cwd=cwd, env=env)
    return result.returncode


def main() -> int:
    root, env = check_environment()

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
