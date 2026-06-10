from __future__ import annotations

import subprocess
import sys

from _bootstrap import repo_root


def run(command: list[str]) -> int:
    print("$", " ".join(command))
    completed = subprocess.run(command, cwd=repo_root())
    return completed.returncode


def main() -> int:
    command = [sys.executable, "-m", "ruff", "format", "."]
    return run(command)


if __name__ == "__main__":
    raise SystemExit(main())
