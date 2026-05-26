from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def run(command: list[str]) -> int:
    print("$", " ".join(command))
    completed = subprocess.run(command, cwd=Path(__file__).resolve().parents[1])
    return completed.returncode


def main() -> int:
    command = [sys.executable, "-m", "ruff", "format", "."]
    return run(command)


if __name__ == "__main__":
    raise SystemExit(main())
