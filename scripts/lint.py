from __future__ import annotations

import argparse
import subprocess
import sys

from _bootstrap import repo_root


def run(command: list[str]) -> int:
    print("$", " ".join(command))
    completed = subprocess.run(command, cwd=repo_root())
    return completed.returncode


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--fix",
        action="store_true",
        help="apply available Ruff fixes before running lint checks",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    commands = [
        [sys.executable, "scripts/guard_numpy_imports.py"],
        [sys.executable, "-m", "ruff", "check", *(["--fix"] if args.fix else []), "."],
        [sys.executable, "-m", "ruff", "format", *([] if args.fix else ["--check"]), "."],
        [sys.executable, "-m", "pyright"],
    ]
    for command in commands:
        result = run(command)
        if result != 0:
            return result
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
