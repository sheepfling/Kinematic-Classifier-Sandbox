from __future__ import annotations

import argparse
import subprocess
import sys

from _bootstrap import check_environment


def run(command: list[str], *, cwd, env: dict[str, str]) -> int:
    print("$", " ".join(command))
    completed = subprocess.run(
        command,
        cwd=cwd,
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
    root, env = check_environment()
    commands = [
        [sys.executable, "scripts/guard_numpy_imports.py"],
        [sys.executable, "scripts/audit/audit_repo_shape.py"],
        [sys.executable, "scripts/audit/audit_import_simplicity.py", "--write-artifacts"],
        [sys.executable, "scripts/audit/audit_human_operability.py"],
        [sys.executable, "-m", "ruff", "check", *(["--fix"] if args.fix else []), "."],
        [sys.executable, "-m", "ruff", "format", *( [] if args.fix else ["--check"] ), "."],
        [sys.executable, "-m", "pyright"],
    ]
    for command in commands:
        result = run(command, cwd=root, env=env)
        if result != 0:
            return result
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
