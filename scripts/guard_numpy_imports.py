from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SEARCH_ROOTS = [ROOT / "src", ROOT / "scripts"]
EXCLUDED_PARTS = {".venv", ".cache", ".mypy_cache", ".pytest_cache", ".ruff_cache", "artifacts", "archive", "build", "cache"}


def iter_python_files() -> list[Path]:
    paths: list[Path] = []
    for search_root in SEARCH_ROOTS:
        if not search_root.exists():
            continue
        for path in search_root.rglob("*.py"):
            if any(part in EXCLUDED_PARTS for part in path.parts):
                continue
            paths.append(path)
    return sorted(paths)


def main() -> int:
    violations: list[str] = []
    for path in iter_python_files():
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            if line.strip() == "import numpy as np":
                violations.append(f"{path}:{line_number}: avoid numpy alias imports; use explicit imports")
    if violations:
        print("numpy alias import guard failed:")
        for violation in violations:
            print(violation)
        return 1
    print("numpy alias import guard: ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
