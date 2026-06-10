from __future__ import annotations

from _bootstrap import repo_root


def main() -> int:
    root = repo_root()
    print(f"repo root: {root}")
    print("development loop:")
    print("- run python3 scripts/check.py")
    print("- run python3 scripts/lint.py")
    print("- run python3 scripts/format.py")
    print("- update the survey and method catalog")
    print("- run python3 scripts/all.py")
    print("- run python3 scripts/export_artifacts.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
