from __future__ import annotations

from _bootstrap import repo_root


def main() -> int:
    root = repo_root()
    print(f"repo root: {root}")
    print("next steps:")
    print("1. Review docs/surveys/kinematic_method_landscape.md.")
    print("2. Run python3 scripts/all.py.")
    print("3. Run python3 scripts/export_artifacts.py.")
    print("4. Pick the first benchmark family before adding training code.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
