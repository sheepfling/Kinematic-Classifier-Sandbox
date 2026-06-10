from __future__ import annotations

from _bootstrap import repo_root


def main() -> int:
    root = repo_root()
    print(f"repo root: {root}")
    print("import contract:")
    print("- install with python3 -m pip install -e '.[dev]' or run scripts with PYTHONPATH=src")
    print("- do not add root compatibility wrappers, broad __init__ reexports, or sys.path mutation")
    print("- run PYTHONPATH=src python3 scripts/audit/audit_import_simplicity.py --strict after import changes")
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
