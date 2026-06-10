from __future__ import annotations

from _bootstrap import repo_root


def main() -> int:
    root = repo_root()
    paths = sorted((root / "src").rglob("*.py"))
    paths.extend(sorted((root / "tests").rglob("*.py")))
    paths.extend(sorted((root / "scripts").rglob("*.py")))

    for path in paths:
        source = path.read_text(encoding="utf-8")
        compile(source, str(path.relative_to(root)), "exec")
        print(path.relative_to(root))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
