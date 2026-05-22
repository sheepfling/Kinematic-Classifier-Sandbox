from __future__ import annotations

from pathlib import Path


def main() -> int:
    root = Path(__file__).resolve().parents[1]
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
