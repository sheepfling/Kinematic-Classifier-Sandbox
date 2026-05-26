from __future__ import annotations

import sys
from pathlib import Path

MIN_VERSION = (3, 12)


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    print(f"repo root: {root}")
    print(f"python: {sys.executable}")
    print(f"version: {sys.version.split()[0]}")
    if sys.version_info < MIN_VERSION:
        required = ".".join(str(part) for part in MIN_VERSION)
        print(f"error: python>={required} is required")
        return 1
    print("environment check: ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
