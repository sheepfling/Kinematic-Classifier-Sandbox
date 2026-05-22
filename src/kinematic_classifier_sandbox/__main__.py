from __future__ import annotations

from .catalog import METHOD_CATALOG


def main() -> int:
    for entry in METHOD_CATALOG:
        print(f"{entry.family}: {entry.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
