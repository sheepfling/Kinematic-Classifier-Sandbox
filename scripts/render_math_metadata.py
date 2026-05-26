from __future__ import annotations

import runpy
from pathlib import Path


def main() -> int:
    script_path = Path(__file__).resolve().parent / "render" / "render_math_metadata.py"
    runpy.run_path(str(script_path), run_name="__main__")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
