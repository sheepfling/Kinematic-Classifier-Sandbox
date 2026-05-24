from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
existing_pythonpath = os.environ.get("PYTHONPATH")
os.environ["PYTHONPATH"] = str(SRC) if not existing_pythonpath else f"{SRC}{os.pathsep}{existing_pythonpath}"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from kinematic_classifier_sandbox.showcase_builder import validate_showcase_artifacts


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python3 scripts/validate_artifacts.py")
    parser.add_argument(
        "--showcase-dir",
        default="artifacts/showcase",
        help="Showcase directory to validate.",
    )
    args = parser.parse_args(argv)
    result = validate_showcase_artifacts(args.showcase_dir)
    print(json.dumps(asdict(result), indent=2))
    return 0 if result.overall_status == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
