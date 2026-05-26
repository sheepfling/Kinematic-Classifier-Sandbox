from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
existing_pythonpath = os.environ.get("PYTHONPATH")
os.environ["PYTHONPATH"] = str(SRC) if not existing_pythonpath else f"{SRC}{os.pathsep}{existing_pythonpath}"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from kinematic_classifier_sandbox.advanced_filters.evaluation import (
    advanced_filter_comparison_surface,
)


def main() -> int:
    parser = argparse.ArgumentParser(prog="python3 scripts/run/run_advanced_filter_comparison.py")
    parser.add_argument("--output-dir", default="artifacts")
    args = parser.parse_args()
    surface = advanced_filter_comparison_surface()
    artifacts = surface.write_artifacts(args.output_dir)
    for line in surface.describe_artifacts(artifacts):
        print(line)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
