from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
existing_pythonpath = os.environ.get("PYTHONPATH")
os.environ["PYTHONPATH"] = str(SRC) if not existing_pythonpath else f"{SRC}{os.pathsep}{existing_pythonpath}"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from kinematic_classifier_sandbox.showcase.builder import build_showcase_artifacts


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python3 scripts/build_gallery.py")
    parser.add_argument("--output-dir", default="artifacts", help="Directory where showcase outputs should be written.")
    args = parser.parse_args(argv)
    artifacts = build_showcase_artifacts(args.output_dir, refresh=False, create_zip=False)
    print(artifacts.plots_dir)
    print(artifacts.reports_dir / "07_visualization_gallery.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
