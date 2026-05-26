from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
existing_pythonpath = os.environ.get("PYTHONPATH")
os.environ["PYTHONPATH"] = str(SRC) if not existing_pythonpath else f"{SRC}{os.pathsep}{existing_pythonpath}"
os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/kinematic-classifier-sandbox-mpl")
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from kinematic_classifier_sandbox.advanced_filters.evaluation import rbpf_witness_surface


def main() -> int:
    parser = argparse.ArgumentParser(prog="python3 scripts/run/run_rbpf_1d_witness.py")
    parser.add_argument("--output-dir", default="artifacts")
    args = parser.parse_args()
    surface = rbpf_witness_surface()
    artifacts = surface.write_artifacts(args.output_dir)
    for line in surface.describe_artifacts(artifacts):
        print(line)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
