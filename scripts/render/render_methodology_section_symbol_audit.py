#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
os.environ.setdefault("PYTHONPYCACHEPREFIX", str(Path(tempfile.gettempdir()) / "kinematic-classifier-sandbox-pycache"))
os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "kinematic-classifier-sandbox-mpl"))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from kinematic_classifier_sandbox.methodology.latex import write_methodology_section_symbol_audit_artifacts


def main() -> int:
    parser = argparse.ArgumentParser(description="Render the methodology section symbol audit artifact bundle.")
    parser.add_argument(
        "--output-dir",
        default="artifacts",
        help="Directory where the methodology section symbol audit bundle should be written.",
    )
    args = parser.parse_args()
    artifacts = write_methodology_section_symbol_audit_artifacts(Path(args.output_dir))
    print(artifacts.run_dir)
    print(artifacts.report_path)
    print(artifacts.summary_path)
    print(artifacts.rows_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
