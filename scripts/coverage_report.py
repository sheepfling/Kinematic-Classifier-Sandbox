from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
existing_pythonpath = os.environ.get("PYTHONPATH")
os.environ["PYTHONPATH"] = (
    str(SRC) if not existing_pythonpath else f"{SRC}{os.pathsep}{existing_pythonpath}"
)
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from kinematic_classifier_sandbox.__main__ import main


if __name__ == "__main__":
    raise SystemExit(main(["coverage-report", *sys.argv[1:]]))
