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
    parser = argparse.ArgumentParser(prog="python3 scripts/export_team_packet.py")
    parser.add_argument("--output-dir", default="artifacts", help="Directory where packet outputs should be written.")
    parser.add_argument("--refresh", action="store_true", help="Refresh source artifacts before exporting the packet.")
    parser.add_argument("--zip", action="store_true", help="Also write the optional team packet zip.")
    args = parser.parse_args(argv)
    artifacts = build_showcase_artifacts(args.output_dir, refresh=args.refresh, create_zip=args.zip)
    print(artifacts.team_packet_dir)
    if artifacts.zip_path is not None:
        print(artifacts.zip_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
