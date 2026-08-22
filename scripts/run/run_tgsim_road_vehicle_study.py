from __future__ import annotations

import argparse
from pathlib import Path

from kinematic_classifier_sandbox.corpus.real_world.road_vehicle_study_runner import (
    run_tgsim_road_vehicle_study,
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prepare the TGSIM passenger-car-vs-truck real-world study.",
    )
    parser.add_argument("input_csv", type=Path)
    parser.add_argument("output_dir", type=Path)
    return parser
####


def main() -> int:
    args = _build_parser().parse_args()
    run = run_tgsim_road_vehicle_study(
        args.input_csv,
        args.output_dir,
    )
    print(run.artifacts.report_path)
    return 0
####


if __name__ == "__main__":
    raise SystemExit(main())
