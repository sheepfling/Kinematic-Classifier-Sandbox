#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from new_study_workflow_common import ensure_declaration_artifacts, load_study_config, phase_dir

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
import sys

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from kinematic_classifier_sandbox.study_confidence import write_study_confidence_artifacts


def run_phase(study_path: str | Path, output_dir: str | Path) -> Path:
    study = load_study_config(study_path)
    ensure_declaration_artifacts(study_path, output_dir)
    output_path = phase_dir(output_dir, study, "04b_confidence")
    write_study_confidence_artifacts(
        output_path,
        workflow_root=Path(output_dir) / str(study["study_id"]),
        study=study,
    )
    return output_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Score study confidence from workflow artifacts.")
    parser.add_argument("--study", required=True, help="Path to the study YAML.")
    parser.add_argument("--output-dir", default="artifacts", help="Workflow artifact root.")
    args = parser.parse_args()
    path = run_phase(args.study, args.output_dir)
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
