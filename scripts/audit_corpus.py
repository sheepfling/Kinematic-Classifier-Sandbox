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

from kinematic_classifier_sandbox import write_corpus_adequacy_artifacts, write_coverage_report_artifacts


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python3 scripts/audit_corpus.py")
    parser.add_argument("--output-dir", default="artifacts", help="Directory where audit outputs should be written.")
    parser.add_argument("--seed", type=int, default=7, help="Random seed for corpus generation.")
    parser.add_argument("--trajectories-per-class", type=int, default=5, help="Trajectories per class per tier.")
    args = parser.parse_args(argv)
    adequacy = write_corpus_adequacy_artifacts(args.output_dir, seed=args.seed, trajectories_per_class=args.trajectories_per_class)
    coverage = write_coverage_report_artifacts(args.output_dir, seed=args.seed, trajectories_per_class=args.trajectories_per_class)
    print(adequacy.run_dir)
    print(adequacy.report_path)
    print(coverage.run_dir)
    print(coverage.report_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
