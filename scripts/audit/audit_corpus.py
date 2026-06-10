from __future__ import annotations

import argparse
from pathlib import Path
from _bootstrap import bootstrap_repo

ROOT = bootstrap_repo(configure_runtime=True)



from kinematic_classifier_sandbox.corpus.adequacy_audit import write_corpus_adequacy_artifacts
from kinematic_classifier_sandbox.corpus.coverage_report import write_coverage_report_artifacts


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python3 scripts/audit/audit_corpus.py")
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
