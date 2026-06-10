from __future__ import annotations

import argparse
from pathlib import Path
from _bootstrap import bootstrap_repo

ROOT = bootstrap_repo(configure_runtime=True)



from kinematic_classifier_sandbox.common_experiment.artifact_io import write_common_experiment_artifacts


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python3 scripts/run_study.py")
    parser.add_argument("config_path", help="Path to the study config YAML.")
    parser.add_argument("--output-dir", default="artifacts", help="Directory where the unified run directory should be written.")
    parser.add_argument("--seed", type=int, default=None, help="Optional override seed.")
    parser.add_argument("--trajectories-per-case", type=int, default=8, help="Executable shared scenario count per class.")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    artifacts = write_common_experiment_artifacts(
        args.output_dir,
        config_path=args.config_path,
        seed=args.seed,
        trajectories_per_case=args.trajectories_per_case,
    )
    print(artifacts.run_dir)
    print(artifacts.report_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
