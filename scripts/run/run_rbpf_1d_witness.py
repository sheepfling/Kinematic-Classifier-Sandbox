from __future__ import annotations

import argparse
from pathlib import Path
from _bootstrap import bootstrap_repo

ROOT = bootstrap_repo(configure_runtime=True)



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
