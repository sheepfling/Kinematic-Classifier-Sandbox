from __future__ import annotations

import argparse
from pathlib import Path
from _bootstrap import bootstrap_repo

ROOT = bootstrap_repo(configure_runtime=True)



from kinematic_classifier_sandbox.corpus.policy_sweep import write_corpus_policy_tuning_artifacts


def main() -> int:
    parser = argparse.ArgumentParser(prog="python3 scripts/render/render_corpus_policy_tuning.py")
    parser.add_argument("--output-dir", default="artifacts")
    args = parser.parse_args()
    artifacts = write_corpus_policy_tuning_artifacts(args.output_dir)
    print(artifacts.run_dir)
    print(artifacts.report_path)
    print(artifacts.recommended_policy_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
