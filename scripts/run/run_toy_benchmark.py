from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
existing_pythonpath = os.environ.get("PYTHONPATH")
os.environ["PYTHONPATH"] = (
    str(SRC) if not existing_pythonpath else f"{SRC}{os.pathsep}{existing_pythonpath}"
)
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from kinematic_classifier_sandbox.witnesses.toy_1d.core import toy_witness_surface


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the 1D toy Bayesian benchmark.")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--steps", type=int, default=32)
    parser.add_argument("--tracks-per-class", type=int, default=8)
    parser.add_argument("--obs-sigma", type=float, default=0.75)
    args = parser.parse_args(argv)

    surface = toy_witness_surface()
    result = surface.run(
        seed=args.seed,
        steps=args.steps,
        tracks_per_class=args.tracks_per_class,
        obs_sigma=args.obs_sigma,
    )
    print(surface.render_markdown(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
