from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from ..utils.runtime import configure_runtime_environment, repo_root

CorrectnessLevel = Literal["smoke", "full", "presentation"]


@dataclass(frozen=True, slots=True)
class CorrectnessStep:
    label: str
    argv: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class CorrectnessPlan:
    level: CorrectnessLevel
    steps: tuple[CorrectnessStep, ...]


CORRECTNESS_LANE_BY_LEVEL: dict[CorrectnessLevel, str] = {
    "smoke": "correctness-smoke",
    "full": "correctness-full",
    "presentation": "correctness-presentation",
}


def _pytest_lane_step(lane: str) -> CorrectnessStep:
    return CorrectnessStep(label=f"pytest lane {lane}", argv=("python3", "scripts/test.py", "--lane", lane))


CORRECTNESS_PRESENTATION_STEPS: tuple[CorrectnessStep, ...] = (
    _pytest_lane_step("correctness-presentation"),
    CorrectnessStep(
        label="presentation hero packet validator",
        argv=(
            "python3",
            "scripts/audit/validate_presentation_hero_packet.py",
            "--packet-dir",
            "artifacts/presentation_hero_charts_v5",
        ),
    ),
    CorrectnessStep(
        label="static admissibility packet validator",
        argv=(
            "python3",
            "-m",
            "kinematic_classifier_sandbox",
            "validate-packet",
            "artifacts/packets/static_admissibility_mvp",
        ),
    ),
    CorrectnessStep(
        label="corpus explorer packet validator",
        argv=(
            "python3",
            "-m",
            "kinematic_classifier_sandbox",
            "validate-packet",
            "artifacts/packets/corpus_explorer_mvp",
            "--profile",
            "corpus_explorer_mvp",
        ),
    ),
)


def build_correctness_plan(level: CorrectnessLevel) -> CorrectnessPlan:
    if level not in CORRECTNESS_LANE_BY_LEVEL:
        raise ValueError(f"unknown correctness level: {level}")
    if level == "presentation":
        steps = (_pytest_lane_step("correctness-full"), *CORRECTNESS_PRESENTATION_STEPS)
    else:
        steps = (_pytest_lane_step(CORRECTNESS_LANE_BY_LEVEL[level]),)
    return CorrectnessPlan(level=level, steps=tuple(steps))


def correctness_summary(level: CorrectnessLevel) -> str:
    plan = build_correctness_plan(level)
    lines = [
        "# Correctness Ladder",
        "",
        f"- level: `{plan.level}`",
        "",
        "## Steps",
        "",
    ]
    for step in plan.steps:
        lines.append(f"- {step.label}: `{' '.join(step.argv)}`")
    lines.extend(
        [
            "",
            "## Ladder",
            "",
            "- L0 schema correctness: inputs and outputs are valid.",
            "- L1 invariant correctness: math cannot silently break.",
            "- L2 toy oracle correctness: tiny witnesses pass exactly.",
            "- L3 statistical regression: stochastic algorithms stay within tolerance.",
            "- L4 claim correctness: charts and docs do not overclaim.",
            "",
            "## Level Semantics",
            "",
            "- smoke: schema and invariant gate.",
            "- full: smoke plus toy oracle and regression coverage.",
            "- presentation: full plus packet and claim-boundary validation.",
        ]
    )
    return "\n".join(lines) + "\n"


def _subprocess_env(root: Path) -> dict[str, str]:
    configure_runtime_environment()
    env = os.environ.copy()
    src_path = str(root / "src")
    current_pythonpath = env.get("PYTHONPATH")
    env["PYTHONPATH"] = src_path if not current_pythonpath else f"{src_path}{os.pathsep}{current_pythonpath}"
    return env


def run_correctness_plan(level: CorrectnessLevel, *, root: Path | None = None) -> int:
    repo = root or repo_root()
    env = _subprocess_env(repo)
    plan = build_correctness_plan(level)
    print(correctness_summary(level), end="")
    for step in plan.steps:
        print(f"$ {' '.join(step.argv)}", flush=True)
        completed = subprocess.run(step.argv, cwd=repo, env=env)
        if completed.returncode != 0:
            return completed.returncode
    return 0
