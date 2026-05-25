#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

python3 "$ROOT/scripts/render_methodology_doc_coverage.py"
python3 "$ROOT/scripts/render_math_metadata.py"
python3 "$ROOT/scripts/render_formal_math_registry.py"
python3 "$ROOT/scripts/render_formal_math_visual_registry.py"
bash "$ROOT/scripts/build_posterior_math.sh"
bash "$ROOT/scripts/build_methodology_evaluation_framework.sh"
bash "$ROOT/scripts/build_classifier_ladder_and_contracts.sh"
bash "$ROOT/scripts/build_corpus_generation_and_search.sh"
bash "$ROOT/scripts/build_dimensional_lift_and_advanced_filter_gates.sh"
bash "$ROOT/scripts/build_methodology_latex.sh"
bash "$ROOT/scripts/build_methodology_compendium.sh"

printf '%s\n' "$ROOT/artifacts/posterior_update_math.pdf"
printf '%s\n' "$ROOT/artifacts/methodology_evaluation_framework.pdf"
printf '%s\n' "$ROOT/artifacts/classifier_ladder_and_contracts.pdf"
printf '%s\n' "$ROOT/artifacts/corpus_generation_and_search.pdf"
printf '%s\n' "$ROOT/artifacts/dimensional_lift_and_advanced_filter_gates.pdf"
printf '%s\n' "$ROOT/artifacts/latex/kinematic_classifier_methodology.pdf"
printf '%s\n' "$ROOT/artifacts/methodology_compendium.md"
printf '%s\n' "$ROOT/artifacts/latex/equation_registry.md"
printf '%s\n' "$ROOT/artifacts/formal_math_registry_v1/formal_math_registry_report.md"
printf '%s\n' "$ROOT/artifacts/formal_math_visual_registry_v1/formal_math_visual_registry_report.md"
