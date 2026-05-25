#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DOC="$ROOT/docs/surveys/methodology_evaluation_framework.tex"
DOC_DIR="$(dirname "$DOC")"
DOC_NAME="$(basename "$DOC")"
OUT_DIR="$ROOT/artifacts/texbuild/methodology_evaluation_framework"
PDF_OUT="$ROOT/artifacts/methodology_evaluation_framework.pdf"
MD_OUT="$ROOT/artifacts/methodology_evaluation_framework.md"

mkdir -p "$OUT_DIR"
mkdir -p "$OUT_DIR/prior_sensitivity_v1"
mkdir -p "$OUT_DIR/abstract_inspection_v1"
mkdir -p "$OUT_DIR/feature_analysis_v1"
mkdir -p "$OUT_DIR/corpus_adequacy_audit_v1"

export TEXMFVAR="${TEXMFVAR:-$OUT_DIR/texmf-var}"
export TEXMFCONFIG="${TEXMFCONFIG:-$OUT_DIR/texmf-config}"
export TEXMFHOME="${TEXMFHOME:-$OUT_DIR/texmf-home}"

cp "$ROOT/docs/surveys/methodology_evaluation_framework.md" "$MD_OUT"
cp "$ROOT/artifacts/prior_sensitivity_v1/trajectory_prior_fragility_overview.png" "$OUT_DIR/prior_sensitivity_v1/"
cp "$ROOT/artifacts/prior_sensitivity_v1/pairwise_flip_threshold_heatmap.png" "$OUT_DIR/prior_sensitivity_v1/"
cp "$ROOT/artifacts/abstract_inspection_v1/feature_set_inspection_summary.png" "$OUT_DIR/abstract_inspection_v1/"
cp "$ROOT/artifacts/abstract_inspection_v1/hardest_class_pairs.png" "$OUT_DIR/abstract_inspection_v1/"
cp "$ROOT/artifacts/feature_analysis_v1/class_confusability_heatmap.png" "$OUT_DIR/feature_analysis_v1/"
cp "$ROOT/artifacts/feature_analysis_v1/pairwise_overlap_heatmap.png" "$OUT_DIR/feature_analysis_v1/"
cp "$ROOT/artifacts/corpus_adequacy_audit_v1/class_pair_coverage_heatmap.png" "$OUT_DIR/corpus_adequacy_audit_v1/"
cp "$ROOT/artifacts/corpus_adequacy_audit_v1/covariate_leakage_audit.png" "$OUT_DIR/corpus_adequacy_audit_v1/"

(
  cd "$DOC_DIR"
  latexmk \
    -pdf \
    -interaction=nonstopmode \
    -halt-on-error \
    -file-line-error \
    -outdir="$OUT_DIR" \
    "$DOC_NAME"
)

cp "$OUT_DIR/methodology_evaluation_framework.pdf" "$PDF_OUT"
printf '%s\n' "$PDF_OUT"
