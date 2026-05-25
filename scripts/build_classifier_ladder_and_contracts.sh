#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DOC="$ROOT/docs/surveys/classifier_ladder_and_contracts.tex"
DOC_DIR="$(dirname "$DOC")"
DOC_NAME="$(basename "$DOC")"
OUT_DIR="$ROOT/artifacts/texbuild/classifier_ladder_and_contracts"
PDF_OUT="$ROOT/artifacts/classifier_ladder_and_contracts.pdf"
MD_OUT="$ROOT/artifacts/classifier_ladder_and_contracts.md"

mkdir -p "$OUT_DIR"
mkdir -p "$OUT_DIR/transition_matrix_accumulator_v1"

export TEXMFVAR="${TEXMFVAR:-$OUT_DIR/texmf-var}"
export TEXMFCONFIG="${TEXMFCONFIG:-$OUT_DIR/texmf-config}"
export TEXMFHOME="${TEXMFHOME:-$OUT_DIR/texmf-home}"

cp "$ROOT/docs/surveys/classifier_ladder_and_contracts.md" "$MD_OUT"
cp "$ROOT/artifacts/transition_matrix_accumulator_v1/transition_matrix_diagnostics.png" "$OUT_DIR/transition_matrix_accumulator_v1/"

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

cp "$OUT_DIR/classifier_ladder_and_contracts.pdf" "$PDF_OUT"
printf '%s\n' "$PDF_OUT"
