#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DOC="$ROOT/docs/surveys/posterior_update_math.tex"
DOC_DIR="$(dirname "$DOC")"
DOC_NAME="$(basename "$DOC")"
OUT_DIR="$ROOT/artifacts/texbuild/posterior_update_math"
PDF_OUT="$ROOT/artifacts/posterior_update_math.pdf"

mkdir -p "$OUT_DIR"

export TEXMFVAR="${TEXMFVAR:-$OUT_DIR/texmf-var}"
export TEXMFCONFIG="${TEXMFCONFIG:-$OUT_DIR/texmf-config}"
export TEXMFHOME="${TEXMFHOME:-$OUT_DIR/texmf-home}"

cp "$ROOT/artifacts/posterior_update_math.png" "$OUT_DIR/"
cp "$ROOT/artifacts/probability_primitives.png" "$OUT_DIR/"
cp "$ROOT/artifacts/posterior_numeric_walkthrough.png" "$OUT_DIR/"
cp "$ROOT/artifacts/toy_1d_feature_confusion.png" "$OUT_DIR/"
cp "$ROOT/artifacts/identity_1d_feature_confusion.png" "$OUT_DIR/"

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

cp "$OUT_DIR/posterior_update_math.pdf" "$PDF_OUT"
printf '%s\n' "$PDF_OUT"
