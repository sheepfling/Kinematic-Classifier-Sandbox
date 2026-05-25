#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DOC="$ROOT/docs/surveys/dimensional_lift_and_advanced_filter_gates.tex"
DOC_DIR="$(dirname "$DOC")"
DOC_NAME="$(basename "$DOC")"
OUT_DIR="$ROOT/artifacts/texbuild/dimensional_lift_and_advanced_filter_gates"
PDF_OUT="$ROOT/artifacts/dimensional_lift_and_advanced_filter_gates.pdf"
MD_OUT="$ROOT/artifacts/dimensional_lift_and_advanced_filter_gates.md"

mkdir -p "$OUT_DIR"

export TEXMFVAR="${TEXMFVAR:-$OUT_DIR/texmf-var}"
export TEXMFCONFIG="${TEXMFCONFIG:-$OUT_DIR/texmf-config}"
export TEXMFHOME="${TEXMFHOME:-$OUT_DIR/texmf-home}"

cp "$ROOT/docs/surveys/dimensional_lift_and_advanced_filter_gates.md" "$MD_OUT"

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

cp "$OUT_DIR/dimensional_lift_and_advanced_filter_gates.pdf" "$PDF_OUT"
printf '%s\n' "$PDF_OUT"
