#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DOC="$ROOT/docs/surveys/corpus_generation_and_search.tex"
DOC_DIR="$(dirname "$DOC")"
DOC_NAME="$(basename "$DOC")"
OUT_DIR="$ROOT/artifacts/texbuild/corpus_generation_and_search"
PDF_OUT="$ROOT/artifacts/corpus_generation_and_search.pdf"
MD_OUT="$ROOT/artifacts/corpus_generation_and_search.md"

mkdir -p "$OUT_DIR"

export TEXMFVAR="${TEXMFVAR:-$OUT_DIR/texmf-var}"
export TEXMFCONFIG="${TEXMFCONFIG:-$OUT_DIR/texmf-config}"
export TEXMFHOME="${TEXMFHOME:-$OUT_DIR/texmf-home}"

cp "$ROOT/docs/surveys/corpus_generation_and_search.md" "$MD_OUT"

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

cp "$OUT_DIR/corpus_generation_and_search.pdf" "$PDF_OUT"
printf '%s\n' "$PDF_OUT"
