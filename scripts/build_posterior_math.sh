#!/usr/bin/env bash
set -euo pipefail
# Delegates to the canonical build script that drives latexmk.
exec "$(dirname "$0")/build/build_posterior_math.sh" "$@"
