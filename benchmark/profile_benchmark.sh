#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# You can override NSYS_DELAY to change when capture starts (defaults to 80s to skip startup/warmup).
NSYS_DELAY="${NSYS_DELAY:-80}"
# Override NSYS_DURATION to set how long nsys captures (defaults to 60s).
NSYS_DURATION="${NSYS_DURATION:-60}"
# You can point to a specific nsys binary (defaults to /usr/local/cuda/bin/nsys).
NSYS_BIN="${NSYS_BIN:-/usr/local/cuda/bin/nsys}"

NSYS_BIN="${NSYS_BIN}" python nsys_profile_runner.py --nsys-delay "${NSYS_DELAY}" --nsys-duration "${NSYS_DURATION}" "$@"
