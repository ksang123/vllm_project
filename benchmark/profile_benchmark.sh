#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG="${1:-"$SCRIPT_DIR/bench_config.yaml"}"
shift || true 2>/dev/null || true

cd "$SCRIPT_DIR"
python nsys_profile_runner.py --config "$CONFIG" "$@"
