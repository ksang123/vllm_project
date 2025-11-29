#!/usr/bin/env bash
set -euo pipefail

# Kill running vLLM API server processes.
pids=$(pgrep -f "vllm.entrypoints.openai.api_server" || true)
if [[ -z "${pids}" ]]; then
  echo "No running vLLM server processes found."
  exit 0
fi

echo "Killing vLLM server PIDs: ${pids}"
kill ${pids}
