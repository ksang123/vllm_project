from __future__ import annotations

import argparse
import logging
import os
import shutil
import signal
import subprocess
import sys
import warnings
from pathlib import Path

from config_handler import load_config
from logger import setup_logger
from vllm_runner import (
    build_server_command,
    prepare_server_config,
    stop_vllm_server,
    wait_for_server,
)


def main() -> None:
    warnings.filterwarnings("ignore", category=FutureWarning, message="The pynvml package is deprecated.*")
    parser = argparse.ArgumentParser(description="Profile vLLM server with nsys and run the request sender.")
    parser.add_argument("--config", default="bench_config.yaml", help="Path to bench_config.yaml")
    parser.add_argument("--debug", action="store_true", help="Include generated texts/config in client output")
    parser.add_argument("--nsys-output", default="nsys_vllm_profile", help="Base name for nsys output files")
    parser.add_argument(
        "--nsys-trace",
        default="cuda",
        help="Comma-separated nsys trace domains (defaults to 'cuda' safe-mode).",
    )
    parser.add_argument(
        "--nsys-delay",
        type=float,
        default=5.0,
        help="Seconds to delay nsys capture so profiling starts after server init.",
    )
    parser.add_argument(
        "--nsys-duration",
        type=int,
        default=60,
        help="Duration in seconds to capture with nsys before stopping the server.",
    )
    parser.add_argument(
        "--nsys-bin",
        default=os.environ.get("NSYS_BIN", "nsys"),
        help="Path to nsys binary (defaults to $NSYS_BIN or 'nsys' on PATH)",
    )
    parser.add_argument("--wait-timeout", type=float, default=300.0, help="Seconds to wait for server readiness")
    args = parser.parse_args()

    config = load_config(args.config)
    if not config:
        raise SystemExit("Failed to load configuration.")

    log_file = config.get("log_config", {}).get("log_file", "benchmark.log")
    setup_logger(log_file)
    logger = logging.getLogger("vllm_benchmark")

    nsys_bin = args.nsys_bin
    if not os.path.isabs(nsys_bin):
        resolved = shutil.which(nsys_bin)
    else:
        resolved = nsys_bin if Path(nsys_bin).exists() else None
    if not resolved:
        raise SystemExit(f"nsys binary not found: {nsys_bin}. Set --nsys-bin or $NSYS_BIN to a valid path.")

    server_cfg = prepare_server_config(config.get("server_config", {}))
    server_cmd, env = build_server_command(server_cfg)
    # Use spawn for safety with profiling.
    env["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
    nsys_cmd = [
        resolved,
        "profile",
        "--output",
        args.nsys_output,
        "--trace",
        args.nsys_trace,
        "--sample",
        "none",
        "--cpuctxsw",
        "none",
        "--trace-fork-before-exec",
        "true",
        "--cuda-graph-trace",
        "node",
        "--force-overwrite",
        "true",
        "--duration",
        str(args.nsys_duration),
    ]
    if args.nsys_delay and args.nsys_delay > 0:
        nsys_cmd.extend(["--delay", str(args.nsys_delay)])

    full_cmd = nsys_cmd + server_cmd
    logger.info(f"Launching vLLM under nsys: {' '.join(full_cmd)}")
    server_proc = subprocess.Popen(
        full_cmd,
        env=env,
        preexec_fn=os.setsid if hasattr(os, "setsid") else None,
    )

    try:
        wait_for_server(server_cfg["base_url"], server_cfg.get("api_key", "EMPTY"), logger, timeout=args.wait_timeout)
    except Exception as exc:
        logger.error(f"vLLM server failed readiness checks: {exc}")
        try:
            if hasattr(os, "killpg"):
                os.killpg(server_proc.pid, signal.SIGTERM)
            else:
                server_proc.terminate()
        except ProcessLookupError:
            pass
        try:
            server_proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            server_proc.kill()
        raise SystemExit(1)

    client_cmd = [
        sys.executable,
        str(Path(__file__).with_name("client_entry.py")),
        "--config",
        args.config,
    ]
    if args.debug:
        client_cmd.append("--debug")
    logger.info(f"Starting request sender: {' '.join(client_cmd)}")
    client_rc = subprocess.call(client_cmd)
    if client_rc != 0:
        logger.warning(f"Request sender exited with code {client_rc}")

    logger.info("Stopping profiled vLLM server...")
    stop_vllm_server(server_proc, logger)

    # Check for expected nsys output.
    expected = Path(args.nsys_output).with_suffix(".nsys-rep")
    if expected.exists():
        logger.info(f"nsys report written: {expected}")
    else:
        logger.warning(f"nsys report {expected} not found; ensure nsys was available on PATH or via --nsys-bin.")


if __name__ == "__main__":
    main()
