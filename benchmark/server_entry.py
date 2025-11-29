from __future__ import annotations

import argparse
import logging
import warnings

from config_handler import load_config
from logger import setup_logger
from vllm_runner import prepare_server_config, start_vllm_server, stop_vllm_server


def main() -> None:
    warnings.filterwarnings("ignore", category=FutureWarning, message="The pynvml package is deprecated.*")
    parser = argparse.ArgumentParser(description="Start the vLLM server using the benchmark config.")
    parser.add_argument("--config", default="bench_config.yaml", help="Path to bench_config.yaml")
    parser.add_argument(
        "--no-wait",
        action="store_true",
        help="Start the server without waiting for readiness checks to pass.",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    if not config:
        raise SystemExit("Failed to load configuration.")

    log_file = config.get("log_config", {}).get("log_file", "benchmark.log")
    setup_logger(log_file)
    logger = logging.getLogger("vllm_benchmark")

    server_cfg = prepare_server_config(config.get("server_config", {}))
    logger.info(f"Starting vLLM server on {server_cfg.get('host', '127.0.0.1')}:{server_cfg.get('port', 8000)}")

    server_proc = start_vllm_server(server_cfg, logger, wait_for_ready=not args.no_wait)
    logger.info(f"vLLM server running (PID {server_proc.pid}). Press Ctrl+C to stop.")
    try:
        server_proc.wait()
    except KeyboardInterrupt:
        logger.info("Stopping vLLM server...")
    finally:
        stop_vllm_server(server_proc, logger)


if __name__ == "__main__":
    main()
