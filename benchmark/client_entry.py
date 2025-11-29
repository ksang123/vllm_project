from __future__ import annotations

import argparse
import logging
import warnings
from pathlib import Path

from config_handler import load_config
from logger import setup_logger
from prompt_loader import load_prompts
from vllm_runner import dump_results, run_client


def main() -> None:
    warnings.filterwarnings("ignore", category=FutureWarning, message="The pynvml package is deprecated.*")
    parser = argparse.ArgumentParser(description="Send prompts to an already-running vLLM server.")
    parser.add_argument("--config", default="bench_config.yaml", help="Path to bench_config.yaml")
    parser.add_argument("--debug", action="store_true", help="Include generated texts/config in output")
    args = parser.parse_args()

    config = load_config(args.config)
    if not config:
        raise SystemExit("Failed to load configuration.")

    log_file = config.get("log_config", {}).get("log_file", "benchmark.log")
    setup_logger(log_file)
    logger = logging.getLogger("vllm_benchmark")

    benchmark_cfg = config.get("benchmark_config", {})
    prompts_dir = benchmark_cfg.get("prompts_dir")
    if not prompts_dir:
        logger.error("No prompts_dir configured; please set benchmark_config.prompts_dir")
        return

    prompts = load_prompts(
        prompts_dir,
        benchmark_cfg.get("num_prompts") or 0,
        benchmark_cfg.get("warmup_prompts", 0) or 0,
        logger,
    )
    if not prompts:
        logger.error("No prompts were loaded; aborting request sending.")
        return

    results = run_client(prompts, config, logger, debug=args.debug, wait_for_server_ready=True)

    output_cfg = config.get("output_config", {})
    out_dir = Path(output_cfg.get("output_dir", "./output"))
    out_dir.mkdir(parents=True, exist_ok=True)

    if output_cfg.get("save_charts", True):
        subdirs = [p for p in out_dir.iterdir() if p.is_dir()]
        if subdirs:
            latest = max(subdirs, key=lambda p: p.stat().st_mtime)
            dump_path = latest / "results.json"
        else:
            dump_path = out_dir / "results.json"
    else:
        dump_path = out_dir / "results.json"

    if output_cfg.get("save_json", True):
        dump_results(results, dump_path, logger)


if __name__ == "__main__":
    main()
