from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from config_handler import load_config
from logger import setup_logger
from prompt_generator import get_prompts
from vllm_runner import run_vllm, dump_results


def main():
    parser = argparse.ArgumentParser(description="Run vLLM benchmark.")
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
    num_prompts = benchmark_cfg.get("num_prompts", 4)
    dataset_name = benchmark_cfg.get("dataset_name", "random")
    random_input_len = benchmark_cfg.get("random_input_len", 512)
    seed = benchmark_cfg.get("seed", 42)

    logger.info("Generating prompts...")
    prompts = get_prompts(dataset_name, num_prompts, prompt_len=random_input_len, seed=seed)

    results = run_vllm(prompts, config, logger, debug=args.debug)

    output_cfg = config.get("output_config", {})
    if output_cfg.get("save_json", True):
        out_dir = Path(output_cfg.get("output_dir", "./output"))
        out_dir.mkdir(parents=True, exist_ok=True)
        dump_results(results, out_dir / "results.json", logger)

    if output_cfg.get("show_stats", True):
        logger.info("\n--- Benchmark Results ---")
        logger.info(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
