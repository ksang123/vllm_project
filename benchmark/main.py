from __future__ import annotations

import argparse
import json
import logging
import warnings
from pathlib import Path

from config_handler import load_config
from logger import setup_logger
from vllm_runner import run_vllm, dump_results


def main():
    # Suppress noisy pynvml deprecation warnings surfaced by torch.
    warnings.filterwarnings("ignore", category=FutureWarning, message="The pynvml package is deprecated.*")
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
    prompts_dir = benchmark_cfg.get("prompts_dir")
    if prompts_dir:
        prompt_files = sorted(Path(prompts_dir).rglob("*.txt"))
        prompts = []
        for p in prompt_files:
            try:
                prompts.append(p.read_text())
            except Exception as exc:
                logger.warning(f"Failed to read prompt {p}: {exc}")
        target_n = benchmark_cfg.get("num_prompts") or 0
        warmup_n = benchmark_cfg.get("warmup_prompts", 0) or 0
        total_needed = target_n + warmup_n
        if total_needed and len(prompts) < total_needed and prompts:
            logger.warning(f"Only {len(prompts)} prompts found, repeating to reach {total_needed}.")
            while len(prompts) < total_needed:
                prompts.extend(prompts[: max(1, total_needed - len(prompts))])
        if total_needed:
            prompts = prompts[:total_needed]
        logger.info(f"Loaded {len(prompts)} prompts from {prompts_dir} (needs {total_needed})")
    else:
        logger.error("No prompts_dir configured; please set benchmark_config.prompts_dir")
        return

    # Run benchmark and capture results
    results = run_vllm(prompts, config, logger, debug=args.debug)

    output_cfg = config.get("output_config", {})
    out_dir = Path(output_cfg.get("output_dir", "./output"))
    out_dir.mkdir(parents=True, exist_ok=True)

    # If charts are saved, render_all already wrote to a timestamped folder.
    # Place results.json alongside the latest charts when available.
    if output_cfg.get("save_charts", True):
        # The render_all function creates a timestamped subfolder; find the latest one.
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
