from __future__ import annotations

import logging
from pathlib import Path


def load_prompts(prompts_dir: str | Path, target_n: int, warmup_n: int, logger: logging.Logger) -> list[str]:
    """Load prompt texts from disk, padding as needed to meet target counts."""
    path = Path(prompts_dir)
    if not path.exists():
        logger.error(f"Prompts directory does not exist: {prompts_dir}")
        return []

    prompt_files = sorted(path.rglob("*.txt"))
    prompts: list[str] = []
    for prompt_path in prompt_files:
        try:
            prompts.append(prompt_path.read_text(encoding="utf-8"))
        except Exception as exc:
            logger.warning(f"Failed to read prompt {prompt_path}: {exc}")

    total_needed = max(0, target_n) + max(0, warmup_n)
    if total_needed and len(prompts) < total_needed and prompts:
        logger.warning(f"Only {len(prompts)} prompts found, repeating to reach {total_needed}.")
        while len(prompts) < total_needed:
            prompts.extend(prompts[: max(1, total_needed - len(prompts))])

    if total_needed:
        prompts = prompts[:total_needed]

    logger.info(f"Loaded {len(prompts)} prompts from {prompts_dir} (needs {total_needed})")
    return prompts
