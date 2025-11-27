"""HTTP request sender for the vLLM benchmark."""

from __future__ import annotations

import time
from typing import Any

import requests


def _sampling_payload(benchmark_cfg: dict[str, Any]) -> dict[str, Any]:
    payload = {
        "temperature": benchmark_cfg.get("temperature"),
        "top_p": benchmark_cfg.get("top_p"),
        "max_tokens": benchmark_cfg.get("random_output_len"),
        "presence_penalty": benchmark_cfg.get("presence_penalty"),
        "frequency_penalty": benchmark_cfg.get("frequency_penalty"),
        "logprobs": benchmark_cfg.get("logprobs"),
        "n": benchmark_cfg.get("n", 1),
        # vLLM OpenAI API ignores best_of for chat; omit to avoid warnings.
        "stop": benchmark_cfg.get("stop") or [],
        "stream": benchmark_cfg.get("stream", False),
    }
    # Drop None values
    return {k: v for k, v in payload.items() if v is not None}


def _get_metrics(base_url: str, logger) -> str | None:
    try:
        resp = requests.get(f"{base_url}/metrics", timeout=10)
        resp.raise_for_status()
        return resp.text
    except requests.RequestException as exc:  # pragma: no cover
        if logger:
            logger.warning(f"Could not fetch /metrics: {exc}")
        return None


def send_requests(
    prompts: list[str],
    server_cfg: dict[str, Any],
    benchmark_cfg: dict[str, Any],
    logger=None,
) -> dict[str, Any]:
    """Send prompts to a running vLLM server and collect client-side metrics."""
    base_url = server_cfg.get("base_url", "http://127.0.0.1:8000")
    api_base = benchmark_cfg.get("api_base") or f"{base_url}/v1"
    api_key = server_cfg.get("api_key") or "EMPTY"
    url = f"{api_base.rstrip('/')}/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}"}
    payload_base = _sampling_payload(benchmark_cfg)

    generated_texts: list[str] = []
    per_request: list[dict[str, Any]] = []

    start = time.time()
    for prompt in prompts:
        body = {
            "model": server_cfg.get("model"),
            "messages": [{"role": "user", "content": prompt}],
            **payload_base,
        }
        t0 = time.time()
        resp = requests.post(url, json=body, headers=headers, timeout=120)
        latency = time.time() - t0
        resp.raise_for_status()
        data = resp.json()
        text = data["choices"][0]["message"]["content"]
        usage = data.get("usage") or {}
        generated_texts.append(text)
        per_request.append(
            {
                "latency_s": latency,
                "usage": {
                    "prompt_tokens": usage.get("prompt_tokens"),
                    "completion_tokens": usage.get("completion_tokens"),
                },
            }
        )
    total_time = time.time() - start

    metrics_text = _get_metrics(base_url, logger)

    return {
        "generated_texts": generated_texts,
        "per_request": per_request,
        "total_time": total_time,
        "metrics_text": metrics_text,
    }
