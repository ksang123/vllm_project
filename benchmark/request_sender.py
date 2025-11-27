"""HTTP request sender for the vLLM benchmark."""

from __future__ import annotations

import multiprocessing as mp
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


def _build_latency_histogram(latencies: list[float]) -> dict[str, Any]:
    if not latencies:
        return {}
    # Log-spaced buckets for better percentile fidelity
    edges = [0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0]
    edges.sort()
    buckets: dict[str, float] = {}
    total = len(latencies)
    latencies_sorted = sorted(latencies)
    idx = 0
    for edge in edges:
        while idx < total and latencies_sorted[idx] <= edge:
            idx += 1
        buckets[str(edge)] = float(idx)
    buckets["+Inf"] = float(total)
    return {
        "name": "client:latency_seconds",
        "type": "histogram",
        "labels": {"source": "client"},
        "buckets": buckets,
        "count": float(total),
        "sum": float(sum(latencies)),
    }


def _send_batch(batch_prompts: list[str], url: str, headers: dict[str, str], body_base: dict[str, Any]) -> dict[str, Any]:
    generated_texts: list[str] = []
    per_request: list[dict[str, Any]] = []
    start = time.time()
    session = requests.Session()
    for prompt in batch_prompts:
        body = {
            **body_base,
            "messages": [{"role": "user", "content": prompt}],
        }
        t0 = time.time()
        resp = session.post(url, json=body, headers=headers, timeout=120)
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
    return {"generated_texts": generated_texts, "per_request": per_request, "total_time": total_time}


def _process_chunk(chunk: list[str], url: str, headers: dict[str, str], body_base: dict[str, Any], inter_request_delay: float) -> dict[str, Any]:
    out = {"generated_texts": [], "per_request": []}
    session = requests.Session()
    last_send = 0.0
    for prompt in chunk:
        now = time.time()
        sleep_for = inter_request_delay - (now - last_send)
        if sleep_for > 0:
            time.sleep(sleep_for)
        last_send = time.time()
        body = {
            **body_base,
            "messages": [{"role": "user", "content": prompt}],
        }
        t0 = time.time()
        resp = session.post(url, json=body, headers=headers, timeout=120)
        latency = time.time() - t0
        resp.raise_for_status()
        data = resp.json()
        text = data["choices"][0]["message"]["content"]
        usage = data.get("usage") or {}
        out["generated_texts"].append(text)
        out["per_request"].append(
            {
                "latency_s": latency,
                "start_ts": t0,
                "end_ts": t0 + latency,
                "usage": {
                    "prompt_tokens": usage.get("prompt_tokens"),
                    "completion_tokens": usage.get("completion_tokens"),
                },
            }
        )
    return out


def send_requests(
    prompts: list[str],
    server_cfg: dict[str, Any],
    benchmark_cfg: dict[str, Any],
    logger=None,
) -> dict[str, Any]:
    """Send prompts to a running vLLM server and collect client-side metrics."""
    base_url = server_cfg.get("base_url", "http://127.0.0.1:8000")
    if not base_url:
        host = server_cfg.get("host", "127.0.0.1")
        port = server_cfg.get("port", 8000)
        base_url = f"http://{host}:{port}"
    api_base = benchmark_cfg.get("api_base") or f"{base_url.rstrip('/')}/v1"
    api_key = server_cfg.get("api_key") or "EMPTY"
    url = f"{api_base}/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}"}
    payload_base = _sampling_payload(benchmark_cfg)
    body_base = {"model": server_cfg.get("model"), **payload_base}

    processes = int(benchmark_cfg.get("client_processes", 1))
    if processes < 1:
        processes = 1

    target_rps = benchmark_cfg.get("target_rps")
    try:
        target_rps = float(target_rps) if target_rps not in (None, "inf", "INF") else float("inf")
    except Exception:
        target_rps = float("inf")
    inter_request_delay = 0.0 if target_rps == float("inf") else 1.0 / max(target_rps, 1e-9)

    max_concurrency = int(benchmark_cfg.get("max_concurrency", processes))
    if max_concurrency < 1:
        max_concurrency = processes

    start = time.time()
    if processes == 1:
        batch_result = _process_chunk(prompts, url, headers, body_base, inter_request_delay)
        generated_texts = batch_result["generated_texts"]
        per_request = batch_result["per_request"]
    else:
        chunks = [[] for _ in range(processes)]
        for i, p in enumerate(prompts):
            chunks[i % processes].append(p)

        with mp.Pool(processes=min(processes, max_concurrency)) as pool:
            results_list = pool.starmap(
                _process_chunk,
                [
                    (chunk, url, headers, body_base, inter_request_delay)
                    for chunk in chunks
                    if chunk
                ],
            )
        generated_texts = []
        per_request = []
        for res in results_list:
            generated_texts.extend(res["generated_texts"])
            per_request.extend(res["per_request"])

    total_time = time.time() - start

    metrics_text = _get_metrics(base_url, logger)
    client_metrics = []
    lat_hist = _build_latency_histogram([r.get("latency_s", 0.0) for r in per_request])
    if lat_hist:
        client_metrics.append(lat_hist)

    return {
        "generated_texts": generated_texts,
        "per_request": per_request,
        "total_time": total_time,
        "metrics_text": metrics_text,
        "client_metrics": client_metrics,
    }
