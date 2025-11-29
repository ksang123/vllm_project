# VLLM runner for the benchmark.
# Orchestrates request sending, metric parsing, and graph generation.

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
import warnings
from dataclasses import asdict, is_dataclass
from statistics import mean
from typing import Any

import requests

from generate_graphs import render_all
from request_sender import send_requests


def _to_serializable(obj: Any) -> Any:
    if obj is None:
        return None
    if is_dataclass(obj):
        return asdict(obj)
    if isinstance(obj, dict):
        return {k: _to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_serializable(v) for v in obj]
    if hasattr(obj, "__dict__"):
        return {k: _to_serializable(v) for k, v in obj.__dict__.items() if not k.startswith("_")}
    return obj


def _summarize(values: list[float]) -> dict[str, float]:
    if not values:
        return {}
    sorted_vals = sorted(values)
    n = len(sorted_vals)

    def pct(p: float) -> float:
        if n == 1:
            return sorted_vals[0]
        k = (n - 1) * (p / 100.0)
        f = int(k)
        c = min(f + 1, n - 1)
        return sorted_vals[f] + (sorted_vals[c] - sorted_vals[f]) * (k - f)

    return {
        "count": n,
        "avg": mean(sorted_vals),
        "p50": pct(50),
        "p90": pct(90),
        "p95": pct(95),
        "p99": pct(99),
        "max": sorted_vals[-1],
        "min": sorted_vals[0],
    }


def _parse_prometheus_metrics(text: str, wanted: set[str]) -> list[dict[str, Any]]:
    type_map: dict[str, str] = {}
    hist: dict[tuple[str, frozenset[tuple[str, str]]], dict[str, Any]] = {}
    results: list[dict[str, Any]] = []

    for line in text.splitlines():
        if not line:
            continue
        if line.startswith("# TYPE"):
            parts = line.split()
            if len(parts) >= 4:
                _, _, name, mtype = parts[:4]
                type_map[name] = mtype
            continue
        if line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 2:
            continue
        name_and_labels, value = parts[0], parts[1]

        if "{" in name_and_labels:
            name, labels_part = name_and_labels.split("{", 1)
            labels_part = labels_part.rstrip("}")
            labels: dict[str, str] = {}
            if labels_part:
                for pair in labels_part.split(","):
                    if "=" in pair:
                        k, v = pair.split("=", 1)
                        labels[k] = v.strip('"')
        else:
            name = name_and_labels
            labels = {}

        base_name = name
        suffix = None
        for suff in ("_bucket", "_sum", "_count"):
            if name.endswith(suff):
                base_name = name[: -len(suff)]
                suffix = suff
                break

        if (suffix and base_name not in wanted) or (not suffix and name not in wanted):
            continue

        try:
            val = float(value)
        except ValueError:
            continue

        if suffix is None:
            mtype = type_map.get(name, "gauge")
            results.append({"name": name, "type": mtype, "labels": labels, "value": val})
            continue

        labels_no_le = dict(labels)
        bucket_le = labels_no_le.pop("le", None)
        key = (base_name, frozenset(labels_no_le.items()))
        entry = hist.get(key)
        if entry is None:
            entry = {
                "name": base_name,
                "type": "histogram",
                "labels": labels_no_le,
                "buckets": {},
                "count": None,
                "sum": None,
            }
            hist[key] = entry

        if suffix == "_bucket" and bucket_le is not None:
            entry["buckets"][bucket_le] = val
        elif suffix == "_count":
            entry["count"] = val
        elif suffix == "_sum":
            entry["sum"] = val

    results.extend(hist.values())
    return results


def _hist_summary(metric: dict[str, Any]) -> dict[str, float] | None:
    if metric.get("type") != "histogram":
        return None
    buckets = metric.get("buckets") or {}
    if not buckets:
        return None
    items = []
    for k, v in buckets.items():
        if k == "+Inf":
            continue
        try:
            upper = float(k)
            items.append((upper, float(v)))
        except ValueError:
            continue
    if not items:
        return None
    items.sort(key=lambda x: x[0])
    total = metric.get("count")
    if total is None:
        total = items[-1][1]
    if not total:
        return None

    targets = {"p50": 0.50 * total, "p90": 0.90 * total, "p95": 0.95 * total, "p99": 0.99 * total}
    out: dict[str, float] = {}
    for label, target in targets.items():
        for upper, cum in items:
            if cum >= target:
                out[label] = upper
                break
    if "p99" not in out:
        out["p99"] = items[-1][0]
    s = metric.get("sum")
    if s is not None:
        out["avg"] = s / total if total else 0.0
    out["min"] = 0.0
    out["max"] = items[-1][0]
    return out


def prepare_server_config(server_cfg: dict[str, Any]) -> dict[str, Any]:
    host = server_cfg.get("host", "127.0.0.1")
    port = int(server_cfg.get("port", 8000))
    base_url = server_cfg.get("base_url") or f"http://{host}:{port}"
    prepared = dict(server_cfg)
    prepared.update({"host": host, "port": port, "base_url": base_url})
    return prepared


def build_server_command(server_cfg: dict[str, Any]) -> tuple[list[str], dict[str, str]]:
    cfg = prepare_server_config(server_cfg)
    model = cfg.get("model")
    tokenizer = cfg.get("tokenizer")
    served_model_name = cfg.get("served_model_name")
    disable_log_stats = cfg.get("disable_log_stats", False)
    api_key = cfg.get("api_key", "EMPTY")

    cmd = [
        sys.executable,
        "-m",
        "vllm.entrypoints.openai.api_server",
        "--host",
        cfg.get("host", "127.0.0.1"),
        "--port",
        str(cfg.get("port", 8000)),
        "--model",
        model,
    ]
    if tokenizer:
        cmd.extend(["--tokenizer", tokenizer])
    if served_model_name:
        cmd.extend(["--served-model-name", served_model_name])
    if disable_log_stats:
        cmd.append("--disable-log-stats")
    if cfg.get("max_model_len"):
        cmd.extend(["--max-model-len", str(cfg["max_model_len"])])
    if cfg.get("gpu_memory_utilization") is not None:
        cmd.extend(["--gpu-memory-utilization", str(cfg["gpu_memory_utilization"])])
    if cfg.get("dtype"):
        cmd.extend(["--dtype", str(cfg["dtype"])])
    if cfg.get("tensor_parallel_size"):
        cmd.extend(["--tensor-parallel-size", str(cfg["tensor_parallel_size"])])
    if cfg.get("quantization"):
        cmd.extend(["--quantization", str(cfg["quantization"])])
    if cfg.get("enable_prefix_caching"):
        cmd.append("--enable-prefix-caching")
    if cfg.get("tokenizer_mode"):
        cmd.extend(["--tokenizer-mode", str(cfg["tokenizer_mode"])])
    if cfg.get("trust_remote_code"):
        cmd.append("--trust-remote-code")
    if cfg.get("engine_use_ray"):
        cmd.append("--engine-use-ray")
    if cfg.get("max_num_seqs"):
        cmd.extend(["--max-num-seqs", str(cfg["max_num_seqs"])])
    if cfg.get("max_num_batched_tokens"):
        cmd.extend(["--max-num-batched-tokens", str(cfg["max_num_batched_tokens"])])
    if cfg.get("block_size"):
        cmd.extend(["--block-size", str(cfg["block_size"])])
    if cfg.get("swap_space") is not None:
        cmd.extend(["--swap-space", str(cfg["swap_space"])])
    if cfg.get("cpu_offload_gb") is not None:
        cmd.extend(["--cpu-offload-gb", str(cfg["cpu_offload_gb"])])
    if cfg.get("revision"):
        cmd.extend(["--revision", str(cfg["revision"])])
    if cfg.get("code_revision"):
        cmd.extend(["--code-revision", str(cfg["code_revision"])])
    if cfg.get("tokenizer_revision"):
        cmd.extend(["--tokenizer-revision", str(cfg["tokenizer_revision"])])
    if cfg.get("max_log_len"):
        cmd.extend(["--max-log-len", str(cfg["max_log_len"])])

    env = os.environ.copy()
    env.setdefault("VLLM_API_KEY", api_key)
    return cmd, env


def wait_for_server(base_url: str, api_key: str, logger, timeout: float = 300.0, interval: float = 2.0) -> None:
    deadline = time.time() + timeout
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    while time.time() < deadline:
        try:
            r = requests.get(f"{base_url}/health", timeout=5)
            if r.status_code == 200:
                return
        except requests.RequestException:
            pass
        try:
            r = requests.get(f"{base_url}/v1/models", timeout=5, headers=headers)
            if r.status_code == 200:
                return
        except requests.RequestException:
            pass
        time.sleep(interval)
    raise RuntimeError(f"vLLM server did not become ready at {base_url} within {timeout}s")


def start_vllm_server(server_cfg: dict[str, Any], logger, wait_for_ready: bool = True) -> subprocess.Popen:
    prepared = prepare_server_config(server_cfg)
    cmd, env = build_server_command(prepared)
    logger.info(f"Starting vLLM server: {' '.join(cmd)}")
    proc = subprocess.Popen(cmd, env=env)
    if wait_for_ready:
        wait_for_server(prepared["base_url"], prepared.get("api_key", "EMPTY"), logger)
    return proc


def stop_vllm_server(proc: subprocess.Popen, logger) -> None:
    if proc:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            logger.warning("vLLM server did not terminate gracefully, killing...")
            proc.kill()


def run_client(
    prompts: list[str],
    config: dict[str, Any],
    logger,
    debug: bool = False,
    wait_for_server_ready: bool = True,
    server_cfg_override: dict[str, Any] | None = None,
) -> dict[str, Any]:
    benchmark_cfg = config.get("benchmark_config", {})
    server_cfg = server_cfg_override or config.get("server_config", {})
    output_cfg = config.get("output_config", {})
    telemetry_cfg = config.get("telemetry_config", {})

    server_cfg = prepare_server_config(server_cfg)
    if wait_for_server_ready:
        wait_for_server(server_cfg["base_url"], server_cfg.get("api_key", "EMPTY"), logger)

    # Warmup if requested
    warmup_n = int(benchmark_cfg.get("warmup_prompts", 0))
    if warmup_n > 0:
        warmup_prompts = prompts[:warmup_n]
        logger.info(f"Warmup with {len(warmup_prompts)} prompts (not measured)")
        warmup_cfg = dict(benchmark_cfg)
        warmup_cfg["target_rps"] = warmup_cfg.get("target_rps", "inf")
        warmup_cfg["client_processes"] = max(1, int(warmup_cfg.get("client_processes", 1)))
        try:
            send_requests(warmup_prompts, server_cfg, warmup_cfg, logger)
        except Exception as exc:
            logger.warning(f"Warmup failed but continuing: {exc}")
        prompts = prompts[warmup_n:]

    # Telemetry sampling (best effort)
    telemetry_samples: dict[str, list[dict[str, float]]] = {"gpu": [], "cpu": []}
    stop_telemetry = False

    def _sample_telemetry():
        import psutil  # type: ignore
        try:
            warnings.filterwarnings("ignore", category=FutureWarning, message="The pynvml package is deprecated.*")
            from pynvml import nvmlDeviceGetHandleByIndex, nvmlDeviceGetMemoryInfo, nvmlDeviceGetUtilizationRates, nvmlInit, nvmlShutdown  # type: ignore

            nvmlInit()
            handle = nvmlDeviceGetHandleByIndex(0)
            has_gpu = True
        except Exception:
            handle = None
            has_gpu = False
        interval = float(telemetry_cfg.get("sample_interval_s", 1.0))
        while not stop_telemetry:
            ts = time.time()
            if telemetry_cfg.get("enable_cpu", True):
                cpu = psutil.cpu_percent(interval=None)
                telemetry_samples["cpu"].append({"ts": ts, "cpu_percent": cpu})
            if telemetry_cfg.get("enable_gpu", True) and has_gpu and handle:
                try:
                    util = nvmlDeviceGetUtilizationRates(handle)
                    mem = nvmlDeviceGetMemoryInfo(handle)
                    telemetry_samples["gpu"].append(
                        {
                            "ts": ts,
                            "gpu_util_percent": util.gpu,
                            "mem_util_percent": (mem.used / mem.total) * 100 if mem.total else 0.0,
                            "mem_used_bytes": mem.used,
                            "mem_total_bytes": mem.total,
                        }
                    )
                except Exception:
                    pass
            time.sleep(interval)
        if has_gpu:
            try:
                nvmlShutdown()
            except Exception:
                pass

    telemetry_thread = None
    if telemetry_cfg.get("enable_cpu", True) or telemetry_cfg.get("enable_gpu", True):
        import threading

        telemetry_thread = threading.Thread(target=_sample_telemetry, daemon=True)
        telemetry_thread.start()

    logger.info(f"Sending {len(prompts)} prompts to {server_cfg.get('base_url')}")
    try:
        send_result = send_requests(prompts, server_cfg, benchmark_cfg, logger)
    finally:
        stop_telemetry = True
        if telemetry_thread:
            telemetry_thread.join(timeout=2)

    per_request = send_result["per_request"]
    total_time = send_result["total_time"]
    metrics_text = send_result.get("metrics_text")
    engine_metrics = []
    if metrics_text:
        wanted = {
            "vllm:kv_cache_usage_perc",
            "vllm:num_requests_running",
            "vllm:num_requests_waiting",
            "vllm:request_prompt_tokens",
            "vllm:request_generation_tokens",
            "vllm:time_to_first_token_seconds",
            "vllm:time_per_output_token_seconds",
            "vllm:request_time_per_output_token_seconds",
            "vllm:e2e_request_latency_seconds",
            "vllm:request_queue_time_seconds",
            "vllm:request_inference_time_seconds",
            "vllm:request_prefill_time_seconds",
            "vllm:request_decode_time_seconds",
            "vllm:prefix_cache_queries",
            "vllm:prefix_cache_hits",
            "vllm:external_prefix_cache_queries",
            "vllm:external_prefix_cache_hits",
            "vllm:mm_cache_queries",
            "vllm:mm_cache_hits",
        }
        engine_metrics = _parse_prometheus_metrics(metrics_text, wanted)

    prompt_tokens = sum((req.get("usage") or {}).get("prompt_tokens", 0) for req in per_request)
    completion_tokens = sum((req.get("usage") or {}).get("completion_tokens", 0) for req in per_request)
    total_tokens = prompt_tokens + completion_tokens

    per_request_latencies = [req.get("latency_s", 0.0) for req in per_request]
    per_request_completion_tokens = [float((req.get("usage") or {}).get("completion_tokens", 0)) for req in per_request]

    results = {
        "model": server_cfg.get("model"),
        "num_prompts": len(prompts),
        "total_run_time_s": total_time,
        "total_prompt_tokens": prompt_tokens,
        "total_output_tokens": completion_tokens,
        "total_tokens": total_tokens,
        "throughput_prompts_per_s": len(prompts) / total_time if total_time > 0 else 0.0,
        "throughput_output_tokens_per_s": completion_tokens / total_time if total_time > 0 else 0.0,
        "throughput_total_tokens_per_s": total_tokens / total_time if total_time > 0 else 0.0,
        "per_request": per_request,
        "per_request_summary": {
            "latency_s": _summarize(per_request_latencies),
            "completion_tokens": _summarize(per_request_completion_tokens),
        },
    }

    if debug:
        results["generated_texts"] = send_result.get("generated_texts")
        results["config"] = _to_serializable(config)

    if engine_metrics:
        results["engine_metrics"] = engine_metrics
        summary_targets = {
            "vllm:time_to_first_token_seconds",
            "vllm:time_per_output_token_seconds",
            "vllm:request_time_per_output_token_seconds",
            "vllm:e2e_request_latency_seconds",
            "vllm:request_queue_time_seconds",
            "vllm:request_inference_time_seconds",
            "vllm:request_prefill_time_seconds",
            "vllm:request_decode_time_seconds",
        }
        hist_summaries = {}
        for m in engine_metrics:
            if m.get("name") in summary_targets:
                s = _hist_summary(m)
                if s:
                    hist_summaries[m["name"]] = s
        if hist_summaries:
            results["engine_metric_summaries"] = hist_summaries

        def _ratio(name_hit: str, name_q: str) -> float:
            hit = next((m for m in engine_metrics if m.get("name") == name_hit), None)
            q = next((m for m in engine_metrics if m.get("name") == name_q), None)
            hv = hit.get("value") if hit else 0.0
            qv = q.get("value") if q else 0.0
            return (hv / qv) if qv else 0.0

        results["cache_stats"] = {
            "prefix_hit_rate": _ratio("vllm:prefix_cache_hits", "vllm:prefix_cache_queries"),
            "external_prefix_hit_rate": _ratio("vllm:external_prefix_cache_hits", "vllm:external_prefix_cache_queries"),
            "mm_cache_hit_rate": _ratio("vllm:mm_cache_hits", "vllm:mm_cache_queries"),
        }

    client_metrics = send_result.get("client_metrics") or []
    if client_metrics:
        results["client_metrics"] = client_metrics

    if telemetry_samples:
        results["telemetry"] = telemetry_samples

    if output_cfg.get("save_charts", True):
        out_dir = output_cfg.get("output_dir", "./output")
        render_all(results, base_out_dir=out_dir)

    return results


def run_vllm(prompts: list[str], config: dict[str, Any], logger, debug: bool = False) -> dict[str, Any]:
    server_cfg = prepare_server_config(config.get("server_config", {}))
    logger.info(f"Starting vLLM server on {server_cfg.get('host', '127.0.0.1')}:{server_cfg.get('port', 8000)}")
    server_proc = start_vllm_server(server_cfg, logger, wait_for_ready=True)
    try:
        return run_client(prompts, config, logger, debug=debug, wait_for_server_ready=False, server_cfg_override=server_cfg)
    finally:
        stop_vllm_server(server_proc, logger)


def dump_results(results: dict[str, Any], path: str, logger) -> None:
    with open(path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Wrote results to {path}")
