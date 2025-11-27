"""Utilities to visualize benchmark output produced by vllm_runner.py."""

from __future__ import annotations

import datetime
import math
from pathlib import Path
from typing import Any, Iterable, Sequence

import matplotlib.pyplot as plt

plt.style.use("ggplot")


def _ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def _histogram_edges(metric: dict[str, Any]) -> tuple[list[float], list[float]]:
    """Convert Prometheus histogram buckets to bin edges and counts."""
    buckets = metric.get("buckets") or {}
    if not buckets:
        return [], []
    items: list[tuple[float, float]] = []
    for k, v in buckets.items():
        if k == "+Inf":
            continue
        try:
            upper = float(k)
            items.append((upper, float(v)))
        except ValueError:
            continue
    items.sort(key=lambda x: x[0])
    edges = [0.0] + [u for u, _ in items]
    counts = []
    prev = 0.0
    for _, cum in items:
        counts.append(cum - prev)
        prev = cum
    return edges, counts


def _plot_hist(metric: dict[str, Any], out_dir: Path) -> None:
    name = metric.get("name", "histogram")
    # Friendly labels based on known metric names.
    label_map = {
        "vllm:request_prompt_tokens": ("Prompt tokens per request", "tokens"),
        "vllm:request_generation_tokens": ("Generation tokens per request", "tokens"),
        "vllm:time_to_first_token_seconds": ("Time to first token", "seconds"),
        "vllm:time_per_output_token_seconds": ("Time per output token", "seconds"),
        "vllm:request_time_per_output_token_seconds": ("Request avg time per output token", "seconds"),
        "vllm:e2e_request_latency_seconds": ("End-to-end latency", "seconds"),
        "vllm:request_queue_time_seconds": ("Queue time", "seconds"),
        "vllm:request_inference_time_seconds": ("Inference time (prefill+decode)", "seconds"),
        "vllm:request_prefill_time_seconds": ("Prefill time", "seconds"),
        "vllm:request_decode_time_seconds": ("Decode time", "seconds"),
    }
    title, xlabel = label_map.get(name, (name, "value"))
    edges, counts = _histogram_edges(metric)
    if not edges or not counts:
        return
    if all(c == counts[0] for c in counts):
        # Skip flat histograms (no variation).
        return

    mids = [(edges[i] + edges[i + 1]) / 2 for i in range(len(counts))]
    plt.figure(figsize=(6, 4))
    ax = plt.gca()
    ax.bar(mids, counts, width=[edges[i + 1] - edges[i] for i in range(len(counts))], edgecolor="black", alpha=0.7)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("count")

    # Overlay CDF for a more informative view.
    total = sum(counts)
    if total > 0:
        cum = [sum(counts[: i + 1]) / total for i in range(len(counts))]
        ax2 = ax.twinx()
        ax2.plot(mids, cum, color="darkred", marker="o", linestyle="-", linewidth=1)
        ax2.set_ylabel("cdf")

    plt.tight_layout()
    outfile = out_dir / f"{name.replace(':', '_')}.png"
    plt.savefig(outfile)
    plt.close()


def _plot_per_request_latencies(per_request: Iterable[dict[str, Any]], out_dir: Path) -> None:
    latencies = [req.get("latency_s", 0.0) for req in per_request]
    if not latencies:
        return
    plt.figure(figsize=(6, 4))
    plt.hist(latencies, bins=min(20, max(5, len(latencies))), edgecolor="black", alpha=0.75)
    plt.title("Per-request latency (s)")
    plt.xlabel("Latency (seconds)")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(out_dir / "per_request_latency.png")
    plt.close()

    # Scatter plot ordered by latency for shape insight.
    plt.figure(figsize=(6, 3))
    sorted_lat = sorted(latencies)
    plt.plot(sorted_lat, marker="o", linestyle="-", linewidth=1)
    plt.title("Per-request latency (sorted)")
    plt.xlabel("Request index (sorted by latency)")
    plt.ylabel("Latency (seconds)")
    plt.tight_layout()
    plt.savefig(out_dir / "per_request_latency_sorted.png")
    plt.close()


def _format_summary(data: dict[str, Any]) -> str:
    def fmt(name: str, val: Any) -> str:
        return f"{name}: {val}"

    lines = [
        fmt("model", data.get("model")),
        fmt("num_prompts", data.get("num_prompts")),
        fmt("total_run_time_s", round(data.get("total_run_time_s", 0), 4)),
        fmt("throughput_total_tokens_per_s", round(data.get("throughput_total_tokens_per_s", 0), 2)),
        fmt("throughput_output_tokens_per_s", round(data.get("throughput_output_tokens_per_s", 0), 2)),
    ]

    prs = data.get("per_request_summary") or {}
    lat = prs.get("latency_s") or {}
    if lat:
        lines.append(
            fmt(
                "latency_s_p50/p95/p99",
                f"{lat.get('p50')}/{lat.get('p95')}/{lat.get('p99')}",
            )
        )

    cache_stats = data.get("cache_stats") or {}
    if cache_stats:
        lines.append(
            fmt(
                "cache_hit_rates",
                f"prefix={cache_stats.get('prefix_hit_rate')}, "
                f"external_prefix={cache_stats.get('external_prefix_hit_rate')}, "
                f"mm={cache_stats.get('mm_cache_hit_rate')}",
            )
        )

    gpu = data.get("gpu_stats") or {}
    if gpu:
        lines.append(
            fmt(
                "gpu_util/mem%",
                f"{gpu.get('gpu_util_percent')}% / {round(gpu.get('mem_util_percent', 0), 2)}%",
            )
        )
    return "\n".join(lines)


def _plot_throughput(data: dict[str, Any], out_dir: Path) -> None:
    vals = {
        "prompts/s": data.get("throughput_prompts_per_s"),
        "output_tokens/s": data.get("throughput_output_tokens_per_s"),
        "total_tokens/s": data.get("throughput_total_tokens_per_s"),
    }
    plt.figure(figsize=(6, 3))
    names = list(vals.keys())
    values = [vals[k] or 0 for k in names]
    plt.bar(names, values, color=["#4c72b0", "#55a868", "#c44e52"])
    plt.title("Throughput")
    plt.ylabel("Rate")
    plt.xlabel("Metric")
    plt.tight_layout()
    plt.savefig(out_dir / "throughput.png")
    plt.close()


def _plot_latency_percentiles(data: dict[str, Any], out_dir: Path) -> None:
    prs = data.get("per_request_summary") or {}
    lat = prs.get("latency_s") or {}
    if not lat:
        return
    labels = ["p50", "p90", "p95", "p99"]
    vals = [lat.get(k) for k in labels]
    plt.figure(figsize=(6, 3))
    plt.bar(labels, vals, color="#4c72b0")
    plt.title("Latency percentiles (per-request)")
    plt.ylabel("Seconds")
    plt.xlabel("Percentile")
    plt.tight_layout()
    plt.savefig(out_dir / "latency_percentiles.png")
    plt.close()


def render_all(data: dict[str, Any], base_out_dir: str | Path = "./output") -> Path:
    """Create plots and summary; returns the timestamped output directory."""
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = _ensure_dir(Path(base_out_dir) / timestamp)

    # Histograms from engine_metrics
    engine_metrics = data.get("engine_metrics") or []
    for metric in engine_metrics:
        if metric.get("type") == "histogram":
            _plot_hist(metric, out_path)

    # Per-request latency plot
    _plot_per_request_latencies(data.get("per_request") or [], out_path)

    # Throughput + latency percentiles
    _plot_throughput(data, out_path)
    _plot_latency_percentiles(data, out_path)

    # Write summary text
    summary = _format_summary(data)
    (out_path / "summary.txt").write_text(summary)
    return out_path
