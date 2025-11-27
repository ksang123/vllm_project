# vLLM Inference Benchmark Harness

A lightweight description of the benchmark harness for exercising vLLM under realistic traffic. This README focuses on intent and flow so you can see how the pieces fit together without wading through code or config.

---

## What This Harness Does

-   **Plug-and-Play vLLM Inference Tool**: This harness serves as a plug-and-play tool for benchmarking vLLM inference.
-   **NVIDIA Nsight Systems Integration**: It leverages NVIDIA's external Nsight Systems to collect detailed performance telemetry.
-   **Realistic Workloads**: Benchmarks vLLM (and forks) on realistic workloads, including bursty, long-context, and mixed patterns.
-   **Automated Prompt Generation**: Generates approximately 50 random prompts for each run to simulate diverse traffic.
-   **Model-Agnostic Benchmarking**: While models are pre-loaded within vLLM, the harness can be configured to target specific models for benchmarking.
-   **Nsight Web API Calls**: Makes API calls to Nsight Web to retrieve and analyze performance results.
-   **Fair Comparisons**: Ensures identical scenarios for baseline vs. optimized branches, allowing for fair comparisons.
-   **Performance Insight**: Collects telemetry that explains *why* performance shifts, not just headline tokens/sec.

---

## How It Works



-   **Scenario Definition and Selection**: A single configuration file allows the user to define and select a scenario, which describes the target model, the mix of workloads, the request arrival pattern, and the duration of the run. This allows users to easily pick and switch between various testing conditions.

-   **Runner**: Orchestrates vLLM locally or via an HTTP endpoint, drives warm-up and measurement phases, and tags each run with metadata.

-   **Telemetry**: Hooks into vLLM stats plus system metrics (GPU/CPU, memory, latency milestones) so every request is traceable, leveraging Nsight Systems for deep profiling.

-   **Analysis**: Converts raw logs into tables and “hockey-stick” plots for throughput and latency across concurrency levels, with data gathered via Nsight Web API calls.



The parts stay modular so you can swap workloads, engines, or telemetry without rewriting the harness.

---

## Using It

1) Pick a scenario: set the model, traffic pattern, and durations in the config file.  
2) Run the harness: execute the runner to produce timestamped run folders with logs and metadata.  
3) Analyze: feed run folders into the analysis script to generate summary tables and plots.  
4) Compare: run baseline and optimized branches with the same scenario, then view side-by-side charts.

---

## What You Get Out

- Per-request metrics (arrival, first token, completion, token counts, status).
- System signals (GPU/CPU utilization and memory) aligned with request timelines.
- Aggregated reports: throughput, latency percentiles, error rates, and concurrency “hockey-stick” curves.
- Reproducible artifacts: config snapshots and environment details alongside results.

---

## Extending the Harness

- Add new traffic patterns or request templates to mirror production mixes.
- Point to different vLLM builds or engine variants to test scheduling and KV-cache tweaks.
- Enrich telemetry with custom probes; extend analysis with new visualizations or metrics.

---

## Minimum Setup Expectations

- Recent Python with CUDA-enabled PyTorch and vLLM installed.
- Access to suitable GPUs (e.g., dual L40S or comparable).
- NVML or similar tools if you want richer hardware telemetry.
