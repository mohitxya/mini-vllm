from __future__ import annotations

import argparse
import statistics
import time
from dataclasses import dataclass

import torch

from mini_vllm.backends.hf_backend import HFBackend
from mini_vllm.runtime.continuous_batch_scheduler import ContinuousBatchScheduler
from mini_vllm.runtime.naive_runtime import NaiveRuntime
from mini_vllm.runtime.request import GenerationRequest
from mini_vllm.runtime.round_robin_scheduler import RoundRobinScheduler
from mini_vllm.runtime.sampling import SamplingConfig, SamplingStrategy


# ---------------------------------------------------------------------
# Benchmark result containers
# ---------------------------------------------------------------------


@dataclass
class BenchmarkResult:
    """
    Stores benchmark metrics for one runtime mode.
    """

    name: str
    total_time_seconds: float
    num_requests: int
    completed_requests: int
    failed_requests: int
    total_generated_tokens: int
    tokens_per_second: float
    avg_latency_seconds: float
    p50_latency_seconds: float
    p95_latency_seconds: float
    avg_wait_time_seconds: float
    avg_runtime_seconds: float
    avg_batch_size: float | None = None
    batch_steps: int | None = None


# ---------------------------------------------------------------------
# Request creation
# ---------------------------------------------------------------------


def make_prompts() -> list[str]:
    """
    Fixed prompt set for repeatable benchmarking.
    """

    return [
        "In simple terms, a GPU is",
        "The reason KV cache speeds up inference is",
        "A model serving runtime decides",
        "Continuous batching improves throughput by",
        "A transformer model predicts the next token by",
        "In machine learning, inference means",
    ]


def make_requests(
    max_new_tokens: int,
    sampling_config: SamplingConfig,
) -> list[GenerationRequest]:
    """
    Create fresh request objects.

    Important:
        Do NOT reuse request objects across benchmark runs.

    A GenerationRequest stores mutable state:
        - generated_ids
        - past_key_values
        - status
        - timestamps
        - generated token count
    """

    return [
        GenerationRequest(
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            sampling_config=sampling_config,
        )
        for prompt in make_prompts()
    ]


# ---------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------


def sync_if_cuda(device: str | None) -> None:
    """
    CUDA operations can be asynchronous.

    If timing GPU code, synchronize before/after timing regions so measured
    wall-clock time includes actual GPU execution.
    """

    if device == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize()


def print_cuda_memory(device: str | None, label: str) -> None:
    """
    Print CUDA memory usage if running on GPU.
    """

    if device == "cuda" and torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / (1024**2)
        reserved = torch.cuda.memory_reserved() / (1024**2)

        print(f"\n================ CUDA Memory: {label} ================")
        print(f"Allocated: {allocated:.2f} MiB")
        print(f"Reserved:  {reserved:.2f} MiB")
        print("======================================================\n")


def percentile(values: list[float], p: float) -> float:
    """
    Compute percentile using simple linear interpolation.
    """

    if not values:
        return 0.0

    sorted_values = sorted(values)

    if len(sorted_values) == 1:
        return sorted_values[0]

    rank = (p / 100) * (len(sorted_values) - 1)
    lower = int(rank)
    upper = min(lower + 1, len(sorted_values) - 1)

    weight = rank - lower

    return sorted_values[lower] * (1 - weight) + sorted_values[upper] * weight


def collect_common_metrics(
    name: str,
    total_time_seconds: float,
    requests: list[GenerationRequest],
    avg_batch_size: float | None = None,
    batch_steps: int | None = None,
) -> BenchmarkResult:
    """
    Convert processed request objects into benchmark metrics.
    """

    completed = [
        request for request in requests
        if request.status.value == "FINISHED"
    ]

    failed = [
        request for request in requests
        if request.status.value == "FAILED"
    ]

    latencies = [
        request.total_time_seconds or 0.0
        for request in completed
    ]

    wait_times = [
        request.wait_time_seconds or 0.0
        for request in completed
    ]

    runtimes = [
        request.runtime_seconds or 0.0
        for request in completed
    ]

    total_generated_tokens = sum(
        request.num_generated_tokens
        for request in completed
    )

    tokens_per_second = (
        total_generated_tokens / total_time_seconds
        if total_time_seconds > 0
        else 0.0
    )

    return BenchmarkResult(
        name=name,
        total_time_seconds=total_time_seconds,
        num_requests=len(requests),
        completed_requests=len(completed),
        failed_requests=len(failed),
        total_generated_tokens=total_generated_tokens,
        tokens_per_second=tokens_per_second,
        avg_latency_seconds=statistics.mean(latencies) if latencies else 0.0,
        p50_latency_seconds=percentile(latencies, 50),
        p95_latency_seconds=percentile(latencies, 95),
        avg_wait_time_seconds=statistics.mean(wait_times) if wait_times else 0.0,
        avg_runtime_seconds=statistics.mean(runtimes) if runtimes else 0.0,
        avg_batch_size=avg_batch_size,
        batch_steps=batch_steps,
    )


# ---------------------------------------------------------------------
# Runtime benchmark functions
# ---------------------------------------------------------------------


def benchmark_naive_runtime(
    backend: HFBackend,
    max_new_tokens: int,
    sampling_config: SamplingConfig,
) -> BenchmarkResult:
    """
    Benchmark NaiveRuntime.

    Execution pattern:
        request A fully
        then request B fully
        then request C fully
    """

    requests = make_requests(
        max_new_tokens=max_new_tokens,
        sampling_config=sampling_config,
    )

    runtime = NaiveRuntime(backend=backend)
    runtime.submit_many(requests)

    sync_if_cuda(backend.device)
    start = time.perf_counter()

    runtime.run_until_empty()

    sync_if_cuda(backend.device)
    total_time = time.perf_counter() - start

    return collect_common_metrics(
        name="naive_sequential",
        total_time_seconds=total_time,
        requests=runtime.all_requests(),
    )


def benchmark_round_robin(
    backend: HFBackend,
    max_new_tokens: int,
    sampling_config: SamplingConfig,
) -> BenchmarkResult:
    """
    Benchmark RoundRobinScheduler.

    Execution pattern:
        prefill A, B, C
        decode A
        decode B
        decode C
        repeat

    Uses KV cache via backend.decode_one().
    """

    requests = make_requests(
        max_new_tokens=max_new_tokens,
        sampling_config=sampling_config,
    )

    scheduler = RoundRobinScheduler(backend=backend)
    scheduler.submit_many(requests)

    sync_if_cuda(backend.device)
    start = time.perf_counter()

    scheduler.run_until_done(debug=False)

    sync_if_cuda(backend.device)
    total_time = time.perf_counter() - start

    return collect_common_metrics(
        name="round_robin_kv_cache",
        total_time_seconds=total_time,
        requests=scheduler.all_requests(),
    )


def benchmark_continuous_batch(
    backend: HFBackend,
    max_new_tokens: int,
    sampling_config: SamplingConfig,
    max_batch_size: int,
) -> BenchmarkResult:
    """
    Benchmark ContinuousBatchScheduler.

    Execution pattern:
        [A, B, C] one batched forward step
        [A, B, C] one batched forward step
        [B, C, D] one batched forward step
        ...

    Simplified implementation:
        Uses full-sequence recomputation, not KV-cache batching.
    """

    requests = make_requests(
        max_new_tokens=max_new_tokens,
        sampling_config=sampling_config,
    )

    scheduler = ContinuousBatchScheduler(
        backend=backend,
        max_batch_size=max_batch_size,
    )

    scheduler.submit_many(requests)

    sync_if_cuda(backend.device)
    start = time.perf_counter()

    scheduler.run_until_done(debug=False)

    sync_if_cuda(backend.device)
    total_time = time.perf_counter() - start

    avg_batch_size = (
        sum(scheduler.batch_sizes_seen) / len(scheduler.batch_sizes_seen)
        if scheduler.batch_sizes_seen
        else 0.0
    )

    return collect_common_metrics(
        name=f"continuous_batch_recompute_bs{max_batch_size}",
        total_time_seconds=total_time,
        requests=scheduler.all_requests(),
        avg_batch_size=avg_batch_size,
        batch_steps=scheduler.batch_steps,
    )


# ---------------------------------------------------------------------
# Printing
# ---------------------------------------------------------------------


def print_result_table(results: list[BenchmarkResult]) -> None:
    """
    Print benchmark results as a simple terminal table.
    """

    print("\n================ Benchmark Results ================")

    header = (
        f"{'Runtime':<34} "
        f"{'Time(s)':>9} "
        f"{'Req':>5} "
        f"{'Done':>5} "
        f"{'Fail':>5} "
        f"{'Tok':>6} "
        f"{'Tok/s':>9} "
        f"{'AvgLat':>9} "
        f"{'P50':>9} "
        f"{'P95':>9} "
        f"{'AvgWait':>9} "
        f"{'AvgRun':>9} "
        f"{'AvgBS':>7}"
    )

    print(header)
    print("-" * len(header))

    for result in results:
        avg_bs = (
            f"{result.avg_batch_size:.2f}"
            if result.avg_batch_size is not None
            else "-"
        )

        print(
            f"{result.name:<34} "
            f"{result.total_time_seconds:>9.3f} "
            f"{result.num_requests:>5} "
            f"{result.completed_requests:>5} "
            f"{result.failed_requests:>5} "
            f"{result.total_generated_tokens:>6} "
            f"{result.tokens_per_second:>9.2f} "
            f"{result.avg_latency_seconds:>9.3f} "
            f"{result.p50_latency_seconds:>9.3f} "
            f"{result.p95_latency_seconds:>9.3f} "
            f"{result.avg_wait_time_seconds:>9.3f} "
            f"{result.avg_runtime_seconds:>9.3f} "
            f"{avg_bs:>7}"
        )

    print("===================================================\n")


def print_interpretation() -> None:
    """
    Print a short explanation of how to interpret results.
    """

    print("\n================ How to Interpret ================")

    print(
        """
Naive sequential:
    Good baseline.
    Processes each request fully before moving to the next.

Round-robin KV cache:
    Interleaves requests token-by-token.
    Uses KV cache, so each decode step processes only one new token per request.
    Improves fairness, but not necessarily throughput.

Continuous batch recompute:
    Processes multiple active requests in one batched forward pass.
    Teaches batching and dynamic active sets.
    But it recomputes full sequences every step, so it may be slower than KV-cache decoding.

Tokens/sec:
    Higher is better.

Average latency:
    Average total lifetime of a request.

P95 latency:
    Tail latency. This matters in serving systems because users notice slow outliers.

Average wait:
    Time spent waiting before generation starts.

Average runtime:
    Time spent actively being processed.
"""
    )

    print("===================================================\n")


# ---------------------------------------------------------------------
# Main CLI
# ---------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark mini-vLLM runtimes."
    )

    parser.add_argument(
        "--model",
        type=str,
        default="distilgpt2",
        help="Hugging Face model name.",
    )

    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use: cpu or cuda. Defaults to auto-detect.",
    )

    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=16,
        help="Maximum new tokens per request.",
    )

    parser.add_argument(
        "--max-batch-size",
        type=int,
        default=2,
        help="Max batch size for continuous batching.",
    )

    parser.add_argument(
        "--strategy",
        type=str,
        default="greedy",
        choices=["greedy", "temperature", "top_k"],
        help="Sampling strategy.",
    )

    parser.add_argument(
        "--temperature",
        type=float,
        default=0.8,
        help="Sampling temperature.",
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=50,
        help="Top-k value for top-k sampling.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print("\n================ Benchmark Config ================")
    print(f"Model:          {args.model}")
    print(f"Requested dev:  {args.device}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"Max new tokens: {args.max_new_tokens}")
    print(f"Max batch size: {args.max_batch_size}")
    print(f"Strategy:       {args.strategy}")
    print(f"Temperature:    {args.temperature}")
    print(f"Top-k:          {args.top_k}")
    print("==================================================\n")

    strategy = SamplingStrategy(args.strategy)

    sampling_config = SamplingConfig(
        strategy=strategy,
        temperature=args.temperature,
        top_k=args.top_k,
        seed=None,
    )

    backend = HFBackend(
        model_name=args.model,
        device=args.device,
    )

    print_cuda_memory(backend.device, label="after model load")

    # Warm-up.
    # This avoids measuring one-time overhead too heavily.
    print("\nRunning warm-up...")

    sync_if_cuda(backend.device)

    backend.generate_one(
        prompt="Warm up the model by generating",
        max_new_tokens=4,
    )

    sync_if_cuda(backend.device)

    print("Warm-up done.")

    results: list[BenchmarkResult] = []

    results.append(
        benchmark_naive_runtime(
            backend=backend,
            max_new_tokens=args.max_new_tokens,
            sampling_config=sampling_config,
        )
    )

    results.append(
        benchmark_round_robin(
            backend=backend,
            max_new_tokens=args.max_new_tokens,
            sampling_config=sampling_config,
        )
    )

    results.append(
        benchmark_continuous_batch(
            backend=backend,
            max_new_tokens=args.max_new_tokens,
            sampling_config=sampling_config,
            max_batch_size=args.max_batch_size,
        )
    )

    print_cuda_memory(backend.device, label="after benchmarks")

    print_result_table(results)
    print_interpretation()


if __name__ == "__main__":
    main()