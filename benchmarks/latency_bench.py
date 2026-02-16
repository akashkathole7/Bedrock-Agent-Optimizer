"""Benchmark: Compare baseline vs optimized multi-agent chain latency.

This benchmark simulates agent invocations with realistic latency profiles
to demonstrate the impact of caching and predictive routing.
"""

import random
import statistics
import time
from unittest.mock import MagicMock, patch

from src.agent_wrapper import BedrockAgentOptimizer
from src.config import OptimizerConfig


# Simulated agent latencies (seconds)
AGENT_LATENCY = {
    "classifier": (0.3, 0.6),
    "retriever": (0.8, 1.2),
    "summarizer": (0.5, 0.9),
}

AGENT_CHAIN = ["classifier", "retriever", "summarizer"]
NUM_REQUESTS = 200
REPEAT_QUERY_RATE = 0.4  # 40% of queries are repeats (cache-friendly)


def simulate_agent_call(agent_id: str, *args, **kwargs):
    """Simulate a Bedrock agent call with realistic latency."""
    lo, hi = AGENT_LATENCY.get(agent_id, (0.3, 0.8))
    time.sleep(random.uniform(lo, hi))
    return {
        "completion": [
            {"chunk": {"bytes": f"Response from {agent_id}".encode()}}
        ]
    }


def generate_queries(n: int) -> list[str]:
    """Generate a mix of unique and repeated queries."""
    unique = [f"Query about topic {i}" for i in range(int(n * (1 - REPEAT_QUERY_RATE)))]
    repeated = random.choices(unique, k=n - len(unique))
    queries = unique + repeated
    random.shuffle(queries)
    return queries


def run_baseline(queries: list[str]) -> list[float]:
    """Run agents sequentially without any optimization."""
    latencies = []
    for query in queries:
        start = time.monotonic()
        for agent_id in AGENT_CHAIN:
            simulate_agent_call(agent_id)
        elapsed = time.monotonic() - start
        latencies.append(elapsed)
    return latencies


def run_optimized(queries: list[str]) -> list[float]:
    """Run agents through the optimizer with caching and prediction."""
    config = OptimizerConfig()
    optimizer = BedrockAgentOptimizer(config)

    # Seed the router with some historical traces
    for _ in range(50):
        optimizer.router.ingest_trace(AGENT_CHAIN)

    latencies = []
    with patch.object(optimizer._bedrock, "invoke_agent", side_effect=simulate_agent_call):
        for query in queries:
            start = time.monotonic()
            optimizer.run_chain(AGENT_CHAIN, query)
            elapsed = time.monotonic() - start
            latencies.append(elapsed)

    print(f"  Cache stats: {optimizer.cache.stats()}")
    print(f"  Router stats: {optimizer.router.stats()}")
    return latencies


def report(label: str, latencies: list[float]):
    latencies_ms = [l * 1000 for l in latencies]
    sorted_l = sorted(latencies_ms)
    p50 = sorted_l[len(sorted_l) // 2]
    p99 = sorted_l[int(len(sorted_l) * 0.99)]
    print(f"\n{'='*50}")
    print(f"  {label}")
    print(f"{'='*50}")
    print(f"  Requests:    {len(latencies)}")
    print(f"  Avg (ms):    {statistics.mean(latencies_ms):.1f}")
    print(f"  p50 (ms):    {p50:.1f}")
    print(f"  p99 (ms):    {p99:.1f}")
    print(f"  Std dev:     {statistics.stdev(latencies_ms):.1f}")


def main():
    print("Generating queries...")
    queries = generate_queries(NUM_REQUESTS)

    print(f"\nRunning BASELINE ({NUM_REQUESTS} requests, {len(AGENT_CHAIN)}-agent chain)...")
    baseline = run_baseline(queries)
    report("Baseline (no optimization)", baseline)

    print(f"\nRunning OPTIMIZED ({NUM_REQUESTS} requests)...")
    optimized = run_optimized(queries)
    report("Optimized (cache + predictive routing)", optimized)

    # Comparison
    baseline_avg = statistics.mean(baseline)
    optimized_avg = statistics.mean(optimized)
    improvement = ((baseline_avg - optimized_avg) / baseline_avg) * 100
    print(f"\n>>> Improvement: {improvement:.1f}% average latency reduction <<<")


if __name__ == "__main__":
    main()
