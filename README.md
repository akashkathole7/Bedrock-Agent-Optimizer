# Bedrock Agent Optimizer

A lightweight caching and predictive routing layer for **Amazon Bedrock multi-agent** workflows. Reduces end-to-end latency by up to **30%** by pre-loading agent contexts and caching intermediate results with Redis.

---

## Problem

When chaining multiple agents in Amazon Bedrock (e.g., a classification agent -> retrieval agent -> summarization agent), each hop introduces latency from:

1. **Cold context assembly** - rebuilding conversation state for each downstream agent
2. **Redundant inference** - identical sub-queries hitting the model repeatedly
3. **Sequential blocking** - each agent waits for the previous one to fully complete

## Solution

This project introduces two optimizations:

### 1. Redis Caching Layer
Caches agent responses keyed on a normalized hash of (agent_id, input_payload, conversation_context). Configurable TTL ensures freshness while eliminating redundant calls.

### 2. Predictive Router
A lightweight classifier trained on historical agent-chain traces that predicts the next agent in the workflow. When confidence exceeds a threshold, it **pre-loads** the downstream agent's context in parallel with the current agent's execution.

```
┌─────────┐      ┌──────────────┐      ┌─────────┐
│ Request  │─────>│ Pred. Router │─────>│ Agent A │
└─────────┘      └──────┬───────┘      └────┬────┘
                        │                    │
                   (predict next)       (execute)
                        │                    │
                  ┌─────▼──────┐        ┌────▼────┐
                  │ Pre-load   │        │ Cache   │
                  │ Agent B ctx│        │ Result  │
                  └─────┬──────┘        └────┬────┘
                        │                    │
                        └───────┬────────────┘
                                │
                          ┌─────▼─────┐
                          │  Agent B  │  (context already warm)
                          └───────────┘
```

## Architecture

```
Bedrock-Agent-Optimizer/
├── src/
│   ├── __init__.py
│   ├── cache.py           # Redis caching layer
│   ├── router.py          # Predictive routing logic
│   ├── agent_wrapper.py   # Bedrock agent invocation wrapper
│   └── config.py          # Configuration management
├── benchmarks/
│   └── latency_bench.py   # Benchmark script with results
├── tests/
│   └── test_cache.py      # Unit tests
├── requirements.txt
└── README.md
```

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Ensure Redis is running
redis-server

# Run the benchmark
python benchmarks/latency_bench.py

# Run with your own agent chain
python -m src.agent_wrapper --agents classifier,retriever,summarizer --input "your query"
```

## Configuration

Set environment variables or use a `.env` file:

```env
REDIS_HOST=localhost
REDIS_PORT=6379
CACHE_TTL_SECONDS=300
PREDICTION_CONFIDENCE_THRESHOLD=0.7
AWS_REGION=us-east-1
BEDROCK_AGENT_IDS=agent1-id,agent2-id,agent3-id
```

## Benchmark Results

Tested on a 3-agent chain (classify -> retrieve -> summarize) with 200 sequential requests:

| Metric                | Baseline  | With Optimizer | Improvement |
|-----------------------|-----------|----------------|-------------|
| Avg latency (p50)     | 2.4s      | 1.7s           | **29%**     |
| Tail latency (p99)    | 4.1s      | 2.8s           | **32%**     |
| Cache hit rate         | N/A       | 41%            | -           |
| Prediction accuracy    | N/A       | 78%            | -           |

## How It Works

1. **Request arrives** - The optimizer hashes the input and checks Redis for a cached response.
2. **Cache miss** - If not cached, the request goes to the first Bedrock agent. Simultaneously, the predictive router guesses the next agent and begins pre-loading its context.
3. **Agent responds** - The response is cached in Redis with the configured TTL.
4. **Chain continues** - The next agent starts with a warm context (if prediction was correct) or cold-starts as usual (if prediction was wrong). No penalty for mispredictions.

## Limitations

- Cache invalidation relies on TTL; no event-driven invalidation yet
- Predictive router requires ~50 traces of historical data to reach useful accuracy
- Redis is a single point of failure (mitigated with Redis Sentinel in production)

## License

MIT
