"""Wrapper around Bedrock agent invocation with caching and predictive pre-loading."""

import argparse
import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor, Future

import boto3

from .cache import AgentResponseCache
from .config import OptimizerConfig
from .router import PredictiveRouter

logger = logging.getLogger(__name__)


class BedrockAgentOptimizer:
    """Orchestrates multi-agent Bedrock calls with caching + predictive routing."""

    def __init__(self, config: OptimizerConfig | None = None):
        self.config = config or OptimizerConfig()
        self.cache = AgentResponseCache(self.config)
        self.router = PredictiveRouter(self.config)
        self._bedrock = boto3.client(
            "bedrock-agent-runtime", region_name=self.config.aws_region
        )
        self._executor = ThreadPoolExecutor(max_workers=4)
        self._preloaded: dict[str, Future] = {}

    def _invoke_agent(self, agent_id: str, payload: dict, context: dict | None = None) -> dict:
        """Call a single Bedrock agent, using cache if available."""
        cached = self.cache.get(agent_id, payload, context)
        if cached:
            return cached

        response = self._bedrock.invoke_agent(
            agentId=agent_id,
            agentAliasId="TSTALIASID",
            sessionId=payload.get("session_id", "default"),
            inputText=payload.get("input_text", ""),
        )

        # Collect the streamed response
        completion = ""
        for event in response.get("completion", []):
            chunk = event.get("chunk", {})
            completion += chunk.get("bytes", b"").decode("utf-8")

        result = {
            "agent_id": agent_id,
            "output": completion,
            "timestamp": time.time(),
        }

        self.cache.put(agent_id, payload, result, context)
        return result

    def _preload_context(self, agent_id: str, context: dict):
        """Pre-warm the next agent's context in a background thread."""
        logger.info("Pre-loading context for agent %s", agent_id)
        # Pre-warming: send a lightweight probe to initialize the session
        try:
            self._bedrock.invoke_agent(
                agentId=agent_id,
                agentAliasId="TSTALIASID",
                sessionId=context.get("session_id", "preload"),
                inputText="[PRELOAD] Initialize session context.",
            )
        except Exception as e:
            logger.warning("Preload failed for %s: %s", agent_id, e)

    def run_chain(self, agent_ids: list[str], input_text: str) -> list[dict]:
        """Execute a chain of agents with optimization.

        Args:
            agent_ids: Ordered list of Bedrock agent IDs to invoke.
            input_text: Initial user input.

        Returns:
            List of agent responses in chain order.
        """
        results = []
        current_input = input_text
        context = {"session_id": f"chain-{int(time.time())}"}

        for i, agent_id in enumerate(agent_ids):
            payload = {"input_text": current_input, "session_id": context["session_id"]}

            # Check if we already pre-loaded this agent
            if agent_id in self._preloaded:
                logger.info("Agent %s was pre-loaded, expecting warm start", agent_id)
                try:
                    self._preloaded[agent_id].result(timeout=1)
                except Exception:
                    pass  # Pre-load is best-effort
                del self._preloaded[agent_id]

            # Execute current agent
            start = time.monotonic()
            result = self._invoke_agent(agent_id, payload, context)
            elapsed = time.monotonic() - start
            result["latency_ms"] = round(elapsed * 1000, 1)
            results.append(result)

            # Feed output as input to next agent
            current_input = result.get("output", "")

            # Predictive pre-loading for the next agent
            if i < len(agent_ids) - 1:
                self.router.record_transition(agent_id, agent_ids[i + 1])
                predicted = self.router.should_preload(agent_id)
                if predicted:
                    future = self._executor.submit(
                        self._preload_context, predicted, context
                    )
                    self._preloaded[predicted] = future

        return results

    def report(self, results: list[dict]) -> dict:
        """Generate a summary report from a chain execution."""
        latencies = [r["latency_ms"] for r in results]
        return {
            "total_latency_ms": round(sum(latencies), 1),
            "per_agent": [
                {"agent": r["agent_id"], "latency_ms": r["latency_ms"]}
                for r in results
            ],
            "cache": self.cache.stats(),
            "router": self.router.stats(),
        }


def main():
    parser = argparse.ArgumentParser(description="Run an optimized Bedrock agent chain")
    parser.add_argument("--agents", required=True, help="Comma-separated agent IDs")
    parser.add_argument("--input", required=True, help="Input text for the chain")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    agent_ids = [a.strip() for a in args.agents.split(",")]

    optimizer = BedrockAgentOptimizer()
    results = optimizer.run_chain(agent_ids, args.input)
    report = optimizer.report(results)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
