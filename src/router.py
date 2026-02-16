"""Predictive router that anticipates the next agent in a chain."""

import logging
from collections import Counter, defaultdict

from .config import OptimizerConfig

logger = logging.getLogger(__name__)


class PredictiveRouter:
    """Learns agent transition patterns and predicts the next agent to pre-warm.

    Uses a simple bigram frequency model: given the current agent, predict
    which agent is most likely to be invoked next based on historical traces.
    """

    def __init__(self, config: OptimizerConfig):
        self.config = config
        # transition_counts[current_agent][next_agent] = count
        self._transitions: dict[str, Counter] = defaultdict(Counter)
        self._total_traces = 0

    def record_transition(self, from_agent: str, to_agent: str):
        """Record an observed agent-to-agent transition."""
        self._transitions[from_agent][to_agent] += 1
        self._total_traces += 1

    def ingest_trace(self, agent_sequence: list[str]):
        """Ingest a full agent chain trace (ordered list of agent IDs)."""
        for i in range(len(agent_sequence) - 1):
            self.record_transition(agent_sequence[i], agent_sequence[i + 1])

    def predict_next(self, current_agent: str) -> tuple[str | None, float]:
        """Predict the most likely next agent given the current one.

        Returns:
            (predicted_agent_id, confidence) or (None, 0.0) if unknown.
        """
        counts = self._transitions.get(current_agent)
        if not counts:
            return None, 0.0

        total = sum(counts.values())
        best_agent, best_count = counts.most_common(1)[0]
        confidence = best_count / total

        if confidence >= self.config.prediction_threshold:
            logger.info(
                "Predicting next agent: %s (confidence=%.2f)", best_agent, confidence
            )
            return best_agent, confidence

        logger.debug(
            "Low confidence prediction for %s -> %s (%.2f < %.2f threshold)",
            current_agent,
            best_agent,
            confidence,
            self.config.prediction_threshold,
        )
        return None, confidence

    def should_preload(self, current_agent: str) -> str | None:
        """Return agent ID to preload, or None if confidence is too low."""
        agent_id, confidence = self.predict_next(current_agent)
        return agent_id if agent_id else None

    def stats(self) -> dict:
        return {
            "total_traces": self._total_traces,
            "known_agents": list(self._transitions.keys()),
            "transition_pairs": sum(
                len(v) for v in self._transitions.values()
            ),
        }
