"""Unit tests for the caching layer and predictive router."""

import unittest
from unittest.mock import MagicMock, patch

from src.cache import AgentResponseCache
from src.config import OptimizerConfig
from src.router import PredictiveRouter


class TestAgentResponseCache(unittest.TestCase):
    def setUp(self):
        self.config = OptimizerConfig()
        with patch("src.cache.redis.Redis"):
            self.cache = AgentResponseCache(self.config)
            self.cache.client = MagicMock()

    def test_cache_miss_returns_none(self):
        self.cache.client.get.return_value = None
        result = self.cache.get("agent-1", {"input_text": "hello"})
        self.assertIsNone(result)
        self.assertEqual(self.cache._misses, 1)

    def test_cache_hit_returns_data(self):
        self.cache.client.get.return_value = '{"output": "cached"}'
        result = self.cache.get("agent-1", {"input_text": "hello"})
        self.assertEqual(result, {"output": "cached"})
        self.assertEqual(self.cache._hits, 1)

    def test_put_stores_with_ttl(self):
        self.cache.put("agent-1", {"input_text": "hello"}, {"output": "world"})
        self.cache.client.setex.assert_called_once()
        args = self.cache.client.setex.call_args
        self.assertEqual(args[0][1], self.config.cache_ttl)

    def test_hit_rate_calculation(self):
        self.cache._hits = 3
        self.cache._misses = 7
        self.assertAlmostEqual(self.cache.hit_rate, 0.3)


class TestPredictiveRouter(unittest.TestCase):
    def setUp(self):
        self.config = OptimizerConfig(prediction_threshold=0.7)
        self.router = PredictiveRouter(self.config)

    def test_unknown_agent_returns_none(self):
        agent, conf = self.router.predict_next("unknown")
        self.assertIsNone(agent)
        self.assertEqual(conf, 0.0)

    def test_high_confidence_prediction(self):
        for _ in range(10):
            self.router.record_transition("A", "B")
        agent, conf = self.router.predict_next("A")
        self.assertEqual(agent, "B")
        self.assertEqual(conf, 1.0)

    def test_low_confidence_returns_none(self):
        self.router.record_transition("A", "B")
        self.router.record_transition("A", "C")
        self.router.record_transition("A", "D")
        agent, conf = self.router.predict_next("A")
        self.assertIsNone(agent)

    def test_ingest_trace(self):
        self.router.ingest_trace(["X", "Y", "Z"])
        agent, _ = self.router.predict_next("X")
        self.assertEqual(agent, "Y")
        agent, _ = self.router.predict_next("Y")
        self.assertEqual(agent, "Z")


if __name__ == "__main__":
    unittest.main()
