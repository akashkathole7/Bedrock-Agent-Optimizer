"""Redis caching layer for Bedrock agent responses."""

import hashlib
import json
import logging

import redis

from .config import OptimizerConfig

logger = logging.getLogger(__name__)


class AgentResponseCache:
    """Caches agent responses keyed on (agent_id, input, context) hash."""

    def __init__(self, config: OptimizerConfig):
        self.config = config
        self.client = redis.Redis(
            host=config.redis_host,
            port=config.redis_port,
            decode_responses=True,
        )
        self._hits = 0
        self._misses = 0

    @staticmethod
    def _build_key(agent_id: str, payload: dict, context: dict | None = None) -> str:
        raw = json.dumps(
            {"agent_id": agent_id, "payload": payload, "context": context or {}},
            sort_keys=True,
        )
        return f"bedrock_cache:{hashlib.sha256(raw.encode()).hexdigest()}"

    def get(self, agent_id: str, payload: dict, context: dict | None = None) -> dict | None:
        key = self._build_key(agent_id, payload, context)
        cached = self.client.get(key)
        if cached:
            self._hits += 1
            logger.debug("Cache HIT for agent %s (key=%s)", agent_id, key[:16])
            return json.loads(cached)
        self._misses += 1
        logger.debug("Cache MISS for agent %s (key=%s)", agent_id, key[:16])
        return None

    def put(self, agent_id: str, payload: dict, response: dict, context: dict | None = None):
        key = self._build_key(agent_id, payload, context)
        self.client.setex(key, self.config.cache_ttl, json.dumps(response))
        logger.debug("Cached response for agent %s (ttl=%ds)", agent_id, self.config.cache_ttl)

    @property
    def hit_rate(self) -> float:
        total = self._hits + self._misses
        return self._hits / total if total > 0 else 0.0

    def stats(self) -> dict:
        return {
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": round(self.hit_rate, 3),
        }
