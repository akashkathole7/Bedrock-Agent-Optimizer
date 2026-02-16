import os
from dataclasses import dataclass, field


@dataclass
class OptimizerConfig:
    redis_host: str = os.getenv("REDIS_HOST", "localhost")
    redis_port: int = int(os.getenv("REDIS_PORT", "6379"))
    cache_ttl: int = int(os.getenv("CACHE_TTL_SECONDS", "300"))
    prediction_threshold: float = float(
        os.getenv("PREDICTION_CONFIDENCE_THRESHOLD", "0.7")
    )
    aws_region: str = os.getenv("AWS_REGION", "us-east-1")
    agent_ids: list[str] = field(default_factory=lambda: os.getenv(
        "BEDROCK_AGENT_IDS", ""
    ).split(","))
