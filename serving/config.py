import os
from dataclasses import dataclass, field


@dataclass
class Settings:
    backend: str = field(default_factory=lambda: os.getenv("BACKEND", "mock"))
    model: str = field(default_factory=lambda: os.getenv("MODEL", "Qwen/Qwen2.5-7B-Instruct"))
    urls: list[str] = field(
        default_factory=lambda: os.getenv(
            "VLLM_URLS", "http://localhost:8001,http://localhost:8002"
        ).split(",")
    )
    redis_url: str = field(
        default_factory=lambda: os.getenv("REDIS_URL", "redis://localhost:6379/0")
    )
    api_key: str = field(default_factory=lambda: os.getenv("API_KEY", ""))
    timeout: float = field(default_factory=lambda: float(os.getenv("REQUEST_TIMEOUT", "60")))
    max_inflight: int = field(default_factory=lambda: int(os.getenv("MAX_INFLIGHT", "32")))
    queue_capacity: int = field(default_factory=lambda: int(os.getenv("QUEUE_CAPACITY", "1000")))
    worker_concurrency: int = field(
        default_factory=lambda: int(os.getenv("WORKER_CONCURRENCY", "8"))
    )
    lease_seconds: int = field(default_factory=lambda: int(os.getenv("LEASE_SECONDS", "90")))
    result_ttl: int = field(default_factory=lambda: int(os.getenv("RESULT_TTL", "86400")))
    max_attempts: int = 3

    def __post_init__(self):
        if self.backend not in {"mock", "vllm"}:
            raise ValueError("BACKEND must be mock or vllm")
        if (
            min(
                self.timeout,
                self.max_inflight,
                self.queue_capacity,
                self.worker_concurrency,
                self.result_ttl,
            )
            <= 0
        ):
            raise ValueError("Limits must be positive")
        if self.lease_seconds <= self.timeout + 5:
            raise ValueError("LEASE_SECONDS must exceed REQUEST_TIMEOUT by more than 5 seconds")
