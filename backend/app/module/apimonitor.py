# LLM Assistant - API Monitoring
# Migrated into module package
import time
import dataclasses
from typing import Dict, Any

@dataclasses.dataclass
class APIMetrics:
    requests_total: int = 0
    requests_failed: int = 0
    requests_success: int = 0
    total_latency_ms: float = 0.0
    last_request_time: float = 0.0
    start_time: float = dataclasses.field(default_factory=time.time)

    @property
    def avg_latency_ms(self) -> float:
        if self.requests_total == 0:
            return 0.0
        return self.total_latency_ms / self.requests_total

    @property
    def uptime_seconds(self) -> float:
        return time.time() - self.start_time

class APIMonitor:
    _instance = None
    metrics: APIMetrics

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(APIMonitor, cls).__new__(cls)
            cls._instance.metrics = APIMetrics()
        return cls._instance

    def record_request(self, latency_ms: float, success: bool = True):
        self.metrics.requests_total += 1
        self.metrics.total_latency_ms += latency_ms
        self.metrics.last_request_time = time.time()

        if success:
            self.metrics.requests_success += 1
        else:
            self.metrics.requests_failed += 1

    def get_stats(self) -> Dict[str, Any]:
        return {
            "uptime_seconds": round(self.metrics.uptime_seconds, 2),
            "requests": {
                "total": self.metrics.requests_total,
                "success": self.metrics.requests_success,
                "failed": self.metrics.requests_failed,
            },
            "performance": {
                "avg_latency_ms": round(self.metrics.avg_latency_ms, 2),
            }
        }

    def get_health_summary(self) -> Dict[str, Any]:
        stats = self.get_stats()
        return {
            "metrics": stats,
            "api": {
                "status": "healthy",
                "last_request_time": self.metrics.last_request_time,
            },
        }

monitor = APIMonitor()