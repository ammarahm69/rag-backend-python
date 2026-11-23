import time
import logging
from typing import Dict, Optional
from contextlib import contextmanager

logger = logging.getLogger(__name__)

class LatencyTracer:
    def __init__(self):
        self.timings: Dict[str, float] = {}

    @contextmanager
    def track(self, stage_name: str):
        start_time = time.perf_counter()
        try:
            yield
        finally:
            end_time = time.perf_counter()
            duration = end_time - start_time
            self.timings[stage_name] = duration
            logger.info(f"Stage {stage_name} took {duration:.4f}s")

    def get_summary(self) -> Dict[str, float]:
        self.timings["total_latency"] = sum(self.timings.values())
        return self.timings
