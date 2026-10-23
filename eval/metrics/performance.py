"""
eval/metrics/performance.py — Métricas de rendimiento, throughput y latencia p95.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import List, Dict, Any
import numpy as np


@dataclass
class PerformanceTracker:
    latencies_ms: List[float] = field(default_factory=list)
    start_time: float = field(default_factory=time.time)

    def record_ms(self, elapsed_ms: float) -> None:
        self.latencies_ms.append(float(elapsed_ms))

    def compute_summary(self) -> Dict[str, Any]:
        if not self.latencies_ms:
            return {
                "frames_count": 0,
                "fps": 0.0,
                "latency_mean_ms": 0.0,
                "latency_p50_ms": 0.0,
                "latency_p95_ms": 0.0,
                "latency_p99_ms": 0.0,
                "latency_min_ms": 0.0,
                "latency_max_ms": 0.0,
            }

        arr = np.array(self.latencies_ms)
        total_time_s = max(1e-5, (time.time() - self.start_time))
        fps = len(self.latencies_ms) / total_time_s

        return {
            "frames_count": len(self.latencies_ms),
            "fps": round(float(fps), 2),
            "latency_mean_ms": round(float(np.mean(arr)), 2),
            "latency_p50_ms": round(float(np.percentile(arr, 50)), 2),
            "latency_p95_ms": round(float(np.percentile(arr, 95)), 2),
            "latency_p99_ms": round(float(np.percentile(arr, 99)), 2),
            "latency_min_ms": round(float(np.min(arr)), 2),
            "latency_max_ms": round(float(np.max(arr)), 2),
        }
