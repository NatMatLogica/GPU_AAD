#!/usr/bin/env python
"""
Cost tracking for benchmark evaluations.

Converts wall-clock time to dollar cost given a cost-per-hour rate.

Usage:
    cost = CostTracker(cost_per_hour=3.50, platform="hgx_h100_8gpu_onprem")
    cost_usd = cost.compute(elapsed_seconds=0.012)
"""

from dataclasses import dataclass


@dataclass
class CostTracker:
    """Converts elapsed time to dollar cost."""
    cost_per_hour: float = 0.0
    platform: str = ""

    def compute(self, elapsed_seconds: float) -> float:
        """Return cost in USD for the given elapsed time."""
        if self.cost_per_hour <= 0:
            return 0.0
        return elapsed_seconds * self.cost_per_hour / 3600.0

    def to_dict(self) -> dict:
        return {
            "cost_per_hour": self.cost_per_hour,
            "platform": self.platform,
        }
