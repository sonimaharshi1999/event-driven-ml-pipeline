# Event-Driven ML Pipeline - Real-Time Feature Engineering
# Author: Maharshi Soni | License: MIT

from __future__ import annotations

import math
from collections import defaultdict, deque
from typing import Any, Deque, Dict, List, Optional

import numpy as np

from src.pipeline.events import Event


class FeatureEngine:
    """Computes streaming features from raw events.

    Maintains running statistics per *group_key* (e.g. user_id, device_id)
    over a configurable look-back window.  Each call to :meth:`compute`
    returns a feature vector suitable for online prediction.
    """

    def __init__(
        self,
        group_key: str = "user_id",
        lookback_size: int = 50,
    ) -> None:
        self.group_key = group_key
        self.lookback_size = lookback_size
        self._history: Dict[str, Deque[Event]] = defaultdict(
            lambda: deque(maxlen=lookback_size)
        )
        self._event_counts: Dict[str, int] = defaultdict(int)

    def ingest(self, event: Event) -> None:
        """Record *event* into the feature store for its group."""
        key = event.payload.get(self.group_key, "__default__")
        self._history[key].append(event)
        self._event_counts[key] += 1

    def compute(self, event: Event) -> Dict[str, float]:
        """Return a feature dict for the group that owns *event*."""
        key = event.payload.get(self.group_key, "__default__")
        history = list(self._history[key])
        values = [e.payload.get("value", 0.0) for e in history]

        features: Dict[str, float] = {}

        # Basic aggregates
        features["count"] = float(len(values))
        features["total_events"] = float(self._event_counts[key])
        features["sum"] = float(np.sum(values)) if values else 0.0
        features["mean"] = float(np.mean(values)) if values else 0.0
        features["std"] = float(np.std(values)) if len(values) > 1 else 0.0
        features["min"] = float(np.min(values)) if values else 0.0
        features["max"] = float(np.max(values)) if values else 0.0

        # Rate features
        if len(history) >= 2:
            dt = history[-1].timestamp - history[0].timestamp
            features["event_rate"] = len(history) / dt if dt > 0 else 0.0
        else:
            features["event_rate"] = 0.0

        # Current event value
        features["current_value"] = float(event.payload.get("value", 0.0))

        # Deviation from mean
        if features["std"] > 0:
            features["z_score"] = (features["current_value"] - features["mean"]) / features["std"]
        else:
            features["z_score"] = 0.0

        # Trend: slope of last N values via simple linear regression
        features["trend_slope"] = self._linear_slope(values)

        return features

    def get_feature_names(self) -> List[str]:
        """Return ordered feature names matching the output of :meth:`compute`."""
        return [
            "count",
            "total_events",
            "sum",
            "mean",
            "std",
            "min",
            "max",
            "event_rate",
            "current_value",
            "z_score",
            "trend_slope",
        ]

    @staticmethod
    def _linear_slope(values: List[float]) -> float:
        """Compute slope via least-squares on index vs value."""
        n = len(values)
        if n < 2:
            return 0.0
        x = np.arange(n, dtype=np.float64)
        y = np.array(values, dtype=np.float64)
        x_mean = x.mean()
        y_mean = y.mean()
        denom = float(np.sum((x - x_mean) ** 2))
        if denom == 0:
            return 0.0
        return float(np.sum((x - x_mean) * (y - y_mean)) / denom)
