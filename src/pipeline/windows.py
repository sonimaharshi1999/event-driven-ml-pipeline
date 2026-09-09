# Event-Driven ML Pipeline - Windowed Aggregation
# Author: Maharshi Soni | License: MIT

from __future__ import annotations

import time
from collections import deque
from typing import Any, Callable, Deque, Dict, List, Optional

from src.pipeline.events import Event


class TumblingWindow:
    """Fixed-size, non-overlapping time window.

    Events are collected into the current window.  When the window closes
    (its duration elapses) the registered aggregation function is applied
    to the buffered events, the result is stored, and the buffer resets.
    """

    def __init__(
        self,
        window_seconds: float,
        agg_fn: Optional[Callable[[List[Event]], Dict[str, Any]]] = None,
    ) -> None:
        self.window_seconds = window_seconds
        self.agg_fn = agg_fn or self._default_agg
        self._buffer: List[Event] = []
        self._window_start: float = time.time()
        self._results: List[Dict[str, Any]] = []

    # ------------------------------------------------------------------

    def add(self, event: Event) -> Optional[Dict[str, Any]]:
        """Add *event*; return aggregation result when the window closes."""
        now = time.time()
        result: Optional[Dict[str, Any]] = None
        if now - self._window_start >= self.window_seconds and self._buffer:
            result = self._flush()
        self._buffer.append(event)
        return result

    def flush(self) -> Optional[Dict[str, Any]]:
        """Force-close the current window and return the aggregation."""
        if not self._buffer:
            return None
        return self._flush()

    # ------------------------------------------------------------------

    def _flush(self) -> Dict[str, Any]:
        result = self.agg_fn(self._buffer)
        result["window_start"] = self._window_start
        result["window_end"] = time.time()
        result["event_count"] = len(self._buffer)
        self._results.append(result)
        self._buffer = []
        self._window_start = time.time()
        return result

    @property
    def results(self) -> List[Dict[str, Any]]:
        return list(self._results)

    @staticmethod
    def _default_agg(events: List[Event]) -> Dict[str, Any]:
        values = [e.payload.get("value", 0.0) for e in events]
        return {
            "count": len(values),
            "sum": sum(values),
            "mean": sum(values) / len(values) if values else 0.0,
            "min": min(values) if values else 0.0,
            "max": max(values) if values else 0.0,
        }


class SlidingWindow:
    """Time-based sliding window with configurable slide step.

    Keeps events in a deque and evicts entries older than *window_seconds*.
    The aggregation is recomputed every *slide_seconds* (or on demand via
    :meth:`compute`).
    """

    def __init__(
        self,
        window_seconds: float,
        slide_seconds: float,
        agg_fn: Optional[Callable[[List[Event]], Dict[str, Any]]] = None,
    ) -> None:
        self.window_seconds = window_seconds
        self.slide_seconds = slide_seconds
        self.agg_fn = agg_fn or TumblingWindow._default_agg
        self._buffer: Deque[Event] = deque()
        self._last_slide: float = time.time()
        self._results: List[Dict[str, Any]] = []

    # ------------------------------------------------------------------

    def add(self, event: Event) -> Optional[Dict[str, Any]]:
        """Add *event*; return aggregation when a slide boundary is crossed."""
        self._buffer.append(event)
        self._evict()
        now = time.time()
        if now - self._last_slide >= self.slide_seconds:
            return self._slide(now)
        return None

    def compute(self) -> Dict[str, Any]:
        """Force-compute aggregation over the current window contents."""
        self._evict()
        events = list(self._buffer)
        result = self.agg_fn(events)
        result["window_size"] = self.window_seconds
        result["event_count"] = len(events)
        return result

    # ------------------------------------------------------------------

    def _evict(self) -> None:
        cutoff = time.time() - self.window_seconds
        while self._buffer and self._buffer[0].timestamp < cutoff:
            self._buffer.popleft()

    def _slide(self, now: float) -> Dict[str, Any]:
        self._last_slide = now
        result = self.compute()
        self._results.append(result)
        return result

    @property
    def results(self) -> List[Dict[str, Any]]:
        return list(self._results)
