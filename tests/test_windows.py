# Event-Driven ML Pipeline - Window Aggregation Tests
# Author: Maharshi Soni | License: MIT

from __future__ import annotations

import time

from src.pipeline.events import Event, EventType
from src.pipeline.windows import TumblingWindow, SlidingWindow


def _make_event(value: float) -> Event:
    return Event(
        event_type=EventType.TRANSACTION,
        topic="test",
        payload={"value": value},
    )


class TestTumblingWindow:
    """Tests for fixed-size tumbling windows."""

    def test_aggregation_on_flush(self) -> None:
        """Flush forces aggregation over buffered events."""
        tw = TumblingWindow(window_seconds=60.0)  # long window so it won't auto-close
        for v in [10.0, 20.0, 30.0]:
            tw.add(_make_event(v))

        result = tw.flush()
        assert result is not None
        assert result["count"] == 3
        assert result["sum"] == 60.0
        assert result["mean"] == 20.0
        assert result["min"] == 10.0
        assert result["max"] == 30.0
        assert result["event_count"] == 3

    def test_empty_flush(self) -> None:
        """Flushing an empty window returns None."""
        tw = TumblingWindow(window_seconds=60.0)
        assert tw.flush() is None


class TestSlidingWindow:
    """Tests for time-based sliding windows."""

    def test_compute(self) -> None:
        """compute() returns aggregation of current window contents."""
        sw = SlidingWindow(window_seconds=60.0, slide_seconds=1.0)
        for v in [5.0, 15.0, 25.0]:
            sw.add(_make_event(v))

        result = sw.compute()
        assert result["count"] == 3
        assert result["sum"] == 45.0
        assert result["window_size"] == 60.0
