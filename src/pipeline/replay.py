# Event-Driven ML Pipeline - Event Replay Store
# Author: Maharshi Soni | License: MIT

from __future__ import annotations

import bisect
import logging
from collections import deque
from typing import Any, Deque, Dict, List, Optional

from src.pipeline.events import Event

logger = logging.getLogger(__name__)


class EventReplayStore:
    """Append-only store that allows replaying historical events.

    Events are stored in timestamp order.  Callers can query by time
    range or by topic.
    """

    def __init__(self, max_size: int = 50_000) -> None:
        self._events: Deque[Event] = deque(maxlen=max_size)
        self._timestamps: Deque[float] = deque(maxlen=max_size)
        self._max_size = max_size

    def store(self, event: Event) -> None:
        """Persist *event* for later replay."""
        self._events.append(event)
        self._timestamps.append(event.timestamp)

    def replay_range(
        self,
        start_ts: float,
        end_ts: float,
        topic: Optional[str] = None,
    ) -> List[Event]:
        """Return events whose timestamp falls in [start_ts, end_ts].

        Optionally filter to a single *topic*.
        """
        ts_list = list(self._timestamps)
        lo = bisect.bisect_left(ts_list, start_ts)
        hi = bisect.bisect_right(ts_list, end_ts)
        events_list = list(self._events)
        result = events_list[lo:hi]
        if topic:
            result = [e for e in result if e.topic == topic]
        return result

    def replay_last(self, n: int, topic: Optional[str] = None) -> List[Event]:
        """Return the most recent *n* events, optionally filtered by topic."""
        events = list(self._events)
        if topic:
            events = [e for e in events if e.topic == topic]
        return events[-n:]

    @property
    def size(self) -> int:
        return len(self._events)

    def get_stats(self) -> Dict[str, Any]:
        topics: Dict[str, int] = {}
        for e in self._events:
            topics[e.topic] = topics.get(e.topic, 0) + 1
        return {
            "stored_events": self.size,
            "max_size": self._max_size,
            "topics": topics,
        }
