# Event-Driven ML Pipeline - Dead Letter Queue
# Author: Maharshi Soni | License: MIT

from __future__ import annotations

import logging
import time
from collections import deque
from typing import Any, Deque, Dict, List, Optional, Tuple

from src.pipeline.events import Event

logger = logging.getLogger(__name__)


class DeadLetterQueue:
    """Stores events that failed processing so they can be inspected or replayed.

    Each entry records the original event, the error message, and the
    timestamp of the failure.
    """

    def __init__(self, max_size: int = 5000) -> None:
        self._queue: Deque[Dict[str, Any]] = deque(maxlen=max_size)
        self._max_size = max_size
        self._total_received: int = 0

    async def handle(self, event: Event, error: Exception) -> None:
        """Async handler suitable for :meth:`MessageBroker.set_dead_letter_handler`."""
        entry = {
            "event": event.model_dump(),
            "error": str(error),
            "error_type": type(error).__name__,
            "failed_at": time.time(),
        }
        self._queue.append(entry)
        self._total_received += 1
        logger.warning(
            "DLQ received event %s: %s", event.event_id[:8], error
        )

    def peek(self, n: int = 10) -> List[Dict[str, Any]]:
        """Return the *n* most recent dead-letter entries (newest first)."""
        items = list(self._queue)
        items.reverse()
        return items[:n]

    def pop(self) -> Optional[Dict[str, Any]]:
        """Remove and return the oldest entry, or None if empty."""
        if self._queue:
            return self._queue.popleft()
        return None

    def clear(self) -> int:
        """Empty the DLQ and return how many entries were removed."""
        count = len(self._queue)
        self._queue.clear()
        return count

    @property
    def size(self) -> int:
        return len(self._queue)

    def get_stats(self) -> Dict[str, Any]:
        return {
            "current_size": self.size,
            "total_received": self._total_received,
            "max_size": self._max_size,
        }
