# Event-Driven ML Pipeline - Dead Letter Queue Tests
# Author: Maharshi Soni | License: MIT

from __future__ import annotations

import asyncio

import pytest

from src.pipeline.dead_letter import DeadLetterQueue
from src.pipeline.events import Event, EventType


@pytest.mark.asyncio
class TestDeadLetterQueue:
    """Tests for the dead-letter queue."""

    async def test_handle_stores_entry(self, dlq: DeadLetterQueue) -> None:
        """handle() persists a failed event with error details."""
        event = Event(
            event_type=EventType.TRANSACTION,
            topic="t",
            payload={"value": 1},
        )
        await dlq.handle(event, ValueError("bad value"))

        assert dlq.size == 1
        entry = dlq.peek(1)[0]
        assert entry["error"] == "bad value"
        assert entry["error_type"] == "ValueError"

    async def test_pop_removes_oldest(self, dlq: DeadLetterQueue) -> None:
        """pop() returns and removes the oldest entry."""
        for i in range(3):
            e = Event(
                event_type=EventType.TRANSACTION,
                topic="t",
                payload={"value": i},
            )
            await dlq.handle(e, RuntimeError(f"err_{i}"))

        first = dlq.pop()
        assert first is not None
        assert first["error"] == "err_0"
        assert dlq.size == 2

    async def test_clear(self, dlq: DeadLetterQueue) -> None:
        """clear() empties the queue and returns the count."""
        for i in range(5):
            e = Event(
                event_type=EventType.TRANSACTION,
                topic="t",
                payload={"value": i},
            )
            await dlq.handle(e, RuntimeError("x"))

        removed = dlq.clear()
        assert removed == 5
        assert dlq.size == 0
