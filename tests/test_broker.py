# Event-Driven ML Pipeline - Broker Tests
# Author: Maharshi Soni | License: MIT

from __future__ import annotations

import asyncio
from typing import List

import pytest

from src.pipeline.broker import MessageBroker
from src.pipeline.events import Event, EventType


@pytest.mark.asyncio
class TestMessageBroker:
    """Tests for the in-process async message broker."""

    async def test_publish_and_subscribe(self, broker: MessageBroker) -> None:
        """A subscriber receives events published to its topic."""
        received: List[Event] = []

        async def handler(event: Event) -> None:
            received.append(event)

        broker.subscribe("test_topic", handler)
        await broker.start()

        event = Event(
            event_type=EventType.TRANSACTION,
            topic="test_topic",
            payload={"value": 1.0},
        )
        await broker.publish(event)
        await asyncio.sleep(0.3)
        await broker.stop()

        assert len(received) == 1
        assert received[0].event_id == event.event_id

    async def test_multiple_subscribers(self, broker: MessageBroker) -> None:
        """Multiple subscribers on the same topic each get every event."""
        counters = [0, 0]

        async def handler_a(event: Event) -> None:
            counters[0] += 1

        async def handler_b(event: Event) -> None:
            counters[1] += 1

        broker.subscribe("multi", handler_a)
        broker.subscribe("multi", handler_b)
        await broker.start()

        for _ in range(5):
            await broker.publish(
                Event(event_type=EventType.TRANSACTION, topic="multi", payload={"value": 1})
            )
        await asyncio.sleep(0.5)
        await broker.stop()

        assert counters[0] == 5
        assert counters[1] == 5

    async def test_dead_letter_on_failure(self) -> None:
        """Failed deliveries route to the dead-letter handler."""
        dead: List[tuple] = []
        broker = MessageBroker()

        async def bad_handler(event: Event) -> None:
            raise ValueError("boom")

        async def dlq_handler(event: Event, exc: Exception) -> None:
            dead.append((event, exc))

        broker.subscribe("fail_topic", bad_handler)
        broker.set_dead_letter_handler(dlq_handler)
        await broker.start()

        await broker.publish(
            Event(event_type=EventType.TRANSACTION, topic="fail_topic", payload={"v": 1})
        )
        await asyncio.sleep(0.3)
        await broker.stop()

        assert len(dead) == 1
        assert isinstance(dead[0][1], ValueError)

    async def test_stats_tracking(self, broker: MessageBroker) -> None:
        """Stats reflect published and delivered counts."""
        async def noop(event: Event) -> None:
            pass

        broker.subscribe("stats_topic", noop)
        await broker.start()

        for _ in range(3):
            await broker.publish(
                Event(event_type=EventType.TRANSACTION, topic="stats_topic", payload={"v": 1})
            )
        await asyncio.sleep(0.3)
        await broker.stop()

        stats = broker.get_stats()
        assert stats["stats_topic"]["published"] == 3
        assert stats["stats_topic"]["delivered"] == 3
