# Event-Driven ML Pipeline - Replay Store Tests
# Author: Maharshi Soni | License: MIT

from __future__ import annotations

import time

from src.pipeline.events import Event, EventType
from src.pipeline.replay import EventReplayStore


class TestEventReplayStore:
    """Tests for event storage and replay."""

    def test_store_and_replay_last(self, replay_store: EventReplayStore) -> None:
        """Stored events can be replayed by count."""
        for i in range(10):
            replay_store.store(
                Event(
                    event_type=EventType.TRANSACTION,
                    topic="t",
                    payload={"value": float(i)},
                )
            )

        last_3 = replay_store.replay_last(3)
        assert len(last_3) == 3
        assert last_3[-1].payload["value"] == 9.0

    def test_replay_by_topic(self, replay_store: EventReplayStore) -> None:
        """Replay can filter by topic."""
        replay_store.store(
            Event(event_type=EventType.TRANSACTION, topic="a", payload={"value": 1})
        )
        replay_store.store(
            Event(event_type=EventType.SENSOR_READING, topic="b", payload={"value": 2})
        )
        replay_store.store(
            Event(event_type=EventType.TRANSACTION, topic="a", payload={"value": 3})
        )

        a_events = replay_store.replay_last(10, topic="a")
        assert len(a_events) == 2

    def test_replay_range(self, replay_store: EventReplayStore) -> None:
        """replay_range returns events within the time window."""
        t0 = time.time()
        for i in range(5):
            replay_store.store(
                Event(
                    event_type=EventType.TRANSACTION,
                    topic="t",
                    payload={"value": float(i)},
                    timestamp=t0 + i,
                )
            )

        # Events at t0+1, t0+2, t0+3
        subset = replay_store.replay_range(t0 + 0.5, t0 + 3.5)
        assert len(subset) == 3

    def test_stats(self, replay_store: EventReplayStore) -> None:
        """Stats report correct counts per topic."""
        for _ in range(3):
            replay_store.store(
                Event(event_type=EventType.TRANSACTION, topic="x", payload={"v": 1})
            )
        replay_store.store(
            Event(event_type=EventType.SENSOR_READING, topic="y", payload={"v": 2})
        )

        stats = replay_store.get_stats()
        assert stats["stored_events"] == 4
        assert stats["topics"]["x"] == 3
        assert stats["topics"]["y"] == 1
