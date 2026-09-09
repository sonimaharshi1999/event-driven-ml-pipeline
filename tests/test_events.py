# Event-Driven ML Pipeline - Event Model Tests
# Author: Maharshi Soni | License: MIT

from __future__ import annotations

import time

from src.pipeline.events import Event, EventType


class TestEvent:
    """Tests for the Event data model."""

    def test_event_creation(self) -> None:
        """Events are created with auto-generated id and timestamp."""
        event = Event(
            event_type=EventType.TRANSACTION,
            topic="transactions",
            payload={"user_id": "u1", "value": 42.0},
        )
        assert event.event_id  # non-empty
        assert event.timestamp > 0
        assert event.event_type == EventType.TRANSACTION
        assert event.topic == "transactions"

    def test_event_age(self) -> None:
        """age_ms returns a positive duration."""
        event = Event(
            event_type=EventType.SENSOR_READING,
            topic="sensors",
            payload={"value": 1.0},
            timestamp=time.time() - 0.05,  # 50 ms ago
        )
        assert event.age_ms() >= 40  # at least ~40 ms

    def test_event_types(self) -> None:
        """All four event types are accessible."""
        assert len(EventType) == 4
        assert EventType.TRANSACTION.value == "transaction"
        assert EventType.USER_ACTION.value == "user_action"
        assert EventType.SENSOR_READING.value == "sensor_reading"
        assert EventType.SYSTEM_METRIC.value == "system_metric"
