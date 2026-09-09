# Event-Driven ML Pipeline - Synthetic Event Generator
# Author: Maharshi Soni | License: MIT

from __future__ import annotations

import asyncio
import random
import time
from typing import AsyncIterator, List

from src.pipeline.events import Event, EventType


async def generate_transaction_stream(
    rate_per_second: float = 100.0,
    n_users: int = 20,
    anomaly_ratio: float = 0.05,
) -> AsyncIterator[Event]:
    """Yield synthetic transaction events at approximately *rate_per_second*.

    About *anomaly_ratio* of the events will have abnormally high values
    to exercise the anomaly detector.
    """
    delay = 1.0 / rate_per_second
    user_ids = [f"user_{i:03d}" for i in range(n_users)]

    while True:
        is_anomaly = random.random() < anomaly_ratio
        value = random.gauss(500, 100) if not is_anomaly else random.gauss(5000, 800)
        event = Event(
            event_type=EventType.TRANSACTION,
            topic="transactions",
            payload={
                "user_id": random.choice(user_ids),
                "value": round(value, 2),
                "currency": "USD",
                "is_synthetic_anomaly": is_anomaly,
            },
        )
        yield event
        await asyncio.sleep(delay)


async def generate_sensor_stream(
    rate_per_second: float = 50.0,
    n_devices: int = 10,
) -> AsyncIterator[Event]:
    """Yield synthetic IoT sensor readings."""
    delay = 1.0 / rate_per_second
    device_ids = [f"device_{i:03d}" for i in range(n_devices)]

    while True:
        event = Event(
            event_type=EventType.SENSOR_READING,
            topic="sensors",
            payload={
                "user_id": random.choice(device_ids),
                "value": round(random.gauss(22.0, 3.0), 2),
                "unit": "celsius",
            },
        )
        yield event
        await asyncio.sleep(delay)


def generate_batch(
    n: int = 100,
    topic: str = "transactions",
    event_type: EventType = EventType.TRANSACTION,
) -> List[Event]:
    """Return a list of *n* synthetic events (synchronous, for testing)."""
    events: List[Event] = []
    user_ids = [f"user_{i:03d}" for i in range(10)]
    for _ in range(n):
        events.append(
            Event(
                event_type=event_type,
                topic=topic,
                payload={
                    "user_id": random.choice(user_ids),
                    "value": round(random.gauss(500, 100), 2),
                },
            )
        )
    return events
