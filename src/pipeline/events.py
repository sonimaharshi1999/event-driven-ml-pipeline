# Event-Driven ML Pipeline - Event Definitions
# Author: Maharshi Soni | License: MIT

from __future__ import annotations

import time
import uuid
from enum import Enum
from typing import Any, Dict, Optional

from pydantic import BaseModel, ConfigDict, Field


class EventType(str, Enum):
    """Supported event types flowing through the pipeline."""

    TRANSACTION = "transaction"
    USER_ACTION = "user_action"
    SENSOR_READING = "sensor_reading"
    SYSTEM_METRIC = "system_metric"


class Event(BaseModel):
    """Immutable event that flows through the pipeline.

    Every event carries a unique id, a timestamp, a type tag, a topic for
    routing, and an arbitrary payload dictionary.
    """

    model_config = ConfigDict(frozen=True)

    event_id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    timestamp: float = Field(default_factory=time.time)
    event_type: EventType
    topic: str
    payload: Dict[str, Any]
    metadata: Optional[Dict[str, Any]] = None

    def age_ms(self) -> float:
        """Return the age of the event in milliseconds."""
        return (time.time() - self.timestamp) * 1000.0
