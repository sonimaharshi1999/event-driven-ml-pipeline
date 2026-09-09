# Event-Driven ML Pipeline - Shared Test Fixtures
# Author: Maharshi Soni | License: MIT

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Ensure the project root is on sys.path so that ``import src.*`` works
# regardless of how pytest is invoked.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.pipeline.events import Event, EventType
from src.pipeline.broker import MessageBroker
from src.pipeline.dead_letter import DeadLetterQueue
from src.pipeline.features import FeatureEngine
from src.pipeline.predictor import OnlinePredictor
from src.pipeline.replay import EventReplayStore
from src.pipeline.windows import TumblingWindow, SlidingWindow


@pytest.fixture
def sample_event() -> Event:
    """A single transaction event for quick tests."""
    return Event(
        event_type=EventType.TRANSACTION,
        topic="transactions",
        payload={"user_id": "user_001", "value": 150.0},
    )


@pytest.fixture
def sample_events() -> list[Event]:
    """A batch of 20 transaction events."""
    from src.synthetic import generate_batch
    return generate_batch(n=20, topic="transactions")


@pytest.fixture
def broker() -> MessageBroker:
    return MessageBroker()


@pytest.fixture
def dlq() -> DeadLetterQueue:
    return DeadLetterQueue(max_size=100)


@pytest.fixture
def feature_engine() -> FeatureEngine:
    return FeatureEngine(group_key="user_id", lookback_size=50)


@pytest.fixture
def predictor(tmp_path: Path) -> OnlinePredictor:
    return OnlinePredictor(
        model_path=str(tmp_path / "model.joblib"),
        scaler_path=str(tmp_path / "scaler.joblib"),
    )


@pytest.fixture
def replay_store() -> EventReplayStore:
    return EventReplayStore(max_size=1000)


@pytest.fixture
def tumbling_window() -> TumblingWindow:
    return TumblingWindow(window_seconds=1.0)


@pytest.fixture
def sliding_window() -> SlidingWindow:
    return SlidingWindow(window_seconds=2.0, slide_seconds=0.5)
