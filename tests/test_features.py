# Event-Driven ML Pipeline - Feature Engine Tests
# Author: Maharshi Soni | License: MIT

from __future__ import annotations

from src.pipeline.events import Event, EventType
from src.pipeline.features import FeatureEngine


class TestFeatureEngine:
    """Tests for real-time feature computation."""

    def test_basic_features(self, feature_engine: FeatureEngine) -> None:
        """Features are computed correctly after ingesting events."""
        for v in [100.0, 200.0, 300.0]:
            event = Event(
                event_type=EventType.TRANSACTION,
                topic="t",
                payload={"user_id": "u1", "value": v},
            )
            feature_engine.ingest(event)

        last = Event(
            event_type=EventType.TRANSACTION,
            topic="t",
            payload={"user_id": "u1", "value": 250.0},
        )
        feature_engine.ingest(last)
        feats = feature_engine.compute(last)

        assert feats["count"] == 4
        assert feats["current_value"] == 250.0
        assert feats["mean"] > 0
        assert feats["std"] > 0
        assert "z_score" in feats
        assert "trend_slope" in feats

    def test_feature_names(self, feature_engine: FeatureEngine) -> None:
        """get_feature_names returns the canonical ordered list."""
        names = feature_engine.get_feature_names()
        assert "mean" in names
        assert "z_score" in names
        assert len(names) == 11

    def test_groups_are_isolated(self, feature_engine: FeatureEngine) -> None:
        """Features for different groups do not leak into each other."""
        for uid in ["a", "b"]:
            for v in [10.0, 20.0]:
                e = Event(
                    event_type=EventType.TRANSACTION,
                    topic="t",
                    payload={"user_id": uid, "value": v},
                )
                feature_engine.ingest(e)

        probe_a = Event(
            event_type=EventType.TRANSACTION,
            topic="t",
            payload={"user_id": "a", "value": 15.0},
        )
        feature_engine.ingest(probe_a)
        feats_a = feature_engine.compute(probe_a)
        assert feats_a["count"] == 3  # 10, 20, 15  (only group "a")
