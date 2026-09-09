# Event-Driven ML Pipeline - Predictor Tests
# Author: Maharshi Soni | License: MIT

from __future__ import annotations

from pathlib import Path

from src.pipeline.predictor import OnlinePredictor


class TestOnlinePredictor:
    """Tests for model training and online prediction."""

    def test_train_and_predict(self, predictor: OnlinePredictor) -> None:
        """Training produces a model that returns valid predictions."""
        result = predictor.train(n_samples=500, random_seed=42)
        assert result["accuracy"] > 0.7

        features = {
            "count": 10.0,
            "total_events": 50.0,
            "sum": 500.0,
            "mean": 50.0,
            "std": 10.0,
            "min": 30.0,
            "max": 70.0,
            "event_rate": 5.0,
            "current_value": 55.0,
            "z_score": 0.5,
            "trend_slope": 0.1,
        }
        pred = predictor.predict(features)
        assert pred["label"] in (0, 1)
        assert 0.0 <= pred["probability"] <= 1.0
        assert isinstance(pred["is_anomaly"], bool)
        assert pred["latency_ms"] >= 0

    def test_model_persistence(self, tmp_path: Path) -> None:
        """A saved model can be loaded by a fresh predictor."""
        p1 = OnlinePredictor(
            model_path=str(tmp_path / "m.joblib"),
            scaler_path=str(tmp_path / "s.joblib"),
        )
        p1.train(n_samples=300)

        p2 = OnlinePredictor(
            model_path=str(tmp_path / "m.joblib"),
            scaler_path=str(tmp_path / "s.joblib"),
        )
        assert p2.load() is True

        pred = p2.predict({"mean": 1.0, "std": 0.5, "z_score": 0.0})
        assert pred["label"] in (0, 1)

    def test_stats(self, predictor: OnlinePredictor) -> None:
        """Prediction stats accumulate correctly."""
        predictor.train(n_samples=300)
        predictor.predict({"mean": 1.0})
        predictor.predict({"mean": 2.0})

        stats = predictor.get_stats()
        assert stats["predictions"] == 2
        assert stats["avg_latency_ms"] > 0
