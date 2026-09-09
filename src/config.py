# Event-Driven ML Pipeline - Configuration
# Author: Maharshi Soni | License: MIT

from __future__ import annotations

from pydantic import BaseModel, Field


class PipelineConfig(BaseModel):
    """Central configuration for the event-driven ML pipeline."""

    # Broker
    broker_max_queue_size: int = Field(10_000, description="Max events per topic queue")

    # Windows
    tumbling_window_seconds: float = Field(5.0, description="Tumbling window duration")
    sliding_window_seconds: float = Field(10.0, description="Sliding window span")
    sliding_step_seconds: float = Field(2.0, description="Sliding window step")

    # Features
    feature_group_key: str = Field("user_id", description="Key to group events by")
    feature_lookback: int = Field(50, description="Max events kept per group")

    # Predictor
    model_n_samples: int = Field(2000, description="Synthetic samples for training")
    model_random_seed: int = Field(42, description="RNG seed for reproducibility")

    # Dead letter queue
    dlq_max_size: int = Field(5000, description="Max entries in the DLQ")

    # Replay store
    replay_max_size: int = Field(50_000, description="Max events in replay store")

    # Synthetic generator
    synthetic_rate: float = Field(100.0, description="Events per second for the generator")
    synthetic_users: int = Field(20, description="Number of synthetic user ids")
    synthetic_anomaly_ratio: float = Field(0.05, description="Fraction of anomalous events")

    # Server
    host: str = "0.0.0.0"
    port: int = 8000
