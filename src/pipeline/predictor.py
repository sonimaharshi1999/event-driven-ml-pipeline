# Event-Driven ML Pipeline - Online Prediction Engine
# Author: Maharshi Soni | License: MIT

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from src.pipeline.features import FeatureEngine

logger = logging.getLogger(__name__)

MODEL_DIR = Path(__file__).resolve().parent.parent.parent / "models"


class OnlinePredictor:
    """Wraps a scikit-learn model for real-time, single-event prediction.

    On first use it trains a small anomaly-style classifier on synthetic
    data (normal vs. anomalous transaction patterns) and persists the
    artefacts under ``models/``.
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        scaler_path: Optional[str] = None,
        feature_names: Optional[List[str]] = None,
    ) -> None:
        self._model_path = model_path or str(MODEL_DIR / "model.joblib")
        self._scaler_path = scaler_path or str(MODEL_DIR / "scaler.joblib")
        self._feature_names = feature_names or FeatureEngine().get_feature_names()
        self._model: Optional[GradientBoostingClassifier] = None
        self._scaler: Optional[StandardScaler] = None
        self._prediction_count: int = 0
        self._total_latency_ms: float = 0.0

    # ------------------------------------------------------------------
    # Load / train
    # ------------------------------------------------------------------

    def load(self) -> bool:
        """Load a persisted model, returning True on success."""
        try:
            self._model = joblib.load(self._model_path)
            self._scaler = joblib.load(self._scaler_path)
            logger.info("Model loaded from %s", self._model_path)
            return True
        except FileNotFoundError:
            logger.warning("No saved model found; call train() first")
            return False

    def train(self, n_samples: int = 2000, random_seed: int = 42) -> Dict[str, Any]:
        """Train on synthetic data and persist artefacts.

        Returns a dict with accuracy and feature importance.
        """
        rng = np.random.RandomState(random_seed)

        # ---- Generate synthetic feature matrix ----
        n_features = len(self._feature_names)
        X_normal = rng.normal(loc=0.0, scale=1.0, size=(n_samples, n_features))
        X_anomaly = rng.normal(loc=2.5, scale=1.5, size=(n_samples // 4, n_features))
        # Make anomalies more pronounced on z_score and trend_slope columns
        z_idx = self._feature_names.index("z_score") if "z_score" in self._feature_names else -1
        trend_idx = self._feature_names.index("trend_slope") if "trend_slope" in self._feature_names else -1
        if z_idx >= 0:
            X_anomaly[:, z_idx] += 3.0
        if trend_idx >= 0:
            X_anomaly[:, trend_idx] += 2.0

        X = np.vstack([X_normal, X_anomaly])
        y = np.concatenate(
            [np.zeros(n_samples), np.ones(n_samples // 4)]
        ).astype(int)

        # ---- Scale and split ----
        self._scaler = StandardScaler()
        X_scaled = self._scaler.fit_transform(X)
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=random_seed, stratify=y
        )

        # ---- Train ----
        self._model = GradientBoostingClassifier(
            n_estimators=80,
            max_depth=4,
            learning_rate=0.1,
            random_state=random_seed,
        )
        self._model.fit(X_train, y_train)
        accuracy = float(self._model.score(X_test, y_test))

        # ---- Persist ----
        os.makedirs(MODEL_DIR, exist_ok=True)
        joblib.dump(self._model, self._model_path)
        joblib.dump(self._scaler, self._scaler_path)
        logger.info("Model trained (acc=%.4f) and saved to %s", accuracy, self._model_path)

        importances = dict(
            zip(
                self._feature_names,
                [round(float(v), 4) for v in self._model.feature_importances_],
            )
        )
        return {"accuracy": round(accuracy, 4), "feature_importances": importances}

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict(self, features: Dict[str, float]) -> Dict[str, Any]:
        """Run inference on a single feature vector.

        Returns predicted label, probability, and latency.
        """
        if self._model is None or self._scaler is None:
            if not self.load():
                self.train()

        vec = np.array(
            [[features.get(f, 0.0) for f in self._feature_names]]
        )
        start = time.perf_counter()
        vec_scaled = self._scaler.transform(vec)  # type: ignore[union-attr]
        proba = self._model.predict_proba(vec_scaled)[0]  # type: ignore[union-attr]
        label = int(self._model.predict(vec_scaled)[0])  # type: ignore[union-attr]
        elapsed_ms = (time.perf_counter() - start) * 1000.0

        self._prediction_count += 1
        self._total_latency_ms += elapsed_ms

        return {
            "label": label,
            "probability": round(float(proba[label]), 4),
            "is_anomaly": label == 1,
            "latency_ms": round(elapsed_ms, 3),
        }

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def get_stats(self) -> Dict[str, Any]:
        return {
            "predictions": self._prediction_count,
            "avg_latency_ms": (
                round(self._total_latency_ms / self._prediction_count, 3)
                if self._prediction_count
                else 0.0
            ),
        }
