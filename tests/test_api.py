# Event-Driven ML Pipeline - API Endpoint Tests
# Author: Maharshi Soni | License: MIT

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from src.api.dashboard import app


@pytest.fixture(scope="module")
def client() -> TestClient:
    """Create a test client that triggers startup/shutdown."""
    with TestClient(app) as c:
        yield c


class TestDashboardAPI:
    """Tests for the FastAPI monitoring endpoints."""

    def test_health(self, client: TestClient) -> None:
        resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"

    def test_stats_endpoint(self, client: TestClient) -> None:
        resp = client.get("/stats")
        assert resp.status_code == 200
        data = resp.json()
        assert "broker" in data
        assert "predictor" in data
        assert "dead_letter" in data
        assert "uptime_seconds" in data

    def test_ingest_event(self, client: TestClient) -> None:
        resp = client.post(
            "/events",
            json={
                "event_type": "transaction",
                "topic": "transactions",
                "payload": {"user_id": "u1", "value": 99.9},
            },
        )
        assert resp.status_code == 200
        assert resp.json()["status"] == "accepted"

    def test_dashboard_html(self, client: TestClient) -> None:
        resp = client.get("/")
        assert resp.status_code == 200
        assert "ML Pipeline Dashboard" in resp.text
