"""
Integration tests for the AISec REST API.

Uses FastAPI's TestClient — no running server needed.
Tests cover all endpoints, validation, security, and
error handling.

Run with: pytest tests/integration/test_api.py -v
"""

from __future__ import annotations

from pathlib import Path

import pytest

# Crucial: Intercept pytest collection phase if FastAPI isn't installed.
# This prevents ModuleNotFoundError crashes in unconfigured or minimal CI environments.
pytest.importorskip(
    "fastapi", 
    reason="FastAPI and dependencies not installed in this execution target."
)

from fastapi.testclient import TestClient

from aisec.api.server import create_app
from aisec.core.engine import AnalysisEngine

# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture
def client(tmp_path: Path) -> TestClient:
    """Return a TestClient backed by a fresh engine."""
    app = create_app(log_path=tmp_path / "api_test.jsonl")
    with TestClient(app) as c:
        yield c


# ── Health endpoint tests ─────────────────────────────────────────────────────


class TestHealthEndpoint:

    def test_health_returns_200(self, client: TestClient) -> None:
        response = client.get("/api/v1/health")
        assert response.status_code == 200

    def test_health_response_structure(self, client: TestClient) -> None:
        data = client.get("/api/v1/health").json()
        assert "status" in data
        assert "version" in data
        assert "audit_chain" in data
        assert "audit_entries" in data
        assert "engine" in data

    def test_health_status_is_healthy(self, client: TestClient) -> None:
        data = client.get("/api/v1/health").json()
        assert data["status"] == "healthy"
        assert data["engine"] == "ready"
        assert data["audit_chain"] == "intact"


# ── Analyse endpoint tests ────────────────────────────────────────────────────


class TestAnalyseEndpoint:

    def test_safe_action_returns_allow(self, client: TestClient) -> None:
        response = client.post(
            "/api/v1/analyse",
            json={
                "action_type": "read_market_data",
                "agent_id": "trading_bot_v1",
                "target": "NYSE",
                "scenario": "trading_ai",
                "payload": {},
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert data["decision"] == "ALLOW"
        assert not data["blocked"]

    def test_dangerous_action_returns_block(self, client: TestClient) -> None:
        response = client.post(
            "/api/v1/analyse",
            json={
                "action_type": "manipulate_news_feed",
                "agent_id": "trading_bot_v1",
                "target": "reuters_feed",
                "scenario": "trading_ai",
                "payload": {},
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert data["decision"] in ("BLOCK", "ESCALATE", "PENDING_REVIEW")
        assert data["blocked"]

    def test_large_trade_is_blocked(self, client: TestClient) -> None:
        response = client.post(
            "/api/v1/analyse",
            json={
                "action_type": "execute_large_trade",
                "agent_id": "trading_bot_v1",
                "target": "MARKET",
                "scenario": "trading_ai",
                "payload": {"amount": 2400000},
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert data["blocked"]

    def test_curfew_is_blocked(self, client: TestClient) -> None:
        response = client.post(
            "/api/v1/analyse",
            json={
                "action_type": "set_curfew",
                "agent_id": "urban_ctrl_v1",
                "target": "city_system",
                "scenario": "urban_ai",
                "payload": {"zone": "ALL", "duration_hours": 48},
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert data["blocked"]

    def test_response_contains_required_fields(self, client: TestClient) -> None:
        response = client.post(
            "/api/v1/analyse",
            json={
                "action_type": "read_market_data",
                "agent_id": "bot_v1",
                "target": "NYSE",
                "scenario": "trading_ai",
            },
        )
        data = response.json()
        required = [
            "event_id",
            "agent_id",
            "action_type",
            "decision",
            "risk_score",
            "rule_hits",
            "explanation",
            "log_entry_id",
            "blocked",
            "requires_review",
        ]
        for field in required:
            assert field in data, f"Missing field: {field}"

    def test_risk_score_in_valid_range(self, client: TestClient) -> None:
        response = client.post(
            "/api/v1/analyse",
            json={
                "action_type": "execute_large_trade",
                "agent_id": "bot",
                "target": "MARKET",
                "scenario": "trading_ai",
                "payload": {"amount": 999999},
            },
        )
        data = response.json()
        assert 0.0 <= data["risk_score"] <= 1.0

    def test_rejects_missing_action_type(self, client: TestClient) -> None:
        response = client.post(
            "/api/v1/analyse",
            json={
                "agent_id": "bot",
                "target": "NYSE",
            },
        )
        assert response.status_code == 422

    def test_rejects_empty_agent_id(self, client: TestClient) -> None:
        response = client.post(
            "/api/v1/analyse",
            json={
                "action_type": "read_market_data",
                "agent_id": "",
                "target": "NYSE",
            },
        )
        assert response.status_code == 422

    def test_rejects_extra_fields(self, client: TestClient) -> None:
        response = client.post(
            "/api/v1/analyse",
            json={
                "action_type": "read_market_data",
                "agent_id": "bot",
                "target": "NYSE",
                "evil_field": "injection_attempt",
            },
        )
        assert response.status_code == 422

    def test_rejects_invalid_scenario(self, client: TestClient) -> None:
        response = client.post(
            "/api/v1/analyse",
            json={
                "action_type": "read_market_data",
                "agent_id": "bot",
                "target": "NYSE",
                "scenario": "invalid_scenario",
            },
        )
        assert response.status_code == 422

    def test_rejects_oversized_payload(self, client: TestClient) -> None:
        huge_payload = {f"key_{i}": i for i in range(100)}
        response = client.post(
            "/api/v1/analyse",
            json={
                "action_type": "read_market_data",
                "agent_id": "bot",
                "target": "NYSE",
                "payload": huge_payload,
            },
        )
        assert response.status_code == 422


# ── Batch analyse tests ───────────────────────────────────────────────────────


class TestBatchAnalyseEndpoint:

    def test_batch_analyses_all_events(self, client: TestClient) -> None:
        response = client.post(
            "/api/v1/analyse/batch",
            json={
                "events": [
                    {
                        "action_type": "read_market_data",
                        "agent_id": "bot",
                        "target": "NYSE",
                        "scenario": "trading_ai",
                    },
                    {
                        "action_type": "manipulate_news_feed",
                        "agent_id": "bot",
                        "target": "reuters",
                        "scenario": "trading_ai",
                    },
                    {
                        "action_type": "read_market_data",
                        "agent_id": "bot",
                        "target": "NASDAQ",
                        "scenario": "trading_ai",
                    },
                ]
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert data["total"] == 3
        assert data["blocked_count"] >= 1
        assert len(data["results"]) == 3

    def test_rejects_empty_batch(self, client: TestClient) -> None:
        response = client.post("/api/v1/analyse/batch", json={"events": []})
        assert response.status_code == 422

    def test_rejects_batch_over_100(self, client: TestClient) -> None:
        events = [
            {
                "action_type": "read_market_data",
                "agent_id": "bot",
                "target": "NYSE",
                "scenario": "trading_ai",
            }
            for _ in range(101)
        ]
        response = client.post("/api/v1/analyse/batch", json={"events": events})
        assert response.status_code == 422

    def test_batch_response_counts_are_correct(self, client: TestClient) -> None:
        response = client.post(
            "/api/v1/analyse/batch",
            json={
                "events": [
                    {
                        "action_type": "read_market_data",
                        "agent_id": "bot",
                        "target": "NYSE",
                        "scenario": "trading_ai",
                    },
                    {
                        "action_type": "read_market_data",
                        "agent_id": "bot",
                        "target": "NASDAQ",
                        "scenario": "trading_ai",
                    },
                ]
            },
        )
        data = response.json()
        assert (
            data["blocked_count"] + data["allowed_count"] + data["review_count"]
        ) == data["total"]


# ── Audit verify tests ────────────────────────────────────────────────────────


class TestAuditVerifyEndpoint:

    def test_verify_intact_chain(self, client: TestClient) -> None:
        # Make some events first
        client.post(
            "/api/v1/analyse",
            json={
                "action_type": "read_market_data",
                "agent_id": "bot",
                "target": "NYSE",
                "scenario": "trading_ai",
            },
        )
        response = client.get("/api/v1/audit/verify")
        assert response.status_code == 200
        data = response.json()
        assert data["chain_intact"] is True
        assert data["entry_count"] >= 1
        assert data["errors"] == []


# ── Metrics tests ─────────────────────────────────────────────────────────────


class TestMetricsEndpoint:

    def test_metrics_returns_200(self, client: TestClient) -> None:
        response = client.get("/api/v1/metrics/summary")
        assert response.status_code == 200

    def test_metrics_structure(self, client: TestClient) -> None:
        data = client.get("/api/v1/metrics/summary").json()
        required = [
            "total_events",
            "blocked_events",
            "allowed_events",
            "review_events",
            "block_rate",
            "avg_risk_score",
            "audit_chain_ok",
        ]
        for field in required:
            assert field in data

    def test_block_rate_after_events(self, client: TestClient) -> None:
        client.post(
            "/api/v1/analyse",
            json={
                "action_type": "read_market_data",
                "agent_id": "bot",
                "target": "NYSE",
                "scenario": "trading_ai",
            },
        )
        client.post(
            "/api/v1/analyse",
            json={
                "action_type": "manipulate_news_feed",
                "agent_id": "bot",
                "target": "reuters",
                "scenario": "trading_ai",
            },
        )
        data = client.get("/api/v1/metrics/summary").json()
        assert data["total_events"] >= 2
        assert data["blocked_events"] >= 1
        assert 0.0 <= data["block_rate"] <= 1.0


# ── API docs tests ────────────────────────────────────────────────────────────


class TestAPIDocs:

    def test_swagger_ui_available(self, client: TestClient) -> None:
        response = client.get("/docs")
        assert response.status_code == 200

    def test_openapi_schema_available(self, client: TestClient) -> None:
        response = client.get("/openapi.json")
        assert response.status_code == 200
        schema = response.json()
        assert schema["info"]["title"] == "AISec — AI Runtime Security API"
        assert "paths" in schema