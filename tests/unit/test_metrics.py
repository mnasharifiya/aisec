"""
Unit tests for Prometheus metrics.
Run with: pytest tests/unit/test_metrics.py -v
"""

from __future__ import annotations

import pytest

from aisec.api.metrics import AISeCMetrics


@pytest.fixture
def metrics() -> AISeCMetrics:
    return AISeCMetrics()


class TestAISeCMetrics:

    def test_initialises_without_error(self, metrics: AISeCMetrics) -> None:
        assert metrics is not None

    def test_available_when_prometheus_installed(self, metrics: AISeCMetrics) -> None:
        # prometheus_client is in requirements — should be available
        assert metrics.available is True

    def test_record_event_does_not_raise(self, metrics: AISeCMetrics) -> None:
        metrics.record_event(
            decision="BLOCK",
            scenario="trading_ai",
            agent_id="trading_bot_v1",
            risk_score=0.94,
            rule_hits=["TRADING-001"],
            latency_s=0.005,
        )

    def test_record_multiple_events(self, metrics: AISeCMetrics) -> None:
        for i in range(10):
            metrics.record_event(
                decision="ALLOW",
                scenario="trading_ai",
                agent_id=f"bot_{i}",
                risk_score=0.1,
                rule_hits=[],
            )

    def test_record_temporal_alert(self, metrics: AISeCMetrics) -> None:
        metrics.record_temporal_alert(
            threat="BURST_ATTACK",
            severity="HIGH",
        )

    def test_record_api_request(self, metrics: AISeCMetrics) -> None:
        metrics.record_api_request(
            endpoint="/api/v1/analyse",
            status_code=200,
            latency_s=0.012,
        )

    def test_update_audit_status_intact(self, metrics: AISeCMetrics) -> None:
        metrics.update_audit_status(chain_intact=True, entry_count=100)

    def test_update_audit_status_broken(self, metrics: AISeCMetrics) -> None:
        metrics.update_audit_status(chain_intact=False, entry_count=50)

    def test_update_queue_size(self, metrics: AISeCMetrics) -> None:
        metrics.update_queue_size(pending=5)

    def test_generate_output_returns_bytes(self, metrics: AISeCMetrics) -> None:
        content, content_type = metrics.generate_output()
        assert isinstance(content, bytes)
        assert len(content) > 0
        assert "text/plain" in content_type

    def test_generate_output_contains_metric_names(self, metrics: AISeCMetrics) -> None:
        metrics.record_event(
            decision="BLOCK",
            scenario="trading_ai",
            agent_id="bot",
            risk_score=0.9,
            rule_hits=["TRADING-001"],
        )
        content, _ = metrics.generate_output()
        text = content.decode("utf-8")
        assert "aisec_events_total" in text
        assert "aisec_risk_score" in text

    def test_agents_seen_tracks_unique_agents(self, metrics: AISeCMetrics) -> None:
        for agent in ["bot_a", "bot_b", "bot_a", "bot_c"]:
            metrics.record_event(
                decision="ALLOW",
                scenario="trading_ai",
                agent_id=agent,
                risk_score=0.1,
                rule_hits=[],
            )
        # 3 unique agents: bot_a, bot_b, bot_c
        content, _ = metrics.generate_output()
        assert "aisec_agents_seen_total" in content.decode()

    def test_record_webhook_delivery_success(self, metrics: AISeCMetrics) -> None:
        metrics.record_webhook_delivery("success")

    def test_record_webhook_delivery_failure(self, metrics: AISeCMetrics) -> None:
        metrics.record_webhook_delivery("failure")
