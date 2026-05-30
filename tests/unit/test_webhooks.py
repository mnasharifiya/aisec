"""
Unit tests for the webhook delivery system.
Run with: pytest tests/unit/test_webhooks.py -v
"""

from __future__ import annotations

import json

import pytest

from aisec.api.webhooks import (
    WebhookConfig,
    WebhookDispatcher,
    WebhookPayload,
)

# ── WebhookConfig tests ───────────────────────────────────────────────────────


class TestWebhookConfig:

    def test_valid_config_creates_successfully(self) -> None:
        config = WebhookConfig(
            url="https://hooks.example.com/aisec",
            secret="a" * 32,
        )
        assert config.enabled is True

    def test_rejects_invalid_url(self) -> None:
        with pytest.raises(ValueError, match="https://"):
            WebhookConfig(url="ftp://invalid.com", secret="a" * 32)

    def test_rejects_short_secret(self) -> None:
        with pytest.raises(ValueError, match="32 characters"):
            WebhookConfig(url="https://example.com", secret="short")

    def test_should_deliver_all_when_events_empty(self) -> None:
        config = WebhookConfig(
            url="https://example.com",
            secret="a" * 32,
            events=[],
        )
        assert config.should_deliver("any_event_type") is True

    def test_should_deliver_matching_event(self) -> None:
        config = WebhookConfig(
            url="https://example.com",
            secret="a" * 32,
            events=["action_blocked"],
        )
        assert config.should_deliver("action_blocked") is True

    def test_should_not_deliver_non_matching_event(self) -> None:
        config = WebhookConfig(
            url="https://example.com",
            secret="a" * 32,
            events=["action_blocked"],
        )
        assert config.should_deliver("analysis_complete") is False

    def test_disabled_config_never_delivers(self) -> None:
        config = WebhookConfig(
            url="https://example.com",
            secret="a" * 32,
            enabled=False,
        )
        assert config.should_deliver("any_event") is False

    def test_http_url_allowed(self) -> None:
        config = WebhookConfig(
            url="http://internal.corp.com/aisec",
            secret="a" * 32,
        )
        assert config.url.startswith("http://")


# ── WebhookPayload tests ──────────────────────────────────────────────────────


class TestWebhookPayload:

    def _make_payload(self, **overrides) -> WebhookPayload:
        base = dict(
            event_id="test-event-001",
            agent_id="trading_bot_v1",
            action_type="manipulate_news_feed",
            decision="BLOCK",
            risk_score=0.94,
            rule_hits=["TRADING-002"],
            scenario="trading_ai",
            explanation="News manipulation detected",
            blocked=True,
            requires_review=False,
            temporal_alerts=[],
        )
        base.update(overrides)
        return WebhookPayload(**base)

    def test_to_json_produces_valid_json(self) -> None:
        payload = self._make_payload()
        body = payload.to_json()
        parsed = json.loads(body)
        assert parsed["event_id"] == "test-event-001"
        assert parsed["decision"] == "BLOCK"
        assert parsed["risk_score"] == 0.94

    def test_to_json_includes_all_required_fields(self) -> None:
        payload = self._make_payload()
        parsed = json.loads(payload.to_json())
        for field in (
            "event_id",
            "agent_id",
            "action_type",
            "decision",
            "risk_score",
            "rule_hits",
            "scenario",
            "explanation",
            "blocked",
            "timestamp",
            "aisec_version",
        ):
            assert field in parsed, f"Missing field: {field}"

    def test_sign_returns_64_char_hex(self) -> None:
        payload = self._make_payload()
        signature = payload.sign("a" * 32)
        assert len(signature) == 64
        assert all(c in "0123456789abcdef" for c in signature)

    def test_signature_is_deterministic(self) -> None:
        p1 = self._make_payload()
        p2 = self._make_payload()
        # Force same timestamp
        p1.timestamp = p2.timestamp = "2025-01-01T00:00:00+00:00"
        assert p1.sign("a" * 32) == p2.sign("a" * 32)

    def test_different_payloads_different_signatures(self) -> None:
        p1 = self._make_payload(decision="BLOCK")
        p2 = self._make_payload(decision="ALLOW")
        p1.timestamp = p2.timestamp = "2025-01-01T00:00:00+00:00"
        assert p1.sign("a" * 32) != p2.sign("a" * 32)

    def test_different_secrets_different_signatures(self) -> None:
        payload = self._make_payload()
        sig1 = payload.sign("a" * 32)
        sig2 = payload.sign("b" * 32)
        assert sig1 != sig2

    def test_explanation_truncated_to_500_chars(self) -> None:
        payload = self._make_payload(explanation="x" * 1000)
        parsed = json.loads(payload.to_json())
        assert len(parsed["explanation"]) <= 500

    def test_risk_score_rounded_to_4_decimals(self) -> None:
        payload = self._make_payload(risk_score=0.123456789)
        parsed = json.loads(payload.to_json())
        assert parsed["risk_score"] == 0.1235

    def test_timestamp_is_set_automatically(self) -> None:
        payload = self._make_payload()
        assert payload.timestamp != ""
        assert "T" in payload.timestamp


# ── WebhookDispatcher tests ───────────────────────────────────────────────────


class TestWebhookDispatcher:

    def _make_config(self, **overrides) -> WebhookConfig:
        base = dict(
            url="https://hooks.example.com/aisec",
            secret="a" * 32,
        )
        base.update(overrides)
        return WebhookConfig(**base)

    def _make_payload(self) -> WebhookPayload:
        return WebhookPayload(
            event_id="evt-001",
            agent_id="bot",
            action_type="manipulate_news_feed",
            decision="BLOCK",
            risk_score=0.94,
            rule_hits=["TRADING-002"],
            scenario="trading_ai",
            explanation="Test",
            blocked=True,
            requires_review=False,
            temporal_alerts=[],
        )

    def test_initialises_with_empty_configs(self) -> None:
        dispatcher = WebhookDispatcher(configs=[])
        assert dispatcher is not None

    def test_dispatch_with_no_endpoints_does_not_crash(self) -> None:
        dispatcher = WebhookDispatcher(configs=[])
        dispatcher.dispatch(self._make_payload())

    def test_dispatch_with_disabled_endpoint_does_not_deliver(
        self,
    ) -> None:
        config = self._make_config(enabled=False)
        dispatcher = WebhookDispatcher(configs=[config])
        # Should not raise — disabled endpoints are skipped
        dispatcher.dispatch(self._make_payload())

    def test_dispatch_filters_by_event_type(self) -> None:
        config = self._make_config(events=["action_blocked"])
        dispatcher = WebhookDispatcher(configs=[config])
        payload = self._make_payload()
        payload.event_type = "analysis_complete"
        # analysis_complete not in ["action_blocked"] — skip
        dispatcher.dispatch(payload)

    def test_multiple_configs_initialise_correctly(self) -> None:
        configs = [
            self._make_config(url="https://endpoint1.com/hook"),
            self._make_config(url="https://endpoint2.com/hook"),
        ]
        dispatcher = WebhookDispatcher(configs=configs)
        assert dispatcher is not None
