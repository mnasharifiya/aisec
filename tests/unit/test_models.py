"""
Unit tests for AISec data models.
Run with: pytest tests/unit/test_models.py -v
"""

import pytest
from aisec.storage.models import (
    AuditLogEntry,
    AnalysisResult,
    Decision,
    Event,
    FeatureVector,
    Scenario,
    Severity,
)

# ── Event tests ───────────────────────────────────────────────────────────────


class TestEvent:

    def test_creates_valid_event(self) -> None:
        e = Event(action_type="buy_stock", agent_id="trading_bot", target="AAPL")
        assert e.action_type == "buy_stock"
        assert e.agent_id == "trading_bot"
        assert e.event_id != ""
        assert e.timestamp != ""

    def test_rejects_empty_action_type(self) -> None:
        with pytest.raises(ValueError, match="action_type"):
            Event(action_type="", agent_id="bot", target="x")

    def test_rejects_empty_agent_id(self) -> None:
        with pytest.raises(ValueError, match="agent_id"):
            Event(action_type="buy", agent_id="", target="x")

    def test_two_events_have_unique_ids(self) -> None:
        e1 = Event(action_type="buy", agent_id="bot", target="x")
        e2 = Event(action_type="buy", agent_id="bot", target="x")
        assert e1.event_id != e2.event_id

    def test_default_scenario_is_unknown(self) -> None:
        e = Event(action_type="buy", agent_id="bot", target="x")
        assert e.scenario == Scenario.UNKNOWN


# ── FeatureVector tests ───────────────────────────────────────────────────────


class TestFeatureVector:

    def test_creates_valid_vector(self) -> None:
        fv = FeatureVector(
            event_id="abc",
            vector=[0.1, 0.2, 0.0, 1.0, 0.0, 0.5, 0.0, 0.0],
        )
        assert len(fv.vector) == 8

    def test_rejects_wrong_dimension(self) -> None:
        with pytest.raises(ValueError, match="dimensions"):
            FeatureVector(event_id="abc", vector=[0.1, 0.2])

    def test_rejects_out_of_range_values(self) -> None:
        with pytest.raises(ValueError, match="range"):
            FeatureVector(
                event_id="abc",
                vector=[1.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            )


# ── AnalysisResult tests ──────────────────────────────────────────────────────


class TestAnalysisResult:

    def test_creates_valid_result(self) -> None:
        r = AnalysisResult(
            event_id="abc",
            risk_score=0.85,
            decision=Decision.BLOCK,
            explanation="High-value trade exceeds threshold",
        )
        assert r.risk_score == 0.85
        assert r.decision == Decision.BLOCK

    def test_rejects_score_above_one(self) -> None:
        with pytest.raises(ValueError, match="risk_score"):
            AnalysisResult(
                event_id="abc",
                risk_score=1.5,
                decision=Decision.ALLOW,
                explanation="test",
            )

    def test_rejects_invalid_decision_type(self) -> None:
        with pytest.raises(ValueError, match="decision"):
            AnalysisResult(
                event_id="abc",
                risk_score=0.5,
                decision="BLOCK",  # string instead of enum
                explanation="test",
            )


# ── AuditLogEntry tests ───────────────────────────────────────────────────────


class TestAuditLogEntry:

    def test_hash_computed_on_creation(self) -> None:
        entry = AuditLogEntry(
            record_type="event",
            record_id="abc",
            payload={"action": "buy"},
            prev_hash="0",
        )
        assert len(entry.current_hash) == 64  # SHA-256 hex digest

    def test_verify_passes_for_unmodified_entry(self) -> None:
        entry = AuditLogEntry(
            record_type="event",
            record_id="abc",
            payload={"action": "buy"},
            prev_hash="0",
        )
        assert entry.verify("0") is True

    def test_verify_fails_after_payload_modification(self) -> None:
        entry = AuditLogEntry(
            record_type="event",
            record_id="abc",
            payload={"action": "buy"},
            prev_hash="0",
        )
        # Simulate tampering
        entry.payload["action"] = "sell_everything"
        assert entry.verify("0") is False

    def test_verify_fails_for_wrong_prev_hash(self) -> None:
        entry = AuditLogEntry(
            record_type="event",
            record_id="abc",
            payload={"action": "buy"},
            prev_hash="0",
        )
        assert entry.verify("wrong_hash") is False

    def test_chain_of_two_entries(self) -> None:
        genesis = AuditLogEntry(
            record_type="event",
            record_id="001",
            payload={"action": "start"},
            prev_hash="0",
        )
        second = AuditLogEntry(
            record_type="event",
            record_id="002",
            payload={"action": "buy"},
            prev_hash=genesis.current_hash,
        )
        assert second.verify(genesis.current_hash) is True
