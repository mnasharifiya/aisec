"""
Integration tests for the analysis engine.

These tests exercise the full pipeline end-to-end:
    Event → Vector → Score → Rules → Decision → Audit log

Run with: pytest tests/integration/test_engine.py -v
"""

from __future__ import annotations

from pathlib import Path

import pytest

from aisec.core.engine import AnalysisEngine
from aisec.storage.audit import AuditLogger
from aisec.storage.models import Decision, Event, Scenario


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def engine(tmp_path: Path) -> AnalysisEngine:
    """Return a fresh engine backed by a temporary audit log."""
    return AnalysisEngine(log_path=tmp_path / "test_audit.jsonl")


def trading(action_type: str, **payload) -> Event:
    return Event(
        action_type=action_type,
        agent_id="trading_bot",
        target="MARKET",
        scenario=Scenario.TRADING_AI,
        raw_payload=payload,
    )


def urban(action_type: str, target: str = "city_system", **payload) -> Event:
    return Event(
        action_type=action_type,
        agent_id="urban_ctrl",
        target=target,
        scenario=Scenario.URBAN_AI,
        raw_payload=payload,
    )


# ── Core pipeline tests ───────────────────────────────────────────────────────

class TestAnalysisEngine:

    def test_returns_engine_result(self, engine: AnalysisEngine) -> None:
        result = engine.analyse(trading("read_market_data"))
        assert result.event        is not None
        assert result.analysis     is not None
        assert result.log_entry_id != ""

    def test_safe_action_is_allowed(self, engine: AnalysisEngine) -> None:
        result = engine.analyse(trading("read_market_data"))
        assert result.decision == Decision.ALLOW
        assert not result.blocked
        assert not result.requires_review

    def test_large_trade_is_blocked(self, engine: AnalysisEngine) -> None:
        result = engine.analyse(trading("execute_large_trade", amount=2_400_000))
        assert result.blocked
        assert result.decision == Decision.BLOCK

    def test_news_manipulation_is_blocked(self, engine: AnalysisEngine) -> None:
        result = engine.analyse(trading("manipulate_news_feed"))
        assert result.blocked
        assert result.decision == Decision.BLOCK

    def test_risk_limit_override_is_escalated(self, engine: AnalysisEngine) -> None:
        result = engine.analyse(trading("override_risk_limit"))
        assert result.blocked
        assert result.decision == Decision.ESCALATE

    def test_curfew_is_blocked(self, engine: AnalysisEngine) -> None:
        result = engine.analyse(
            urban("set_curfew", zone="ALL", duration_hours=48)
        )
        assert result.blocked
        assert result.decision == Decision.BLOCK

    def test_power_grid_shutdown_is_escalated(self, engine: AnalysisEngine) -> None:
        result = engine.analyse(urban("shutdown_power_grid", zone="North"))
        assert result.blocked
        assert result.decision == Decision.ESCALATE

    def test_sensor_read_is_allowed(self, engine: AnalysisEngine) -> None:
        result = engine.analyse(urban("read_sensor", target="traffic_sensor_1"))
        assert result.decision == Decision.ALLOW
        assert not result.blocked

    def test_emergency_services_target_is_blocked(
        self, engine: AnalysisEngine
    ) -> None:
        result = engine.analyse(
            urban("adjust_routing", target="ambulance_routing")
        )
        assert result.blocked

    def test_risk_score_is_in_valid_range(self, engine: AnalysisEngine) -> None:
        result = engine.analyse(trading("execute_large_trade", amount=500_000))
        assert 0.0 <= result.risk_score <= 1.0


# ── Audit log integration tests ───────────────────────────────────────────────

class TestAuditIntegration:

    def test_every_analysis_writes_to_audit_log(
        self, engine: AnalysisEngine
    ) -> None:
        engine.analyse(trading("read_market_data"))
        engine.analyse(trading("execute_large_trade", amount=2_000_000))
        engine.analyse(urban("read_sensor"))
        assert engine.audit_count() == 3

    def test_audit_chain_is_intact_after_analyses(
        self, engine: AnalysisEngine
    ) -> None:
        for i in range(10):
            engine.analyse(trading("read_market_data"))
        ok, errors = engine.verify_audit_chain()
        assert ok is True
        assert errors == []

    def test_blocked_actions_appear_in_audit_log(
        self, tmp_path: Path
    ) -> None:
        log_path = tmp_path / "audit.jsonl"
        engine   = AnalysisEngine(log_path=log_path)

        engine.analyse(trading("manipulate_news_feed"))

        logger  = AuditLogger(log_path=log_path)
        entries = logger.get_all()
        assert len(entries) == 1
        assert entries[0].payload["decision"] == "BLOCK"
        assert entries[0].payload["action_type"] == "manipulate_news_feed"

    def test_audit_log_entry_contains_required_fields(
        self, tmp_path: Path
    ) -> None:
        log_path = tmp_path / "audit.jsonl"
        engine   = AnalysisEngine(log_path=log_path)
        result   = engine.analyse(trading("read_market_data"))

        logger  = AuditLogger(log_path=log_path)
        entries = logger.get_all()

        assert len(entries) == 1
        payload = entries[0].payload

        assert "agent_id"    in payload
        assert "action_type" in payload
        assert "risk_score"  in payload
        assert "decision"    in payload
        assert "rule_hits"   in payload
        assert "explanation" in payload

    def test_log_entry_id_matches_audit_entry(
        self, tmp_path: Path
    ) -> None:
        log_path = tmp_path / "audit.jsonl"
        engine   = AnalysisEngine(log_path=log_path)
        result   = engine.analyse(trading("read_market_data"))

        logger  = AuditLogger(log_path=log_path)
        entries = logger.get_all()

        assert entries[0].log_id == result.log_entry_id