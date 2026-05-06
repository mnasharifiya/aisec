"""
Unit tests for the rule engine.
Run with: pytest tests/unit/test_rules.py -v
"""

from __future__ import annotations

import pytest

from aisec.core.rules import RuleEngine
from aisec.storage.models import Decision, Event, Scenario


# ── Fixture helpers ───────────────────────────────────────────────────────────

def trading_event(
    action_type: str,
    target: str = "MARKET",
    **payload_kwargs,
) -> Event:
    """Return a trading AI event with the given action and payload."""
    return Event(
        action_type=action_type,
        agent_id="trading_bot",
        target=target,
        scenario=Scenario.TRADING_AI,
        raw_payload=payload_kwargs,
    )


def urban_event(
    action_type: str,
    target: str = "city_system",
    **payload_kwargs,
) -> Event:
    """Return an urban AI event with the given action and payload."""
    return Event(
        action_type=action_type,
        agent_id="urban_ctrl",
        target=target,
        scenario=Scenario.URBAN_AI,
        raw_payload=payload_kwargs,
    )


@pytest.fixture
def engine() -> RuleEngine:
    return RuleEngine()


# ── Trading AI rule tests ─────────────────────────────────────────────────────

class TestTradingRules:

    def test_large_trade_is_blocked(self, engine: RuleEngine) -> None:
        event  = trading_event("execute_trade", amount=2_400_000)
        result = engine.evaluate(event)
        assert result.any_fired
        assert result.final_decision == Decision.BLOCK
        assert "TRADING-001" in result.rule_ids

    def test_small_trade_passes(self, engine: RuleEngine) -> None:
        event  = trading_event("execute_trade", amount=800)
        result = engine.evaluate(event)
        assert not result.any_fired
        assert result.final_decision is None

    def test_news_manipulation_is_blocked(self, engine: RuleEngine) -> None:
        event  = trading_event("manipulate_news_feed")
        result = engine.evaluate(event)
        assert result.final_decision == Decision.BLOCK
        assert "TRADING-002" in result.rule_ids

    def test_risk_limit_override_is_escalated(self, engine: RuleEngine) -> None:
        event  = trading_event("override_risk_limit")
        result = engine.evaluate(event)
        assert result.final_decision == Decision.ESCALATE
        assert "TRADING-003" in result.rule_ids

    def test_after_hours_trade_is_reviewed(self, engine: RuleEngine) -> None:
        event  = trading_event("execute_trade", after_hours=True)
        result = engine.evaluate(event)
        assert result.final_decision == Decision.PENDING_REVIEW
        assert "TRADING-004" in result.rule_ids

    def test_block_overrides_review_when_both_fire(self, engine: RuleEngine) -> None:
        # Large trade ($2M) AND after hours — BLOCK must win over REVIEW
        event  = trading_event(
            "execute_trade", amount=2_000_000, after_hours=True
        )
        result = engine.evaluate(event)
        assert result.final_decision == Decision.BLOCK
        assert "TRADING-001" in result.rule_ids
        assert "TRADING-004" in result.rule_ids

    def test_unknown_action_does_not_fire(self, engine: RuleEngine) -> None:
        event  = trading_event("read_market_data")
        result = engine.evaluate(event)
        assert not result.any_fired


# ── Urban AI rule tests ───────────────────────────────────────────────────────

class TestUrbanRules:

    def test_curfew_is_blocked(self, engine: RuleEngine) -> None:
        event  = urban_event("set_curfew", zone="A", duration_hours=48)
        result = engine.evaluate(event)
        assert result.final_decision == Decision.BLOCK
        assert "URBAN-001" in result.rule_ids

    def test_power_grid_shutdown_is_escalated(self, engine: RuleEngine) -> None:
        event  = urban_event("shutdown_power_grid", zone="North")
        result = engine.evaluate(event)
        assert result.final_decision == Decision.ESCALATE
        assert "URBAN-002" in result.rule_ids

    def test_emergency_services_target_is_blocked(self, engine: RuleEngine) -> None:
        event  = urban_event(
            "adjust_routing", target="ambulance_routing"
        )
        result = engine.evaluate(event)
        assert result.final_decision == Decision.BLOCK
        assert "URBAN-003" in result.rule_ids

    def test_large_traffic_override_is_reviewed(self, engine: RuleEngine) -> None:
        event  = urban_event(
            "mass_traffic_redirect", affected_intersections=120
        )
        result = engine.evaluate(event)
        assert result.final_decision == Decision.PENDING_REVIEW
        assert "URBAN-004" in result.rule_ids

    def test_small_traffic_override_passes(self, engine: RuleEngine) -> None:
        event  = urban_event(
            "mass_traffic_redirect", affected_intersections=10
        )
        result = engine.evaluate(event)
        assert not result.any_fired

    def test_sensor_read_is_always_allowed(self, engine: RuleEngine) -> None:
        event  = urban_event("read_sensor", target="traffic_sensor_42")
        result = engine.evaluate(event)
        assert not result.any_fired


# ── Scenario isolation tests ──────────────────────────────────────────────────

class TestScenarioIsolation:

    def test_urban_rules_do_not_fire_for_trading_events(
        self, engine: RuleEngine
    ) -> None:
        # set_curfew is an urban rule — should not fire for trading scenario
        event  = Event(
            action_type="set_curfew",
            agent_id="trading_bot",
            target="MARKET",
            scenario=Scenario.TRADING_AI,
        )
        result = engine.evaluate(event)
        assert not result.any_fired

    def test_trading_rules_do_not_fire_for_urban_events(
        self, engine: RuleEngine
    ) -> None:
        # execute_trade is a trading rule — should not fire for urban scenario
        event  = Event(
            action_type="execute_trade",
            agent_id="urban_ctrl",
            target="city_system",
            scenario=Scenario.URBAN_AI,
            raw_payload={"amount": 5_000_000},
        )
        result = engine.evaluate(event)
        assert not result.any_fired

    def test_unknown_scenario_fires_no_rules(self, engine: RuleEngine) -> None:
        event  = Event(
            action_type="execute_trade",
            agent_id="unknown_bot",
            target="unknown",
            scenario=Scenario.UNKNOWN,
            raw_payload={"amount": 5_000_000},
        )
        result = engine.evaluate(event)
        assert not result.any_fired