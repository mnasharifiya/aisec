"""
Unit tests for the decision engine.
Run with: pytest tests/unit/test_decision.py -v
"""

from __future__ import annotations

import pytest

from aisec.core.decision import (
    DecisionContext,
    DecisionEngine,
    THRESHOLD_BLOCK,
    THRESHOLD_REVIEW,
    THRESHOLD_WATCH,
)
from aisec.core.rules import RuleEngine, RuleEngineResult
from aisec.core.scorer import RiskScorer, ScoreResult
from aisec.core.vector import FeatureVectorBuilder
from aisec.storage.models import Decision, Event, FeatureVector, Scenario


# ── Helpers ───────────────────────────────────────────────────────────────────

def make_score(risk: float) -> ScoreResult:
    """Return a ScoreResult with the given risk score."""
    return ScoreResult(
        risk_score=risk,
        raw_score=risk,
        weights_used="test",
        explanation=f"test score={risk:.3f}",
    )


def make_event(
    action_type: str,
    scenario: Scenario = Scenario.TRADING_AI,
    **payload,
) -> Event:
    return Event(
        action_type=action_type,
        agent_id="test_bot",
        target="test_target",
        scenario=scenario,
        raw_payload=payload,
    )


def empty_rule_result() -> RuleEngineResult:
    """Return a RuleEngineResult with no rules fired."""
    return RuleEngineResult()


@pytest.fixture
def engine() -> DecisionEngine:
    return DecisionEngine()


@pytest.fixture
def rule_engine() -> RuleEngine:
    return RuleEngine()


@pytest.fixture
def scorer() -> RiskScorer:
    return RiskScorer()


@pytest.fixture
def builder() -> FeatureVectorBuilder:
    return FeatureVectorBuilder()


# ── Score-only decision tests ─────────────────────────────────────────────────

class TestScoreOnlyDecisions:
    """Tests where no rules fire — decision driven by score alone."""

    def test_low_score_is_allowed(self, engine: DecisionEngine) -> None:
        ctx = DecisionContext(
            event=make_event("read_market_data"),
            rule_result=empty_rule_result(),
            score_result=make_score(0.10),
        )
        result = engine.decide(ctx)
        assert result.decision == Decision.ALLOW

    def test_watch_score_is_allowed_but_logged(
        self, engine: DecisionEngine
    ) -> None:
        ctx = DecisionContext(
            event=make_event("execute_trade"),
            rule_result=empty_rule_result(),
            score_result=make_score(THRESHOLD_WATCH + 0.01),
        )
        result = engine.decide(ctx)
        assert result.decision == Decision.ALLOW
        assert "WATCH" in result.explanation

    def test_review_score_triggers_pending_review(
        self, engine: DecisionEngine
    ) -> None:
        ctx = DecisionContext(
            event=make_event("execute_trade"),
            rule_result=empty_rule_result(),
            score_result=make_score(THRESHOLD_REVIEW + 0.01),
        )
        result = engine.decide(ctx)
        assert result.decision == Decision.PENDING_REVIEW

    def test_block_score_triggers_block(self, engine: DecisionEngine) -> None:
        ctx = DecisionContext(
            event=make_event("execute_large_trade"),
            rule_result=empty_rule_result(),
            score_result=make_score(THRESHOLD_BLOCK + 0.01),
        )
        result = engine.decide(ctx)
        assert result.decision == Decision.BLOCK

    def test_exact_block_threshold_triggers_block(
        self, engine: DecisionEngine
    ) -> None:
        ctx = DecisionContext(
            event=make_event("execute_large_trade"),
            rule_result=empty_rule_result(),
            score_result=make_score(THRESHOLD_BLOCK),
        )
        result = engine.decide(ctx)
        assert result.decision == Decision.BLOCK


# ── Rule-driven decision tests ────────────────────────────────────────────────

class TestRuleDrivenDecisions:
    """Tests where rules fire and drive the decision."""

    def test_rule_block_overrides_low_score(
        self, engine: DecisionEngine, rule_engine: RuleEngine
    ) -> None:
        # News manipulation rule fires BLOCK regardless of score
        event       = make_event("manipulate_news_feed", scenario=Scenario.TRADING_AI)
        rule_result = rule_engine.evaluate(event)
        ctx         = DecisionContext(
            event=event,
            rule_result=rule_result,
            score_result=make_score(0.10),   # Low score — rule must still block
        )
        result = engine.decide(ctx)
        assert result.decision == Decision.BLOCK
        assert "RULE BLOCK" in result.explanation
        assert "TRADING-002" in result.rule_hits

    def test_rule_escalate_overrides_low_score(
        self, engine: DecisionEngine, rule_engine: RuleEngine
    ) -> None:
        event       = make_event("override_risk_limit", scenario=Scenario.TRADING_AI)
        rule_result = rule_engine.evaluate(event)
        ctx         = DecisionContext(
            event=event,
            rule_result=rule_result,
            score_result=make_score(0.10),
        )
        result = engine.decide(ctx)
        assert result.decision == Decision.ESCALATE
        assert "RULE ESCALATE" in result.explanation

    def test_rule_review_applies_when_score_is_low(
        self, engine: DecisionEngine, rule_engine: RuleEngine
    ) -> None:
        event       = make_event(
            "execute_trade",
            scenario=Scenario.TRADING_AI,
            after_hours=True,
        )
        rule_result = rule_engine.evaluate(event)
        ctx         = DecisionContext(
            event=event,
            rule_result=rule_result,
            score_result=make_score(0.10),   # Low score — rule drives review
        )
        result = engine.decide(ctx)
        assert result.decision == Decision.PENDING_REVIEW

    def test_score_block_overrides_rule_review(
        self, engine: DecisionEngine, rule_engine: RuleEngine
    ) -> None:
        # Rule says REVIEW but score says BLOCK — score wins for blocking
        event       = make_event(
            "execute_trade",
            scenario=Scenario.TRADING_AI,
            after_hours=True,
        )
        rule_result = rule_engine.evaluate(event)
        ctx         = DecisionContext(
            event=event,
            rule_result=rule_result,
            score_result=make_score(0.95),   # Score demands block
        )
        result = engine.decide(ctx)
        assert result.decision == Decision.BLOCK

    def test_urban_curfew_is_blocked(
        self, engine: DecisionEngine, rule_engine: RuleEngine
    ) -> None:
        event       = make_event(
            "set_curfew",
            scenario=Scenario.URBAN_AI,
            zone="ALL",
            duration_hours=48,
        )
        rule_result = rule_engine.evaluate(event)
        ctx         = DecisionContext(
            event=event,
            rule_result=rule_result,
            score_result=make_score(0.20),
        )
        result = engine.decide(ctx)
        assert result.decision == Decision.BLOCK
        assert "URBAN-001" in result.rule_hits


# ── End-to-end pipeline tests ─────────────────────────────────────────────────

class TestEndToEndPipeline:
    """
    Full pipeline: Event → FeatureVector → Score → Rules → Decision.
    These are the most realistic tests in the suite.
    """

    def _run_pipeline(
        self,
        event: Event,
        builder: FeatureVectorBuilder,
        scorer: RiskScorer,
        rule_engine: RuleEngine,
        engine: DecisionEngine,
    ) -> Decision:
        fv          = builder.build(event)
        score       = scorer.score(fv, event.scenario)
        rule_result = rule_engine.evaluate(event)
        ctx         = DecisionContext(
            event=event,
            rule_result=rule_result,
            score_result=score,
        )
        return engine.decide(ctx).decision

    def test_large_trade_is_blocked_end_to_end(
        self,
        builder: FeatureVectorBuilder,
        scorer: RiskScorer,
        rule_engine: RuleEngine,
        engine: DecisionEngine,
    ) -> None:
        event = Event(
            action_type="execute_large_trade",
            agent_id="trading_bot",
            target="MARKET",
            scenario=Scenario.TRADING_AI,
            raw_payload={"amount": 2_400_000},
        )
        decision = self._run_pipeline(event, builder, scorer, rule_engine, engine)
        assert decision == Decision.BLOCK

    def test_curfew_is_blocked_end_to_end(
        self,
        builder: FeatureVectorBuilder,
        scorer: RiskScorer,
        rule_engine: RuleEngine,
        engine: DecisionEngine,
    ) -> None:
        event = Event(
            action_type="set_curfew",
            agent_id="urban_ctrl",
            target="city_system",
            scenario=Scenario.URBAN_AI,
            raw_payload={"zone": "ALL", "duration_hours": 48},
        )
        decision = self._run_pipeline(event, builder, scorer, rule_engine, engine)
        assert decision == Decision.BLOCK

    def test_sensor_read_is_allowed_end_to_end(
        self,
        builder: FeatureVectorBuilder,
        scorer: RiskScorer,
        rule_engine: RuleEngine,
        engine: DecisionEngine,
    ) -> None:
        event = Event(
            action_type="read_sensor",
            agent_id="urban_ctrl",
            target="traffic_sensor_42",
            scenario=Scenario.URBAN_AI,
        )
        decision = self._run_pipeline(event, builder, scorer, rule_engine, engine)
        assert decision == Decision.ALLOW

    def test_market_data_read_is_allowed_end_to_end(
        self,
        builder: FeatureVectorBuilder,
        scorer: RiskScorer,
        rule_engine: RuleEngine,
        engine: DecisionEngine,
    ) -> None:
        event = Event(
            action_type="read_market_data",
            agent_id="trading_bot",
            target="NYSE",
            scenario=Scenario.TRADING_AI,
        )
        decision = self._run_pipeline(event, builder, scorer, rule_engine, engine)
        assert decision == Decision.ALLOW

    def test_news_manipulation_is_blocked_end_to_end(
        self,
        builder: FeatureVectorBuilder,
        scorer: RiskScorer,
        rule_engine: RuleEngine,
        engine: DecisionEngine,
    ) -> None:
        event = Event(
            action_type="manipulate_news_feed",
            agent_id="trading_bot",
            target="reuters_feed",
            scenario=Scenario.TRADING_AI,
        )
        decision = self._run_pipeline(event, builder, scorer, rule_engine, engine)
        assert decision == Decision.BLOCK