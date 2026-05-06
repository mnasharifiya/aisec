"""
Unit tests for the risk scorer and feature vector builder.
Run with: pytest tests/unit/test_scorer.py -v
"""

from __future__ import annotations

import pytest

from aisec.core.scorer import RiskScorer
from aisec.core.vector import FeatureVectorBuilder
from aisec.storage.models import Event, FeatureVector, Scenario


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def scorer() -> RiskScorer:
    return RiskScorer()


@pytest.fixture
def builder() -> FeatureVectorBuilder:
    return FeatureVectorBuilder()


def trading_event(action_type: str, **payload) -> Event:
    return Event(
        action_type=action_type,
        agent_id="trading_bot",
        target="MARKET",
        scenario=Scenario.TRADING_AI,
        raw_payload=payload,
    )


def urban_event(action_type: str, target: str = "city_system", **payload) -> Event:
    return Event(
        action_type=action_type,
        agent_id="urban_ctrl",
        target=target,
        scenario=Scenario.URBAN_AI,
        raw_payload=payload,
    )


# ── FeatureVectorBuilder tests ────────────────────────────────────────────────

class TestFeatureVectorBuilder:

    def test_builds_8_dimensional_vector(self, builder: FeatureVectorBuilder) -> None:
        event = trading_event("read_market_data")
        fv    = builder.build(event)
        assert len(fv.vector) == 8

    def test_all_values_in_range(self, builder: FeatureVectorBuilder) -> None:
        event = trading_event("execute_large_trade", amount=2_000_000)
        fv    = builder.build(event)
        assert all(0.0 <= v <= 1.0 for v in fv.vector)

    def test_safe_read_produces_low_vector(self, builder: FeatureVectorBuilder) -> None:
        event = trading_event("read_market_data")
        fv    = builder.build(event)
        assert sum(fv.vector) < 1.0

    def test_dangerous_action_produces_high_vector(
        self, builder: FeatureVectorBuilder
    ) -> None:
        event = trading_event("manipulate_news_feed")
        fv    = builder.build(event)
        assert sum(fv.vector) > 2.0

    def test_sensitive_target_sets_flag(self, builder: FeatureVectorBuilder) -> None:
        event = urban_event("read_sensor", target="emergency_dispatch")
        fv    = builder.build(event)
        assert fv.vector[6] == 1.0

    def test_privileged_action_sets_flag(self, builder: FeatureVectorBuilder) -> None:
        event = urban_event("shutdown_power_grid")
        fv    = builder.build(event)
        assert fv.vector[7] == 1.0

    def test_burst_rate_sets_frequency_score(
        self, builder: FeatureVectorBuilder
    ) -> None:
        event = urban_event("read_sensor", burst_rate=50.0)
        fv    = builder.build(event)
        assert fv.vector[2] == pytest.approx(0.5)

    def test_burst_rate_clamped_at_one(
        self, builder: FeatureVectorBuilder
    ) -> None:
        event = urban_event("read_sensor", burst_rate=9999.0)
        fv    = builder.build(event)
        assert fv.vector[2] == 1.0


# ── RiskScorer tests ──────────────────────────────────────────────────────────

class TestRiskScorer:

    def test_score_is_in_valid_range(self, scorer: RiskScorer) -> None:
        fv     = FeatureVector(event_id="x", vector=[0.5] * 8)
        result = scorer.score(fv, Scenario.TRADING_AI)
        assert 0.0 < result.risk_score < 1.0

    def test_zero_vector_produces_low_score(self, scorer: RiskScorer) -> None:
        fv     = FeatureVector(event_id="x", vector=[0.0] * 8)
        result = scorer.score(fv, Scenario.TRADING_AI)
        assert result.risk_score < 0.5

    def test_one_vector_produces_high_score(self, scorer: RiskScorer) -> None:
        fv     = FeatureVector(event_id="x", vector=[1.0] * 8)
        result = scorer.score(fv, Scenario.TRADING_AI)
        assert result.risk_score > 0.5

    def test_dangerous_action_scores_above_threshold(
        self,
        builder: FeatureVectorBuilder,
        scorer: RiskScorer,
    ) -> None:
        event  = trading_event("manipulate_news_feed")
        fv     = builder.build(event)
        result = scorer.score(fv, Scenario.TRADING_AI)
        assert result.risk_score > 0.60

    def test_safe_read_scores_below_dangerous_action(
        self,
        builder: FeatureVectorBuilder,
        scorer: RiskScorer,
    ) -> None:
        safe_event     = trading_event("read_market_data")
        safe_fv        = builder.build(safe_event)
        safe_result    = scorer.score(safe_fv, Scenario.TRADING_AI)

        danger_event   = trading_event("manipulate_news_feed")
        danger_fv      = builder.build(danger_event)
        danger_result  = scorer.score(danger_fv, Scenario.TRADING_AI)

        assert safe_result.risk_score < danger_result.risk_score

    def test_explanation_is_non_empty(self, scorer: RiskScorer) -> None:
        fv     = FeatureVector(event_id="x", vector=[0.5] * 8)
        result = scorer.score(fv, Scenario.TRADING_AI)
        assert len(result.explanation) > 0

    def test_urban_weights_used_for_urban_scenario(self, scorer: RiskScorer) -> None:
        fv     = FeatureVector(event_id="x", vector=[0.5] * 8)
        result = scorer.score(fv, Scenario.URBAN_AI)
        assert result.weights_used == "urban_ai"

    def test_raises_for_wrong_vector_size(self, scorer: RiskScorer) -> None:
        fv            = FeatureVector.__new__(FeatureVector)
        fv.event_id   = "x"
        fv.vector     = [0.5, 0.5]
        fv.dimensions = []
        with pytest.raises(ValueError, match="features"):
            scorer.score(fv, Scenario.TRADING_AI)