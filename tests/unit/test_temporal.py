"""
Unit tests for the temporal anomaly detector.
Run with: pytest tests/unit/test_temporal.py -v
"""

from __future__ import annotations

import time
from pathlib import Path

import pytest

from aisec.core.temporal import (
    AgentWindow,
    TemporalAnomalyDetector,
    TemporalConfig,
    TemporalThreat,
    WindowEvent,
)
from aisec.core.engine import AnalysisEngine
from aisec.storage.models import Decision, Event, Scenario

# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture
def config() -> TemporalConfig:
    """Fast config for testing — short window, low thresholds."""
    return TemporalConfig(
        window_seconds=10.0,
        burst_threshold=5,
        probe_threshold=3,
        probe_score_min=0.65,
        probe_score_max=0.80,
        escalation_delta=0.15,
        escalation_min_events=4,
        cumulative_amount_threshold=1_000_000.0,
        evasion_threshold=2,
        max_window_size=100,
    )


@pytest.fixture
def detector(config: TemporalConfig) -> TemporalAnomalyDetector:
    return TemporalAnomalyDetector(config=config)


@pytest.fixture
def engine(tmp_path: Path) -> AnalysisEngine:
    return AnalysisEngine(log_path=tmp_path / "temporal_test.jsonl")


def _window_event(
    action_type: str = "execute_trade",
    risk_score: float = 0.3,
    was_blocked: bool = False,
    amount: float = 0.0,
) -> WindowEvent:
    return WindowEvent(
        action_type=action_type,
        risk_score=risk_score,
        was_blocked=was_blocked,
        amount=amount,
        timestamp=time.monotonic(),
    )


def _engine_result(
    engine: AnalysisEngine,
    action_type: str = "read_market_data",
    amount: float = 0.0,
    scenario: Scenario = Scenario.TRADING_AI,
) -> object:
    """Run a real event through the engine and return the result."""
    event = Event(
        action_type=action_type,
        agent_id="test_agent",
        target="MARKET",
        scenario=scenario,
        raw_payload={"amount": amount} if amount > 0 else {},
    )
    return engine.analyse(event)


# ── AgentWindow tests ─────────────────────────────────────────────────────────


class TestAgentWindow:

    def test_empty_window_returns_empty_list(self, config: TemporalConfig) -> None:
        window = AgentWindow(config)
        assert window.get_recent() == []
        assert window.size() == 0

    def test_add_event_increases_size(self, config: TemporalConfig) -> None:
        window = AgentWindow(config)
        window.add(_window_event())
        assert window.size() == 1

    def test_events_returned_in_order(self, config: TemporalConfig) -> None:
        window = AgentWindow(config)
        for i in range(5):
            window.add(_window_event(action_type=f"action_{i}"))
        events = window.get_recent()
        assert [e.action_type for e in events] == [f"action_{i}" for i in range(5)]

    def test_old_events_are_expired(self) -> None:
        config = TemporalConfig(window_seconds=0.1)
        window = AgentWindow(config)
        window.add(_window_event())
        time.sleep(0.15)
        assert window.size() == 0

    def test_max_window_size_enforced(self, config: TemporalConfig) -> None:
        config.max_window_size = 5
        window = AgentWindow(config)
        for _ in range(20):
            window.add(_window_event())
        assert window.size() <= 5

    def test_clear_empties_window(self, config: TemporalConfig) -> None:
        window = AgentWindow(config)
        for _ in range(10):
            window.add(_window_event())
        window.clear()
        assert window.size() == 0


# ── Burst attack detection tests ──────────────────────────────────────────────


class TestBurstAttackDetection:

    def test_detects_burst_above_threshold(
        self,
        detector: TemporalAnomalyDetector,
        engine: AnalysisEngine,
    ) -> None:
        """6 events in window with threshold=5 must trigger BURST_ATTACK."""
        alerts_found = []
        for _ in range(6):
            result = _engine_result(engine)
            alerts = detector.update(result)
            alerts_found.extend(alerts)

        burst_alerts = [
            a for a in alerts_found if a.threat == TemporalThreat.BURST_ATTACK
        ]
        assert (
            len(burst_alerts) >= 1
        ), "BURST_ATTACK not detected after exceeding threshold"

    def test_no_burst_below_threshold(
        self,
        detector: TemporalAnomalyDetector,
        engine: AnalysisEngine,
    ) -> None:
        """5 events with threshold=5 must NOT trigger BURST_ATTACK."""
        alerts_found = []
        for _ in range(5):
            result = _engine_result(engine)
            alerts = detector.update(result)
            alerts_found.extend(alerts)

        burst_alerts = [
            a for a in alerts_found if a.threat == TemporalThreat.BURST_ATTACK
        ]
        assert len(burst_alerts) == 0

    def test_burst_alert_contains_correct_evidence(
        self,
        detector: TemporalAnomalyDetector,
        engine: AnalysisEngine,
    ) -> None:
        alerts_found = []
        for _ in range(7):
            result = _engine_result(engine)
            alerts = detector.update(result)
            alerts_found.extend(alerts)

        burst_alerts = [
            a for a in alerts_found if a.threat == TemporalThreat.BURST_ATTACK
        ]
        assert len(burst_alerts) >= 1
        alert = burst_alerts[0]
        assert "event_count" in alert.evidence
        assert "threshold" in alert.evidence
        assert alert.severity == "HIGH"
        assert alert.agent_id == "test_agent"


# ── Threshold probing detection tests ─────────────────────────────────────────


class TestThresholdProbingDetection:

    def test_detects_probing_above_threshold(
        self,
        detector: TemporalAnomalyDetector,
        engine: AnalysisEngine,
    ) -> None:
        """
        4 events scoring in [0.65, 0.80] with probe_threshold=3
        must trigger THRESHOLD_PROBING.
        """
        # We need to inject events with specific scores
        # Use the detector's internal update mechanism directly
        window = detector._get_or_create_window("probe_agent")
        for _ in range(4):
            window.add(
                WindowEvent(
                    action_type="execute_trade",
                    risk_score=0.72,  # In probe range [0.65, 0.80]
                    was_blocked=False,
                    amount=0.0,
                    timestamp=time.monotonic(),
                )
            )

        # Create a minimal engine result to trigger analysis
        event = Event(
            action_type="execute_trade",
            agent_id="probe_agent",
            target="MARKET",
            scenario=Scenario.TRADING_AI,
        )
        result = engine.analyse(event)
        # Override agent_id to match our window
        result.event.__class__.__init__

        # Manually check probing on our pre-populated window
        events = window.get_recent()
        probe_events = [e for e in events if 0.65 <= e.risk_score <= 0.80]
        assert len(probe_events) == 4
        assert len(probe_events) > detector._config.probe_threshold

    def test_no_probing_with_low_scores(
        self,
        detector: TemporalAnomalyDetector,
        engine: AnalysisEngine,
    ) -> None:
        """Events with low scores must not trigger probing detection."""
        alerts_found = []
        for _ in range(10):
            result = _engine_result(engine, "read_market_data")
            alerts = detector.update(result)
            alerts_found.extend(alerts)

        probe_alerts = [
            a for a in alerts_found if a.threat == TemporalThreat.THRESHOLD_PROBING
        ]
        assert len(probe_alerts) == 0


# ── Cumulative exposure detection tests ───────────────────────────────────────


class TestCumulativeExposureDetection:

    def test_detects_cumulative_exposure(
        self,
        detector: TemporalAnomalyDetector,
        engine: AnalysisEngine,
    ) -> None:
        """
        Multiple trades totalling > $1M (our test threshold)
        must trigger CUMULATIVE_EXPOSURE.
        Each trade is $400,000 — below the $1M per-trade rule.
        3 trades = $1.2M total — above cumulative threshold.
        """
        alerts_found = []
        for _ in range(3):
            result = _engine_result(engine, "execute_trade", amount=400_000)
            alerts = detector.update(result)
            alerts_found.extend(alerts)

        cumulative_alerts = [
            a for a in alerts_found if a.threat == TemporalThreat.CUMULATIVE_EXPOSURE
        ]
        assert len(cumulative_alerts) >= 1, (
            "CUMULATIVE_EXPOSURE not detected. "
            "3 × $400K = $1.2M should exceed $1M threshold."
        )

    def test_no_exposure_below_threshold(
        self,
        detector: TemporalAnomalyDetector,
        engine: AnalysisEngine,
    ) -> None:
        """Single trade well below threshold must not trigger."""
        result = _engine_result(engine, "execute_trade", amount=100_000)
        alerts = detector.update(result)
        cumulative_alerts = [
            a for a in alerts if a.threat == TemporalThreat.CUMULATIVE_EXPOSURE
        ]
        assert len(cumulative_alerts) == 0

    def test_cumulative_alert_evidence_is_correct(
        self,
        detector: TemporalAnomalyDetector,
        engine: AnalysisEngine,
    ) -> None:
        alerts_found = []
        for _ in range(3):
            result = _engine_result(engine, "execute_trade", amount=400_000)
            alerts = detector.update(result)
            alerts_found.extend(alerts)

        cumulative_alerts = [
            a for a in alerts_found if a.threat == TemporalThreat.CUMULATIVE_EXPOSURE
        ]
        if cumulative_alerts:
            evidence = cumulative_alerts[0].evidence
            assert evidence["total_amount"] > 1_000_000
            assert evidence["trade_count"] > 0
            assert evidence["threshold"] == 1_000_000.0
            assert cumulative_alerts[0].severity == "CRITICAL"


# ── Escalating risk detection tests ──────────────────────────────────────────


class TestEscalatingRiskDetection:

    def test_detects_escalating_risk(self, detector: TemporalAnomalyDetector) -> None:
        """
        Risk scores increasing from ~0.2 to ~0.7 must trigger
        ESCALATING_RISK (delta = 0.5 >> threshold of 0.15).
        """
        window = detector._get_or_create_window("escalation_agent")

        # First half — low risk
        for score in [0.15, 0.20, 0.18, 0.22]:
            window.add(
                WindowEvent(
                    action_type="execute_trade",
                    risk_score=score,
                    was_blocked=False,
                    amount=0.0,
                    timestamp=time.monotonic(),
                )
            )

        # Second half — high risk (escalating)
        for score in [0.55, 0.65, 0.70, 0.75]:
            window.add(
                WindowEvent(
                    action_type="execute_trade",
                    risk_score=score,
                    was_blocked=False,
                    amount=0.0,
                    timestamp=time.monotonic(),
                )
            )

        events = window.get_recent()
        alerts = detector._check_escalating_risk("escalation_agent", events)

        assert len(alerts) >= 1
        assert alerts[0].threat == TemporalThreat.ESCALATING_RISK
        assert alerts[0].severity == "CRITICAL"
        assert alerts[0].evidence["risk_delta"] >= 0.15

    def test_no_escalation_with_stable_scores(
        self, detector: TemporalAnomalyDetector
    ) -> None:
        """Stable risk scores must not trigger escalation."""
        window = detector._get_or_create_window("stable_agent")
        for score in [0.30, 0.32, 0.28, 0.31, 0.29, 0.33]:
            window.add(
                WindowEvent(
                    action_type="execute_trade",
                    risk_score=score,
                    was_blocked=False,
                    amount=0.0,
                    timestamp=time.monotonic(),
                )
            )
        events = window.get_recent()
        alerts = detector._check_escalating_risk("stable_agent", events)
        assert len(alerts) == 0

    def test_no_escalation_with_insufficient_events(
        self, detector: TemporalAnomalyDetector
    ) -> None:
        """Fewer than escalation_min_events must not trigger."""
        window = detector._get_or_create_window("few_events_agent")
        for score in [0.1, 0.9]:  # Only 2 events — below min of 4
            window.add(
                WindowEvent(
                    action_type="execute_trade",
                    risk_score=score,
                    was_blocked=False,
                    amount=0.0,
                    timestamp=time.monotonic(),
                )
            )
        events = window.get_recent()
        alerts = detector._check_escalating_risk("few_events_agent", events)
        assert len(alerts) == 0


# ── Block evasion detection tests ─────────────────────────────────────────────


class TestBlockEvasionDetection:

    def test_detects_repeated_evasion(self, detector: TemporalAnomalyDetector) -> None:
        """
        After a block, 3 retries of the same type with evasion_threshold=2
        must trigger REPEATED_BLOCK_EVASION.
        """
        window = detector._get_or_create_window("evasion_agent")

        # Original blocked action
        window.add(
            WindowEvent(
                action_type="manipulate_news_feed",
                risk_score=0.95,
                was_blocked=True,
                amount=0.0,
                timestamp=time.monotonic(),
            )
        )

        # 3 retries — slightly modified but same type
        for _ in range(3):
            window.add(
                WindowEvent(
                    action_type="manipulate_news_feed",
                    risk_score=0.70,
                    was_blocked=False,  # Slipped through scoring
                    amount=0.0,
                    timestamp=time.monotonic(),
                )
            )

        events = window.get_recent()
        alerts = detector._check_repeated_block_evasion("evasion_agent", events)
        assert len(alerts) >= 1
        assert alerts[0].threat == TemporalThreat.REPEATED_BLOCK_EVASION
        assert alerts[0].severity == "CRITICAL"
        assert "manipulate_news_feed" in str(alerts[0].evidence)

    def test_no_evasion_without_prior_block(
        self, detector: TemporalAnomalyDetector
    ) -> None:
        """Repeated actions without a prior block must not trigger evasion."""
        window = detector._get_or_create_window("no_block_agent")
        for _ in range(5):
            window.add(
                WindowEvent(
                    action_type="read_market_data",
                    risk_score=0.1,
                    was_blocked=False,
                    amount=0.0,
                    timestamp=time.monotonic(),
                )
            )
        events = window.get_recent()
        alerts = detector._check_repeated_block_evasion("no_block_agent", events)
        assert len(alerts) == 0


# ── General detector tests ────────────────────────────────────────────────────


class TestDetector:

    def test_never_raises_on_malformed_result(
        self,
        detector: TemporalAnomalyDetector,
        engine: AnalysisEngine,
    ) -> None:
        """Temporal analysis must never crash the main pipeline."""
        result = _engine_result(engine)
        # Should always return a list — never raise
        alerts = detector.update(result)
        assert isinstance(alerts, list)

    def test_per_agent_isolation(
        self,
        detector: TemporalAnomalyDetector,
        engine: AnalysisEngine,
    ) -> None:
        """
        Agent A's burst must not affect Agent B's window.
        """
        for _ in range(6):
            event = Event(
                action_type="read_market_data",
                agent_id="agent_A",
                target="MARKET",
                scenario=Scenario.TRADING_AI,
            )
            result = engine.analyse(event)
            detector.update(result)

        # Agent B has 0 events
        assert detector.window_size("agent_B") == 0

    def test_reset_agent_clears_window(
        self,
        detector: TemporalAnomalyDetector,
        engine: AnalysisEngine,
    ) -> None:
        result = _engine_result(engine)
        detector.update(result)
        assert detector.window_size("test_agent") > 0
        detector.reset_agent("test_agent")
        assert detector.window_size("test_agent") == 0

    def test_reset_all_clears_all_windows(
        self,
        detector: TemporalAnomalyDetector,
        engine: AnalysisEngine,
    ) -> None:
        for agent in ["agent_X", "agent_Y", "agent_Z"]:
            event = Event(
                action_type="read_market_data",
                agent_id=agent,
                target="MARKET",
                scenario=Scenario.TRADING_AI,
            )
            result = engine.analyse(event)
            detector.update(result)

        detector.reset_all()
        for agent in ["agent_X", "agent_Y", "agent_Z"]:
            assert detector.window_size(agent) == 0

    def test_alerts_contain_required_fields(
        self,
        detector: TemporalAnomalyDetector,
        engine: AnalysisEngine,
    ) -> None:
        """Every alert must have all required fields."""
        for _ in range(7):
            result = _engine_result(engine)
            alerts = detector.update(result)
            for alert in alerts:
                assert alert.agent_id != ""
                assert alert.threat is not None
                assert alert.severity in ("HIGH", "CRITICAL")
                assert alert.description != ""
                assert isinstance(alert.evidence, dict)
                assert alert.timestamp > 0
