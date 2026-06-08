"""
Unit tests for the AISec multi-agent correlation detector.

Run with:
    pytest tests/unit/test_correlation.py -v
"""

from __future__ import annotations

import threading

import pytest

from aisec.security.correlation import (
    CorrelationAction,
    CorrelationConfig,
    CorrelationSeverity,
    CorrelationThreat,
    MultiAgentCorrelationDetector,
)


@pytest.fixture
def config() -> CorrelationConfig:
    return CorrelationConfig(
        window_seconds=60.0,
        coordinated_amount_threshold=1_000_000.0,
        sync_burst_threshold=3,
        sync_window_seconds=10.0,
        escalation_window_seconds=30.0,
        shared_target_agent_threshold=3,
        min_agents_for_correlation=2,
        max_events_tracked=10_000,
        max_agents_tracked=1_000,
        alert_cooldown_seconds=10.0,
        max_agents_in_evidence=20,
    )


@pytest.fixture
def detector(config: CorrelationConfig) -> MultiAgentCorrelationDetector:
    return MultiAgentCorrelationDetector(config=config)


def _submit(
    detector: MultiAgentCorrelationDetector,
    agent_id: str,
    action_type: str = "execute_trade",
    amount: float = 0.0,
    was_blocked: bool = False,
    risk_score: float = 0.3,
    target: str = "MARKET",
    timestamp: float | None = None,
) -> list:
    return detector.update(
        agent_id=agent_id,
        action_type=action_type,
        risk_score=risk_score,
        was_blocked=was_blocked,
        amount=amount,
        target=target,
        timestamp=timestamp,
    )


class TestCoordinatedThresholdEvasion:

    def test_detects_coordinated_amount_across_agents(
        self,
        detector: MultiAgentCorrelationDetector,
    ) -> None:
        _submit(detector, "bot_a", amount=600_000, timestamp=1000.0)
        alerts = _submit(detector, "bot_b", amount=600_000, timestamp=1001.0)

        coord_alerts = [
            alert
            for alert in alerts
            if alert.threat == CorrelationThreat.COORDINATED_THRESHOLD_EVASION
        ]

        assert len(coord_alerts) == 1
        alert = coord_alerts[0]
        assert alert.severity == CorrelationSeverity.CRITICAL
        assert alert.recommended_action == CorrelationAction.REVIEW
        assert alert.correlation_score > 0.0
        assert "bot_a" in alert.agents
        assert "bot_b" in alert.agents
        assert alert.evidence["total_amount"] >= 1_000_000
        assert alert.fingerprint

    def test_no_alert_when_single_agent(
        self,
        detector: MultiAgentCorrelationDetector,
    ) -> None:
        alerts = _submit(detector, "bot_a", amount=1_200_000, timestamp=1000.0)

        coord_alerts = [
            alert
            for alert in alerts
            if alert.threat == CorrelationThreat.COORDINATED_THRESHOLD_EVASION
        ]

        assert coord_alerts == []

    def test_no_alert_below_threshold(
        self,
        detector: MultiAgentCorrelationDetector,
    ) -> None:
        _submit(detector, "bot_a", amount=200_000, timestamp=1000.0)
        alerts = _submit(detector, "bot_b", amount=200_000, timestamp=1001.0)

        coord_alerts = [
            alert
            for alert in alerts
            if alert.threat == CorrelationThreat.COORDINATED_THRESHOLD_EVASION
        ]

        assert coord_alerts == []

    def test_evidence_contains_top_agents(
        self,
        detector: MultiAgentCorrelationDetector,
    ) -> None:
        _submit(detector, "bot_a", amount=700_000, timestamp=1000.0)
        alerts = _submit(detector, "bot_b", amount=400_000, timestamp=1001.0)

        coord_alerts = [
            alert
            for alert in alerts
            if alert.threat == CorrelationThreat.COORDINATED_THRESHOLD_EVASION
        ]

        assert coord_alerts
        evidence = coord_alerts[0].evidence
        assert "top_agents" in evidence
        assert "total_amount" in evidence
        assert "agent_count" in evidence
        assert evidence["total_amount"] >= 1_000_000


class TestSynchronizedBurst:

    def test_detects_sync_burst_across_agents(
        self,
        detector: MultiAgentCorrelationDetector,
    ) -> None:
        _submit(detector, "bot_a", action_type="manipulate_news_feed", timestamp=1000.0)
        _submit(detector, "bot_b", action_type="manipulate_news_feed", timestamp=1001.0)
        alerts = _submit(
            detector,
            "bot_c",
            action_type="manipulate_news_feed",
            timestamp=1002.0,
        )

        burst_alerts = [
            alert
            for alert in alerts
            if alert.threat == CorrelationThreat.SYNCHRONIZED_ACTION_BURST
        ]

        assert len(burst_alerts) == 1
        assert burst_alerts[0].severity == CorrelationSeverity.HIGH
        assert burst_alerts[0].recommended_action == CorrelationAction.REVIEW

    def test_no_burst_with_different_actions(
        self,
        detector: MultiAgentCorrelationDetector,
    ) -> None:
        _submit(detector, "bot_a", action_type="execute_trade", timestamp=1000.0)
        _submit(detector, "bot_b", action_type="read_market_data", timestamp=1001.0)
        alerts = _submit(detector, "bot_c", action_type="set_curfew", timestamp=1002.0)

        burst_alerts = [
            alert
            for alert in alerts
            if alert.threat == CorrelationThreat.SYNCHRONIZED_ACTION_BURST
        ]

        assert burst_alerts == []

    def test_no_burst_below_threshold(
        self,
        detector: MultiAgentCorrelationDetector,
    ) -> None:
        _submit(detector, "bot_a", action_type="execute_trade", timestamp=1000.0)
        alerts = _submit(
            detector, "bot_b", action_type="execute_trade", timestamp=1001.0
        )

        burst_alerts = [
            alert
            for alert in alerts
            if alert.threat == CorrelationThreat.SYNCHRONIZED_ACTION_BURST
        ]

        assert burst_alerts == []

    def test_no_burst_outside_sync_window(
        self,
        detector: MultiAgentCorrelationDetector,
    ) -> None:
        _submit(detector, "bot_a", action_type="execute_trade", timestamp=1000.0)
        _submit(detector, "bot_b", action_type="execute_trade", timestamp=1001.0)
        alerts = _submit(
            detector, "bot_c", action_type="execute_trade", timestamp=1020.0
        )

        burst_alerts = [
            alert
            for alert in alerts
            if alert.threat == CorrelationThreat.SYNCHRONIZED_ACTION_BURST
        ]

        assert burst_alerts == []


class TestCrossAgentEscalation:

    def test_detects_handoff_after_block(
        self,
        detector: MultiAgentCorrelationDetector,
    ) -> None:
        _submit(
            detector,
            "bot_a",
            action_type="override_risk_limit",
            was_blocked=True,
            risk_score=0.95,
            timestamp=1000.0,
        )

        alerts = _submit(
            detector,
            "bot_b",
            action_type="override_risk_limit",
            was_blocked=False,
            risk_score=0.50,
            timestamp=1005.0,
        )

        escalation_alerts = [
            alert
            for alert in alerts
            if alert.threat == CorrelationThreat.CROSS_AGENT_ESCALATION
        ]

        assert len(escalation_alerts) == 1
        assert "bot_a" in escalation_alerts[0].agents
        assert "bot_b" in escalation_alerts[0].agents
        assert escalation_alerts[0].evidence["new_agent"] == "bot_b"

    def test_no_escalation_same_agent(
        self,
        detector: MultiAgentCorrelationDetector,
    ) -> None:
        _submit(
            detector,
            "bot_a",
            action_type="execute_trade",
            was_blocked=True,
            timestamp=1000.0,
        )

        alerts = _submit(
            detector,
            "bot_a",
            action_type="execute_trade",
            timestamp=1001.0,
        )

        escalation_alerts = [
            alert
            for alert in alerts
            if alert.threat == CorrelationThreat.CROSS_AGENT_ESCALATION
        ]

        assert escalation_alerts == []

    def test_no_escalation_different_action(
        self,
        detector: MultiAgentCorrelationDetector,
    ) -> None:
        _submit(
            detector,
            "bot_a",
            action_type="execute_trade",
            was_blocked=True,
            timestamp=1000.0,
        )

        alerts = _submit(
            detector,
            "bot_b",
            action_type="read_market_data",
            timestamp=1001.0,
        )

        escalation_alerts = [
            alert
            for alert in alerts
            if alert.threat == CorrelationThreat.CROSS_AGENT_ESCALATION
        ]

        assert escalation_alerts == []

    def test_no_escalation_outside_window(
        self,
        detector: MultiAgentCorrelationDetector,
    ) -> None:
        _submit(
            detector,
            "bot_a",
            action_type="override_risk_limit",
            was_blocked=True,
            timestamp=1000.0,
        )

        alerts = _submit(
            detector,
            "bot_b",
            action_type="override_risk_limit",
            timestamp=1040.0,
        )

        escalation_alerts = [
            alert
            for alert in alerts
            if alert.threat == CorrelationThreat.CROSS_AGENT_ESCALATION
        ]

        assert escalation_alerts == []


class TestSharedTargetAttack:

    def test_detects_shared_target(
        self,
        detector: MultiAgentCorrelationDetector,
    ) -> None:
        _submit(detector, "bot_a", target="risk_management_system", timestamp=1000.0)
        _submit(detector, "bot_b", target="risk_management_system", timestamp=1001.0)
        alerts = _submit(
            detector,
            "bot_c",
            target="risk_management_system",
            timestamp=1002.0,
        )

        target_alerts = [
            alert
            for alert in alerts
            if alert.threat == CorrelationThreat.SHARED_TARGET_ATTACK
        ]

        assert len(target_alerts) == 1
        assert target_alerts[0].severity == CorrelationSeverity.HIGH
        assert target_alerts[0].recommended_action == CorrelationAction.REVIEW
        assert target_alerts[0].evidence["target"] == "risk_management_system"

    def test_no_alert_for_generic_targets(
        self,
        detector: MultiAgentCorrelationDetector,
    ) -> None:
        _submit(detector, "bot_a", target="MARKET", timestamp=1000.0)
        _submit(detector, "bot_b", target="MARKET", timestamp=1001.0)
        alerts = _submit(detector, "bot_c", target="MARKET", timestamp=1002.0)

        target_alerts = [
            alert
            for alert in alerts
            if alert.threat == CorrelationThreat.SHARED_TARGET_ATTACK
        ]

        assert target_alerts == []


class TestExpiryAndDeduplication:

    def test_old_events_expire(
        self,
        detector: MultiAgentCorrelationDetector,
    ) -> None:
        _submit(detector, "bot_a", amount=600_000, timestamp=1000.0)
        _submit(detector, "bot_b", amount=600_000, timestamp=1001.0)

        # Move beyond 60s window.
        _submit(detector, "bot_c", amount=1.0, timestamp=1070.0)

        assert detector.active_agent_count() == 1

    def test_duplicate_alert_suppressed_inside_cooldown(
        self,
        detector: MultiAgentCorrelationDetector,
    ) -> None:
        _submit(detector, "bot_a", amount=600_000, timestamp=1000.0)
        first_alerts = _submit(detector, "bot_b", amount=600_000, timestamp=1001.0)

        # Same agents, same pattern, inside cooldown.
        duplicate_alerts = _submit(detector, "bot_b", amount=1.0, timestamp=1002.0)

        first_coord = [
            alert
            for alert in first_alerts
            if alert.threat == CorrelationThreat.COORDINATED_THRESHOLD_EVASION
        ]
        duplicate_coord = [
            alert
            for alert in duplicate_alerts
            if alert.threat == CorrelationThreat.COORDINATED_THRESHOLD_EVASION
        ]

        assert len(first_coord) == 1
        assert duplicate_coord == []

    def test_alert_repeats_after_cooldown(
        self,
        config: CorrelationConfig,
    ) -> None:
        cfg = CorrelationConfig(
            window_seconds=60.0,
            coordinated_amount_threshold=1_000_000.0,
            sync_burst_threshold=3,
            sync_window_seconds=10.0,
            escalation_window_seconds=30.0,
            shared_target_agent_threshold=3,
            min_agents_for_correlation=2,
            alert_cooldown_seconds=2.0,
        )
        detector = MultiAgentCorrelationDetector(config=cfg)

        _submit(detector, "bot_a", amount=600_000, timestamp=1000.0)
        first_alerts = _submit(detector, "bot_b", amount=600_000, timestamp=1001.0)

        second_alerts = _submit(detector, "bot_b", amount=1.0, timestamp=1004.0)

        first_coord = [
            alert
            for alert in first_alerts
            if alert.threat == CorrelationThreat.COORDINATED_THRESHOLD_EVASION
        ]
        second_coord = [
            alert
            for alert in second_alerts
            if alert.threat == CorrelationThreat.COORDINATED_THRESHOLD_EVASION
        ]

        assert len(first_coord) == 1
        assert len(second_coord) == 1


class TestSanitisationAndBounds:

    def test_labels_are_sanitised(
        self,
        detector: MultiAgentCorrelationDetector,
    ) -> None:
        _submit(
            detector,
            "bot_a\nLOG_INJECTION",
            action_type="execute_trade\r\nbad",
            amount=600_000,
            target="risk\nsystem",
            timestamp=1000.0,
        )
        alerts = _submit(
            detector,
            "bot_b",
            action_type="execute_trade",
            amount=600_000,
            target="risk\nsystem",
            timestamp=1001.0,
        )

        assert alerts
        for alert in alerts:
            assert all("\n" not in agent for agent in alert.agents)
            assert "\n" not in str(alert.evidence)

    def test_evidence_is_bounded_for_many_agents(self) -> None:
        cfg = CorrelationConfig(
            coordinated_amount_threshold=100_000.0,
            min_agents_for_correlation=2,
            max_agents_in_evidence=5,
        )
        detector = MultiAgentCorrelationDetector(config=cfg)

        alerts = []
        for i in range(20):
            alerts = _submit(
                detector,
                f"bot_{i}",
                amount=10_000,
                target="MARKET",
                timestamp=1000.0 + i,
            )

        coord_alerts = [
            alert
            for alert in alerts
            if alert.threat == CorrelationThreat.COORDINATED_THRESHOLD_EVASION
        ]

        assert coord_alerts
        alert = coord_alerts[0]
        assert len(alert.agents) <= 5
        assert len(alert.evidence["top_agents"]) <= 5
        assert alert.evidence["evidence_bounded"] is True

    def test_max_events_tracked_is_enforced(self) -> None:
        cfg = CorrelationConfig(
            max_events_tracked=5,
            min_agents_for_correlation=1,
        )
        detector = MultiAgentCorrelationDetector(config=cfg)

        for i in range(20):
            _submit(detector, f"bot_{i}", timestamp=1000.0 + i)

        assert detector.event_count() <= 5


class TestDetectorGeneral:

    def test_never_raises_on_any_input(
        self,
        detector: MultiAgentCorrelationDetector,
    ) -> None:
        for i in range(10):
            alerts = detector.update(
                agent_id=f"bot_{i}",
                action_type="execute_trade",
                risk_score=0.5,
                was_blocked=False,
                amount=100_000,
                target="NYSE",
                timestamp=1000.0 + i,
            )
            assert isinstance(alerts, list)

    def test_invalid_numeric_inputs_do_not_crash(
        self,
        detector: MultiAgentCorrelationDetector,
    ) -> None:
        alerts = detector.update(
            agent_id="bot_a",
            action_type="execute_trade",
            risk_score="bad",  # type: ignore[arg-type]
            was_blocked=False,
            amount="bad",  # type: ignore[arg-type]
            target="NYSE",
            timestamp=1000.0,
        )

        assert isinstance(alerts, list)

    def test_active_agent_count(
        self,
        detector: MultiAgentCorrelationDetector,
    ) -> None:
        assert detector.active_agent_count() == 0
        _submit(detector, "bot_a", timestamp=1000.0)
        _submit(detector, "bot_b", timestamp=1001.0)
        assert detector.active_agent_count() == 2

    def test_reset_clears_all_events(
        self,
        detector: MultiAgentCorrelationDetector,
    ) -> None:
        _submit(detector, "bot_a", amount=500_000, timestamp=1000.0)
        _submit(detector, "bot_b", amount=500_000, timestamp=1001.0)
        detector.reset()

        assert detector.active_agent_count() == 0
        assert detector.event_count() == 0

    def test_thread_safety(
        self,
        detector: MultiAgentCorrelationDetector,
    ) -> None:
        errors: list[Exception] = []

        def submit_many(agent_id: str) -> None:
            try:
                for i in range(20):
                    detector.update(
                        agent_id=agent_id,
                        action_type="execute_trade",
                        risk_score=0.3,
                        was_blocked=False,
                        amount=50_000,
                        target="NYSE",
                        timestamp=1000.0 + i,
                    )
            except Exception as exc:
                errors.append(exc)

        threads = [
            threading.Thread(target=submit_many, args=(f"bot_{i}",)) for i in range(10)
        ]

        for thread in threads:
            thread.start()

        for thread in threads:
            thread.join()

        assert errors == []
