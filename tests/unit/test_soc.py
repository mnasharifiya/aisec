"""
Unit tests for AISec SOC console.

Run with:
    pytest tests/unit/test_soc.py -v
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import click
import pytest
from click.testing import CliRunner

from aisec.cli.soc import (
    _display_metrics,
    _display_queue,
    _display_safe_state,
    _get_role,
    _release_safe_state,
    _resolve_event,
    soc_command,
)
from aisec.core.engine import AnalysisEngine
from aisec.security.rbac import (
    AccessDeniedError,
    Permission,
    Principal,
    RBACEnforcer,
    Role,
)
from aisec.storage.models import Event, Scenario


@pytest.fixture
def engine(tmp_path: Path) -> AnalysisEngine:
    return AnalysisEngine(log_path=tmp_path / "soc_test.jsonl")


@pytest.fixture
def analyst() -> Principal:
    return Principal("analyst_01", Role.ANALYST, "Alice Analyst")


@pytest.fixture
def admin() -> Principal:
    return Principal("admin_01", Role.ADMIN, "Bob Admin")


@pytest.fixture
def viewer() -> Principal:
    return Principal("viewer_01", Role.VIEWER, "Victor Viewer")


@pytest.fixture
def enforcer() -> RBACEnforcer:
    return RBACEnforcer()


def _log_analysis(
    engine: AnalysisEngine,
    *,
    record_id: str,
    decision: str = "PENDING_REVIEW",
    action_type: str = "execute_trade",
    agent_id: str = "bot",
    risk_score: float = 0.91,
    explanation: str = "Requires human review",
) -> None:
    engine._logger.log(
        record_type="analysis",
        record_id=record_id,
        payload={
            "decision": decision,
            "action_type": action_type,
            "agent_id": agent_id,
            "risk_score": risk_score,
            "explanation": explanation,
        },
    )


class TestSOCRoleParsing:
    def test_get_role_returns_analyst_role(self) -> None:
        assert _get_role("analyst") == Role.ANALYST

    def test_get_role_returns_admin_role(self) -> None:
        assert _get_role("admin") == Role.ADMIN

    def test_get_role_is_case_insensitive(self) -> None:
        assert _get_role("AnAlYsT") == Role.ANALYST
        assert _get_role("AdMiN") == Role.ADMIN

    def test_get_role_rejects_invalid_role(self) -> None:
        with pytest.raises(click.BadParameter):
            _get_role("viewer")


class TestSOCPermissions:
    def test_analyst_can_view_queue(
        self,
        enforcer: RBACEnforcer,
        analyst: Principal,
    ) -> None:
        decision = enforcer.require(analyst, Permission.VIEW_QUEUE)
        assert decision.allowed is True

    def test_analyst_can_resolve_queue(
        self,
        enforcer: RBACEnforcer,
        analyst: Principal,
    ) -> None:
        decision = enforcer.require(analyst, Permission.RESOLVE_QUEUE)
        assert decision.allowed is True

    def test_analyst_cannot_manage_safe_state(
        self,
        enforcer: RBACEnforcer,
        analyst: Principal,
    ) -> None:
        with pytest.raises(AccessDeniedError):
            enforcer.require(analyst, Permission.MANAGE_SAFE_STATE)

    def test_admin_can_manage_safe_state(
        self,
        enforcer: RBACEnforcer,
        admin: Principal,
    ) -> None:
        decision = enforcer.require(admin, Permission.MANAGE_SAFE_STATE)
        assert decision.allowed is True

    def test_admin_can_export_audit_log(
        self,
        enforcer: RBACEnforcer,
        admin: Principal,
    ) -> None:
        decision = enforcer.require(admin, Permission.EXPORT_AUDIT_LOG)
        assert decision.allowed is True

    def test_analyst_cannot_export_audit_log(
        self,
        enforcer: RBACEnforcer,
        analyst: Principal,
    ) -> None:
        with pytest.raises(AccessDeniedError):
            enforcer.require(analyst, Permission.EXPORT_AUDIT_LOG)


class TestSOCQueueDisplay:
    def test_empty_queue_prints_empty_message(
        self,
        engine: AnalysisEngine,
        enforcer: RBACEnforcer,
        analyst: Principal,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        result = _display_queue(engine, enforcer, analyst)

        captured = capsys.readouterr()
        assert result is None
        assert "Pending review: 0 event(s)" in captured.out
        assert "Queue is empty" in captured.out

    def test_pending_review_event_is_displayed(
        self,
        engine: AnalysisEngine,
        enforcer: RBACEnforcer,
        analyst: Principal,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        _log_analysis(
            engine,
            record_id="event-001",
            decision="PENDING_REVIEW",
            action_type="execute_trade",
            agent_id="trading_bot",
            risk_score=0.88,
            explanation="After-hours trade requires review",
        )

        unresolved = _display_queue(engine, enforcer, analyst)

        captured = capsys.readouterr()
        assert unresolved is not None
        assert len(unresolved) == 1
        assert unresolved[0].record_id == "event-001"
        assert "execute_trade" in captured.out
        assert "trading_bot" in captured.out
        assert "PENDING_REVIEW" in captured.out
        assert "event-001" in captured.out

    def test_escalated_event_is_displayed(
        self,
        engine: AnalysisEngine,
        enforcer: RBACEnforcer,
        analyst: Principal,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        _log_analysis(
            engine,
            record_id="event-002",
            decision="ESCALATE",
            action_type="disable_camera",
            agent_id="drone_01",
            risk_score=0.96,
            explanation="Drone action requires escalation",
        )

        unresolved = _display_queue(engine, enforcer, analyst)

        captured = capsys.readouterr()
        assert unresolved is not None
        assert len(unresolved) == 1
        assert unresolved[0].record_id == "event-002"
        assert "ESCALATE" in captured.out
        assert "drone_01" in captured.out

    def test_resolved_event_is_not_displayed(
        self,
        engine: AnalysisEngine,
        enforcer: RBACEnforcer,
        analyst: Principal,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        _log_analysis(
            engine,
            record_id="event-003",
            decision="PENDING_REVIEW",
        )

        engine._logger.log(
            record_type="analyst_decision",
            record_id="event-003",
            payload={
                "analyst_id": analyst.principal_id,
                "analyst_decision": "approve",
                "reason": "Already reviewed",
                "event_id": "event-003",
            },
        )

        unresolved = _display_queue(engine, enforcer, analyst)

        captured = capsys.readouterr()
        assert unresolved is None
        assert "Pending review: 0 event(s)" in captured.out
        assert "Queue is empty" in captured.out

    def test_viewer_can_view_queue_because_viewer_has_view_queue_permission(
        self,
        engine: AnalysisEngine,
        enforcer: RBACEnforcer,
        viewer: Principal,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        _log_analysis(engine, record_id="event-004")

        unresolved = _display_queue(engine, enforcer, viewer)

        captured = capsys.readouterr()
        assert unresolved is not None
        assert len(unresolved) == 1
        assert "event-004" in captured.out


class TestSOCResolveEvent:
    def test_analyst_can_resolve_event(
        self,
        engine: AnalysisEngine,
        enforcer: RBACEnforcer,
        analyst: Principal,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        _resolve_event(
            event_id="event-005",
            decision="approve",
            engine=engine,
            enforcer=enforcer,
            principal=analyst,
        )

        entries = engine._logger.get_all()
        decisions = [
            entry for entry in entries if entry.record_type == "analyst_decision"
        ]

        captured = capsys.readouterr()
        assert len(decisions) == 1
        assert decisions[0].record_id == "event-005"
        assert decisions[0].payload["analyst_id"] == "analyst_01"
        assert decisions[0].payload["analyst_decision"] == "approve"
        assert decisions[0].payload["event_id"] == "event-005"
        assert "Decision recorded" in captured.out

    def test_analyst_can_block_event(
        self,
        engine: AnalysisEngine,
        enforcer: RBACEnforcer,
        analyst: Principal,
    ) -> None:
        _resolve_event(
            event_id="event-006",
            decision="block",
            engine=engine,
            enforcer=enforcer,
            principal=analyst,
        )

        entries = engine._logger.get_all()
        decisions = [
            entry for entry in entries if entry.record_type == "analyst_decision"
        ]

        assert len(decisions) == 1
        assert decisions[0].payload["analyst_decision"] == "block"

    def test_viewer_cannot_resolve_event(
        self,
        engine: AnalysisEngine,
        enforcer: RBACEnforcer,
        viewer: Principal,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        _resolve_event(
            event_id="event-007",
            decision="approve",
            engine=engine,
            enforcer=enforcer,
            principal=viewer,
        )

        entries = engine._logger.get_all()
        decisions = [
            entry for entry in entries if entry.record_type == "analyst_decision"
        ]

        captured = capsys.readouterr()
        assert decisions == []
        assert "Access denied" in captured.out


class TestSOCSafeStateDisplay:
    def test_safe_state_empty_display(
        self,
        engine: AnalysisEngine,
        enforcer: RBACEnforcer,
        analyst: Principal,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        _display_safe_state(engine, enforcer, analyst)

        captured = capsys.readouterr()
        assert "Agents in safe state: 0" in captured.out
        assert "No agents currently restricted" in captured.out

    def test_safe_state_active_agent_displayed(
        self,
        engine: AnalysisEngine,
        enforcer: RBACEnforcer,
        analyst: Principal,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        engine.safe_state.enter_safe_state(
            "bot_v1",
            "test",
            "BURST_ATTACK",
        )

        _display_safe_state(engine, enforcer, analyst)

        captured = capsys.readouterr()
        assert "Agents in safe state: 1" in captured.out
        assert "bot_v1" in captured.out
        assert "RESTRICTED" in captured.out
        assert "BURST_ATTACK" in captured.out


class TestSOCSafeStateRelease:
    def test_admin_can_release_safe_state(
        self,
        engine: AnalysisEngine,
        enforcer: RBACEnforcer,
        admin: Principal,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        engine.safe_state.enter_safe_state(
            "bot_v1",
            "test",
            "BURST_ATTACK",
        )

        monkeypatch.setattr(click, "confirm", lambda *args, **kwargs: True)

        _release_safe_state(
            agent_id="bot_v1",
            engine=engine,
            enforcer=enforcer,
            principal=admin,
        )

        captured = capsys.readouterr()
        assert "released from safe state" in captured.out
        assert engine.safe_state.active_count() == 0

        entries = engine._logger.get_all()
        exits = [entry for entry in entries if entry.record_type == "safe_state_exit"]
        assert len(exits) >= 1
        assert exits[-1].payload["admin_id"] == "admin_01"

    def test_admin_release_can_be_cancelled(
        self,
        engine: AnalysisEngine,
        enforcer: RBACEnforcer,
        admin: Principal,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        engine.safe_state.enter_safe_state(
            "bot_v1",
            "test",
            "BURST_ATTACK",
        )

        monkeypatch.setattr(click, "confirm", lambda *args, **kwargs: False)

        _release_safe_state(
            agent_id="bot_v1",
            engine=engine,
            enforcer=enforcer,
            principal=admin,
        )

        captured = capsys.readouterr()
        assert "Cancelled" in captured.out
        assert engine.safe_state.active_count() == 1

    def test_analyst_cannot_release_safe_state(
        self,
        engine: AnalysisEngine,
        enforcer: RBACEnforcer,
        analyst: Principal,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        engine.safe_state.enter_safe_state(
            "bot_v1",
            "test",
            "BURST_ATTACK",
        )

        _release_safe_state(
            agent_id="bot_v1",
            engine=engine,
            enforcer=enforcer,
            principal=analyst,
        )

        captured = capsys.readouterr()
        assert "Access denied" in captured.out
        assert engine.safe_state.active_count() == 1


class TestSOCMetrics:
    def test_metrics_display_no_events(
        self,
        engine: AnalysisEngine,
        enforcer: RBACEnforcer,
        analyst: Principal,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        _display_metrics(engine, enforcer, analyst)

        captured = capsys.readouterr()
        assert "No events analysed yet" in captured.out

    def test_metrics_display_counts_events(
        self,
        engine: AnalysisEngine,
        enforcer: RBACEnforcer,
        analyst: Principal,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        _log_analysis(
            engine,
            record_id="allowed-001",
            decision="ALLOW",
            risk_score=0.10,
        )
        _log_analysis(
            engine,
            record_id="review-001",
            decision="PENDING_REVIEW",
            risk_score=0.70,
        )
        _log_analysis(
            engine,
            record_id="blocked-001",
            decision="BLOCK",
            risk_score=0.95,
        )
        _log_analysis(
            engine,
            record_id="escalated-001",
            decision="ESCALATE",
            risk_score=0.99,
        )

        _display_metrics(engine, enforcer, analyst)

        captured = capsys.readouterr()
        assert "Events analysed:  4" in captured.out
        assert "Blocked:" in captured.out
        assert "Pending review:" in captured.out
        assert "Allowed:" in captured.out
        assert "Audit chain:" in captured.out


class TestSOCQueueIntegration:
    def test_pending_review_events_appear_in_audit(
        self,
        engine: AnalysisEngine,
    ) -> None:
        engine.analyse(
            Event(
                action_type="execute_trade",
                agent_id="bot",
                target="MARKET",
                scenario=Scenario.TRADING_AI,
                raw_payload={"after_hours": True},
            )
        )

        entries = engine._logger.get_all()
        analysis = [entry for entry in entries if entry.record_type == "analysis"]

        assert len(analysis) >= 1

    def test_analyst_decision_is_logged(
        self,
        engine: AnalysisEngine,
        analyst: Principal,
    ) -> None:
        engine._logger.log(
            record_type="analyst_decision",
            record_id="test-event-001",
            payload={
                "analyst_id": analyst.principal_id,
                "analyst_decision": "approve",
                "reason": "Reviewed and safe",
                "event_id": "test-event-001",
            },
        )

        entries = engine._logger.get_all()
        decisions = [
            entry for entry in entries if entry.record_type == "analyst_decision"
        ]

        assert len(decisions) >= 1
        assert decisions[0].payload["analyst_id"] == "analyst_01"

    def test_safe_state_admin_release_logged(
        self,
        engine: AnalysisEngine,
        admin: Principal,
        enforcer: RBACEnforcer,
    ) -> None:
        engine.safe_state.enter_safe_state(
            "bot_v1",
            "test",
            "BURST_ATTACK",
        )

        enforcer.require(admin, Permission.MANAGE_SAFE_STATE)
        engine.safe_state.exit_safe_state("bot_v1", admin.principal_id)

        entries = engine._logger.get_all()
        exits = [entry for entry in entries if entry.record_type == "safe_state_exit"]

        assert len(exits) >= 1
        assert exits[0].payload["admin_id"] == "admin_01"

    def test_audit_chain_intact_after_soc_operations(
        self,
        engine: AnalysisEngine,
        analyst: Principal,
    ) -> None:
        for _ in range(5):
            engine.analyse(
                Event(
                    action_type="read_market_data",
                    agent_id="bot",
                    target="NYSE",
                    scenario=Scenario.TRADING_AI,
                )
            )

        engine._logger.log(
            record_type="analyst_decision",
            record_id="evt-001",
            payload={
                "analyst_id": analyst.principal_id,
                "analyst_decision": "approve",
                "reason": "Safe",
                "event_id": "evt-001",
            },
        )

        ok, errors = engine.verify_audit_chain()
        assert ok is True, f"Chain broken: {errors}"


class TestSOCCommand:
    def test_soc_command_help_loads(self) -> None:
        runner = CliRunner()
        result = runner.invoke(soc_command, ["--help"])

        assert result.exit_code == 0
        assert "Interactive SOC analyst console" in result.output
        assert "--role" in result.output
        assert "--principal-id" in result.output
        assert "--log-path" in result.output

    def test_soc_command_rejects_invalid_role(self) -> None:
        runner = CliRunner()
        result = runner.invoke(soc_command, ["--role", "viewer"])

        assert result.exit_code != 0
        assert "Invalid value" in result.output
