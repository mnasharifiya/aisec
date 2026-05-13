"""
tests/unit/test_interceptor.py
─────────────────────────────────────────────────────────────────────────────
Unit tests for aisec/core/interceptor.py.

Every test is independent. No test depends on another test's state.
Mocks are used for the engine and audit logger — the interceptor's
behaviour is what we are testing, not the engine's.

Coverage targets:
    ✔ Happy path — valid action → engine result returned
    ✔ ALLOW, BLOCK, PENDING_REVIEW, ESCALATE decisions passed through
    ✔ agent_id validation (empty, wrong type, too long)
    ✔ action_type validation (empty, wrong type, too long)
    ✔ params validation (wrong type, too many keys)
    ✔ scenario validation (unknown, wrong type)
    ✔ Engine exception → fail-closed BLOCK
    ✔ Audit logger exception → does not raise, still blocks
    ✔ Stats tracking — correct counters after mixed decisions
    ✔ Params are deep-copied — mutation after submission has no effect
    ✔ InterceptionResult.allowed property
    ✔ InterceptionResult.pre_engine_rejection property
    ✔ interception_id is a valid UUID
    ✔ timestamp is present and non-empty
"""

from __future__ import annotations

import uuid
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from aisec.core.interceptor import (
    MAX_FIELD_LENGTH,
    MAX_PARAM_KEYS,
    Interceptor,
    InterceptionError,
    InterceptionResult,
)
from aisec.storage.models import AnalysisResult, Decision, Scenario, Severity


# ── Fixtures ─────────────────────────────────────────────────────────────────

def _make_analysis(decision: Decision = Decision.ALLOW) -> AnalysisResult:
    """Return a minimal valid AnalysisResult with the given decision."""
    return AnalysisResult(
        event_id            = str(uuid.uuid4()),
        decision            = decision,
        risk_score          = 0.1,
        rule_hits           = [],
        explanation         = "test explanation",
        baseline_similarity = 1.0,
        risk_delta          = 0.0,
    )


def _make_interceptor(decision: Decision = Decision.ALLOW) -> tuple[Interceptor, MagicMock, MagicMock]:
    """
    Return (interceptor, mock_engine, mock_audit_logger).
    The engine is pre-configured to return the given decision.
    """
    engine = MagicMock()
    engine.analyse.return_value = _make_analysis(decision)

    audit = MagicMock()
    audit.log.return_value = None

    interceptor = Interceptor(engine=engine, audit_logger=audit, strict_mode=True)
    return interceptor, engine, audit


VALID_PARAMS = {
    "agent_id":    "trading_bot_01",
    "action_type": "execute_trade",
    "params":      {"symbol": "AAPL", "quantity": 500},
    "scenario":    "trading_ai",
}


# ── Construction ──────────────────────────────────────────────────────────────

class TestInterceptorConstruction:
    def test_raises_if_engine_is_none(self):
        audit = MagicMock()
        with pytest.raises(ValueError, match="engine"):
            Interceptor(engine=None, audit_logger=audit)

    def test_raises_if_audit_logger_is_none(self):
        engine = MagicMock()
        with pytest.raises(ValueError, match="audit logger"):
            Interceptor(engine=engine, audit_logger=None)

    def test_valid_construction(self):
        ic, _, _ = _make_interceptor()
        assert ic is not None


# ── Happy path ────────────────────────────────────────────────────────────────

class TestHappyPath:
    def test_allow_decision_is_passed_through(self):
        ic, engine, _ = _make_interceptor(Decision.ALLOW)
        result = ic.intercept(**VALID_PARAMS)
        assert result.decision == Decision.ALLOW
        assert result.allowed is True
        assert result.pre_engine_rejection is False

    def test_block_decision_is_passed_through(self):
        ic, engine, _ = _make_interceptor(Decision.BLOCK)
        result = ic.intercept(**VALID_PARAMS)
        assert result.decision == Decision.BLOCK
        assert result.allowed is False

    def test_pending_review_decision_is_passed_through(self):
        ic, engine, _ = _make_interceptor(Decision.PENDING_REVIEW)
        result = ic.intercept(**VALID_PARAMS)
        assert result.decision == Decision.PENDING_REVIEW
        assert result.allowed is False

    def test_escalate_decision_is_passed_through(self):
        ic, engine, _ = _make_interceptor(Decision.ESCALATE)
        result = ic.intercept(**VALID_PARAMS)
        assert result.decision == Decision.ESCALATE
        assert result.allowed is False

    def test_engine_is_called_once(self):
        ic, engine, _ = _make_interceptor()
        ic.intercept(**VALID_PARAMS)
        engine.analyse.assert_called_once()

    def test_result_contains_valid_uuid(self):
        ic, _, _ = _make_interceptor()
        result = ic.intercept(**VALID_PARAMS)
        parsed = uuid.UUID(result.interception_id)
        assert str(parsed) == result.interception_id

    def test_result_contains_timestamp(self):
        ic, _, _ = _make_interceptor()
        result = ic.intercept(**VALID_PARAMS)
        assert result.timestamp
        assert len(result.timestamp) > 10

    def test_result_contains_reason(self):
        ic, _, _ = _make_interceptor()
        result = ic.intercept(**VALID_PARAMS)
        assert isinstance(result.reason, str)
        assert len(result.reason) > 0

    def test_urban_ai_scenario_accepted(self):
        ic, _, _ = _make_interceptor()
        result = ic.intercept(
            agent_id="urban_bot_01",
            action_type="close_road",
            params={"road_id": "A1"},
            scenario="urban_ai",
        )
        assert result.allowed is True


# ── Params deep copy ──────────────────────────────────────────────────────────

class TestParamsDeepCopy:
    def test_mutation_after_submission_does_not_affect_event(self):
        ic, engine, _ = _make_interceptor()
        params = {"symbol": "AAPL", "quantity": 500}
        ic.intercept(
            agent_id="trading_bot",
            action_type="execute_trade",
            params=params,
            scenario="trading_ai",
        )
        # Mutate after submission
        params["symbol"] = "HACKED"
        params["injected"] = True

        # The event passed to the engine should have the original params
        call_args = engine.analyse.call_args[0][0]
        assert call_args.params["symbol"] == "AAPL"
        assert "injected" not in call_args.params


# ── agent_id validation ───────────────────────────────────────────────────────

class TestAgentIdValidation:
    def test_empty_agent_id_is_blocked(self):
        ic, _, _ = _make_interceptor()
        result = ic.intercept(agent_id="", action_type="execute_trade",
                              params={}, scenario="trading_ai")
        assert result.allowed is False
        assert result.pre_engine_rejection is True

    def test_whitespace_only_agent_id_is_blocked(self):
        ic, _, _ = _make_interceptor()
        result = ic.intercept(agent_id="   ", action_type="execute_trade",
                              params={}, scenario="trading_ai")
        assert result.allowed is False

    def test_non_string_agent_id_is_blocked(self):
        ic, _, _ = _make_interceptor()
        result = ic.intercept(agent_id=12345, action_type="execute_trade",
                              params={}, scenario="trading_ai")
        assert result.allowed is False

    def test_too_long_agent_id_is_blocked(self):
        ic, _, _ = _make_interceptor()
        result = ic.intercept(agent_id="a" * (MAX_FIELD_LENGTH + 1),
                              action_type="execute_trade",
                              params={}, scenario="trading_ai")
        assert result.allowed is False

    def test_max_length_agent_id_is_accepted(self):
        ic, _, _ = _make_interceptor()
        result = ic.intercept(agent_id="a" * MAX_FIELD_LENGTH,
                              action_type="execute_trade",
                              params={}, scenario="trading_ai")
        assert result.allowed is True


# ── action_type validation ───────────────────────────────────────────────────

class TestActionTypeValidation:
    def test_empty_action_type_is_blocked(self):
        ic, _, _ = _make_interceptor()
        result = ic.intercept(agent_id="bot", action_type="",
                              params={}, scenario="trading_ai")
        assert result.allowed is False

    def test_non_string_action_type_is_blocked(self):
        ic, _, _ = _make_interceptor()
        result = ic.intercept(agent_id="bot", action_type=None,
                              params={}, scenario="trading_ai")
        assert result.allowed is False

    def test_too_long_action_type_is_blocked(self):
        ic, _, _ = _make_interceptor()
        result = ic.intercept(agent_id="bot",
                              action_type="x" * (MAX_FIELD_LENGTH + 1),
                              params={}, scenario="trading_ai")
        assert result.allowed is False


# ─ params validation ─────────────────────────────────────────────────────────

class TestParamsValidation:
    def test_non_dict_params_is_blocked(self):
        ic, _, _ = _make_interceptor()
        result = ic.intercept(agent_id="bot", action_type="trade",
                              params=["not", "a", "dict"], scenario="trading_ai")
        assert result.allowed is False

    def test_too_many_param_keys_is_blocked(self):
        ic, _, _ = _make_interceptor()
        huge_params = {str(i): i for i in range(MAX_PARAM_KEYS + 1)}
        result = ic.intercept(agent_id="bot", action_type="trade",
                              params=huge_params, scenario="trading_ai")
        assert result.allowed is False

    def test_exactly_max_param_keys_is_accepted(self):
        ic, _, _ = _make_interceptor()
        max_params = {str(i): i for i in range(MAX_PARAM_KEYS)}
        result = ic.intercept(agent_id="bot", action_type="trade",
                              params=max_params, scenario="trading_ai")
        assert result.allowed is True

    def test_empty_params_dict_is_accepted(self):
        ic, _, _ = _make_interceptor()
        result = ic.intercept(agent_id="bot", action_type="trade",
                              params={}, scenario="trading_ai")
        assert result.allowed is True


# ── scenario validation ───────────────────────────────────────────────────────

class TestScenarioValidation:
    def test_unknown_scenario_is_blocked(self):
        ic, _, _ = _make_interceptor()
        result = ic.intercept(agent_id="bot", action_type="trade",
                              params={}, scenario="nuclear_ai")
        assert result.allowed is False

    def test_non_string_scenario_is_blocked(self):
        ic, _, _ = _make_interceptor()
        result = ic.intercept(agent_id="bot", action_type="trade",
                              params={}, scenario=42)
        assert result.allowed is False


# ── Fail-closed behaviour ─────────────────────────────────────────────────────

class TestFailClosed:
    def test_engine_exception_results_in_block(self):
        engine = MagicMock()
        engine.analyse.side_effect = RuntimeError("Engine exploded")
        audit = MagicMock()
        ic = Interceptor(engine=engine, audit_logger=audit)
        result = ic.intercept(**VALID_PARAMS)
        assert result.allowed is False
        assert result.pre_engine_rejection is True

    def test_engine_exception_increments_error_counter(self):
        engine = MagicMock()
        engine.analyse.side_effect = RuntimeError("boom")
        audit = MagicMock()
        ic = Interceptor(engine=engine, audit_logger=audit)
        ic.intercept(**VALID_PARAMS)
        assert ic.stats()["errors"] == 1

    def test_audit_logger_exception_does_not_raise(self):
        engine = MagicMock()
        engine.analyse.side_effect = RuntimeError("engine down")
        audit = MagicMock()
        audit.log.side_effect = IOError("disk full")
        ic = Interceptor(engine=engine, audit_logger=audit)
        # Must not raise — must return a BLOCK result silently
        result = ic.intercept(**VALID_PARAMS)
        assert result.allowed is False

    def test_intercept_never_raises(self):
        engine = MagicMock()
        engine.analyse.side_effect = Exception("catastrophic failure")
        audit = MagicMock()
        audit.log.side_effect = Exception("audit also broken")
        ic = Interceptor(engine=engine, audit_logger=audit)
        # This must NOT raise under any circumstances
        result = ic.intercept(**VALID_PARAMS)
        assert result is not None


# ── Statistics ────────────────────────────────────────────────────────────────

class TestStats:
    def test_stats_start_at_zero(self):
        ic, _, _ = _make_interceptor()
        s = ic.stats()
        assert s["total"] == 0
        assert s["allowed"] == 0

    def test_allowed_increments(self):
        ic, _, _ = _make_interceptor(Decision.ALLOW)
        ic.intercept(**VALID_PARAMS)
        assert ic.stats()["allowed"] == 1
        assert ic.stats()["total"] == 1

    def test_blocked_increments(self):
        ic, _, _ = _make_interceptor(Decision.BLOCK)
        ic.intercept(**VALID_PARAMS)
        assert ic.stats()["blocked"] == 1

    def test_escalated_increments(self):
        ic, _, _ = _make_interceptor(Decision.ESCALATE)
        ic.intercept(**VALID_PARAMS)
        assert ic.stats()["escalated"] == 1

    def test_reviewed_increments(self):
        ic, _, _ = _make_interceptor(Decision.PENDING_REVIEW)
        ic.intercept(**VALID_PARAMS)
        assert ic.stats()["reviewed"] == 1

    def test_mixed_decisions_tracked_correctly(self):
        ic, engine, _ = _make_interceptor()
        decisions = [
            Decision.ALLOW, Decision.ALLOW,
            Decision.BLOCK,
            Decision.ESCALATE,
            Decision.PENDING_REVIEW,
        ]
        for d in decisions:
            engine.analyse.return_value = _make_analysis(d)
            ic.intercept(**VALID_PARAMS)

        s = ic.stats()
        assert s["total"]     == 5
        assert s["allowed"]   == 2
        assert s["blocked"]   == 1
        assert s["escalated"] == 1
        assert s["reviewed"]  == 1

    def test_reset_stats(self):
        ic, _, _ = _make_interceptor(Decision.ALLOW)
        ic.intercept(**VALID_PARAMS)
        ic.reset_stats()
        assert ic.stats()["total"] == 0

    def test_stats_returns_snapshot_not_reference(self):
        ic, _, _ = _make_interceptor()
        s1 = ic.stats()
        ic.intercept(**VALID_PARAMS)
        s2 = ic.stats()
        # s1 should not have changed
        assert s1["total"] == 0
        assert s2["total"] == 1


# ── Audit logging ─────────────────────────────────────────────────────────────

class TestAuditLogging:
    def test_pre_engine_rejection_is_audited(self):
        ic, _, audit = _make_interceptor()
        ic.intercept(agent_id="", action_type="trade",
                     params={}, scenario="trading_ai")
        audit.log.assert_called_once()
        call_kwargs = audit.log.call_args
        assert call_kwargs is not None

    def test_valid_action_triggers_engine_analyse(self):
        ic, engine, _ = _make_interceptor()
        ic.intercept(**VALID_PARAMS)
        engine.analyse.assert_called_once()