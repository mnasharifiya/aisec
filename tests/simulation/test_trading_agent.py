"""
Simulation tests for the trading agent.

These tests run the full agent simulation and verify that
AISec correctly intercepts dangerous actions while allowing
safe ones through.

Run with: pytest tests/simulation/ -v
"""

from __future__ import annotations

from pathlib import Path

import pytest

from aisec.agents.trading_agent import (
    TradingAgent,
    DANGEROUS_ACTIONS,
    SAFE_ACTIONS,
)
from aisec.core.engine import AnalysisEngine
from aisec.storage.models import Decision


@pytest.fixture
def engine(tmp_path: Path) -> AnalysisEngine:
    return AnalysisEngine(log_path=tmp_path / "audit.jsonl")


@pytest.fixture
def agent(engine: AnalysisEngine) -> TradingAgent:
    return TradingAgent(engine)


class TestTradingAgentSecurity:

    def test_all_dangerous_actions_are_blocked(self, agent: TradingAgent) -> None:
        """Every dangerous action must be blocked or escalated."""
        for action in DANGEROUS_ACTIONS:
            result = agent.attempt_action(action)
            assert result.blocked, (
                f"SECURITY FAILURE: dangerous action '{action.action_type}' "
                f"was NOT blocked. Decision: {result.decision}. "
                f"This means AISec failed to intercept a threat."
            )

    def test_all_safe_actions_are_allowed(self, agent: TradingAgent) -> None:
        """Every safe action must be allowed through."""
        for action in SAFE_ACTIONS:
            result = agent.attempt_action(action)
            assert result.decision == Decision.ALLOW, (
                f"Safe action '{action.action_type}' was unexpectedly "
                f"blocked with decision {result.decision}. "
                f"This indicates a false positive."
            )

    def test_simulation_run_blocks_dangerous_actions(self, agent: TradingAgent) -> None:
        """Run full simulation and confirm dangerous actions are blocked."""
        results = agent.run(steps=30)
        for result in results:
            if result.event.action_type in {a.action_type for a in DANGEROUS_ACTIONS}:
                assert result.blocked, (
                    f"Dangerous action '{result.event.action_type}' "
                    "was not blocked during simulation run."
                )

    def test_every_action_is_audit_logged(
        self, agent: TradingAgent, engine: AnalysisEngine
    ) -> None:
        """Every action — safe or dangerous — must appear in the audit log."""
        steps = 12
        agent.run(steps=steps)
        assert engine.audit_count() == steps

    def test_audit_chain_remains_intact_after_simulation(
        self, agent: TradingAgent, engine: AnalysisEngine
    ) -> None:
        """The hash chain must be intact after a full simulation run."""
        agent.run(steps=20)
        ok, errors = engine.verify_audit_chain()
        assert ok is True, f"Audit chain broken after simulation: {errors}"

    def test_agent_rejects_non_engine_argument(self) -> None:
        """Agent must reject invalid engine arguments at construction."""
        with pytest.raises(TypeError, match="AnalysisEngine"):
            TradingAgent("not_an_engine")  # type: ignore[arg-type]

    def test_agent_rejects_zero_steps(self, agent: TradingAgent) -> None:
        with pytest.raises(ValueError, match="steps"):
            agent.run(steps=0)

    def test_agent_id_is_always_class_constant(self, agent: TradingAgent) -> None:
        """
        Agent identity submitted to the engine must always be
        the class constant — never anything else.
        """
        for action in SAFE_ACTIONS:
            result = agent.attempt_action(action)
            assert result.event.agent_id == TradingAgent.AGENT_ID
            assert result.event.scenario == TradingAgent.SCENARIO
