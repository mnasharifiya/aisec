"""
Simulation tests for the urban AI agent.

Run with: pytest tests/simulation/ -v
"""

from __future__ import annotations

from pathlib import Path

import pytest

from aisec.agents.urban_agent import (
    UrbanAgent,
    DANGEROUS_ACTIONS,
    SAFE_ACTIONS,
)
from aisec.core.engine import AnalysisEngine
from aisec.storage.models import Decision


@pytest.fixture
def engine(tmp_path: Path) -> AnalysisEngine:
    return AnalysisEngine(log_path=tmp_path / "audit.jsonl")


@pytest.fixture
def agent(engine: AnalysisEngine) -> UrbanAgent:
    return UrbanAgent(engine)


class TestUrbanAgentSecurity:

    def test_all_dangerous_actions_are_blocked(self, agent: UrbanAgent) -> None:
        """Every dangerous city action must be blocked or escalated."""
        for action in DANGEROUS_ACTIONS:
            result = agent.attempt_action(action)
            assert result.blocked, (
                f"SECURITY FAILURE: dangerous city action "
                f"'{action.action_type}' targeting '{action.target}' "
                f"was NOT blocked. Decision: {result.decision}."
            )

    def test_all_safe_actions_are_allowed(self, agent: UrbanAgent) -> None:
        """Routine city operations must not be blocked."""
        for action in SAFE_ACTIONS:
            result = agent.attempt_action(action)
            assert result.decision == Decision.ALLOW, (
                f"Safe city action '{action.action_type}' was blocked "
                f"with {result.decision}. False positive detected."
            )

    def test_simulation_run_blocks_dangerous_actions(self, agent: UrbanAgent) -> None:
        """Full simulation confirms dangerous actions are intercepted."""
        results = agent.run(steps=40)
        dangerous_types = {a.action_type for a in DANGEROUS_ACTIONS}
        for result in results:
            if result.event.action_type in dangerous_types:
                assert result.blocked, (
                    f"Urban dangerous action '{result.event.action_type}' "
                    "escaped AISec during simulation."
                )

    def test_every_action_is_audit_logged(
        self, agent: UrbanAgent, engine: AnalysisEngine
    ) -> None:
        steps = 16
        agent.run(steps=steps)
        assert engine.audit_count() == steps

    def test_audit_chain_remains_intact_after_simulation(
        self, agent: UrbanAgent, engine: AnalysisEngine
    ) -> None:
        agent.run(steps=30)
        ok, errors = engine.verify_audit_chain()
        assert ok is True, f"Chain broken after urban simulation: {errors}"

    def test_agent_rejects_non_engine_argument(self) -> None:
        with pytest.raises(TypeError, match="AnalysisEngine"):
            UrbanAgent(None)  # type: ignore[arg-type]

    def test_agent_rejects_zero_steps(self, agent: UrbanAgent) -> None:
        with pytest.raises(ValueError, match="steps"):
            agent.run(steps=0)

    def test_agent_id_is_always_class_constant(self, agent: UrbanAgent) -> None:
        """
        Agent identity submitted to the engine must always be
        the class constant — never anything else.
        """
        for action in SAFE_ACTIONS:
            result = agent.attempt_action(action)
            assert result.event.agent_id == UrbanAgent.AGENT_ID
            assert result.event.scenario == UrbanAgent.SCENARIO
