"""
Integration tests for the LangChain callback adapter.

These tests require LangChain to be installed:
    pip install langchain langchain-core

Tests are automatically skipped if LangChain is not available.

Run with: pytest tests/integration/test_langchain_adapter.py -v
"""

from __future__ import annotations

import threading
from pathlib import Path
from uuid import UUID, uuid4

import pytest

# Skip entire module gracefully if LangChain is not installed.
# This allows the test suite to run without LangChain in CI
# environments where it is not needed.
pytest.importorskip(
    "langchain_core",
    reason="LangChain not installed. Run: pip install langchain langchain-core",
)

from aisec.integrations.langchain import (
    AISeCCallbackHandler,
    AISeCSecurityError,
    _extract_payload,
    _hash_input,
    _sanitise_input,
    _sanitise_tool_name,
)
from aisec.core.engine import AnalysisEngine
from aisec.storage.models import Decision, Scenario


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def engine(tmp_path: Path) -> AnalysisEngine:
    return AnalysisEngine(log_path=tmp_path / "langchain_test.jsonl")


@pytest.fixture
def trading_handler(engine: AnalysisEngine) -> AISeCCallbackHandler:
    return AISeCCallbackHandler(
        engine=engine,
        scenario=Scenario.TRADING_AI,
        agent_id="test_trading_bot",
    )


@pytest.fixture
def urban_handler(engine: AnalysisEngine) -> AISeCCallbackHandler:
    return AISeCCallbackHandler(
        engine=engine,
        scenario=Scenario.URBAN_AI,
        agent_id="test_urban_ctrl",
    )


def _run_id() -> UUID:
    return uuid4()


def _serialized(name: str) -> dict:
    return {"name": name, "id": ["tools", name]}


# ── Input sanitisation tests ──────────────────────────────────────────────────

class TestInputSanitisation:

    def test_sanitise_tool_name_allows_safe_chars(self) -> None:
        assert _sanitise_tool_name("execute_trade") == "execute_trade"
        assert _sanitise_tool_name("read-sensor.v2") == "read-sensor.v2"

    def test_sanitise_tool_name_removes_dangerous_chars(self) -> None:
        result = _sanitise_tool_name("tool; DROP TABLE audit;--")
        # Semicolons, spaces, and SQL special chars must be removed
        assert ";" not in result
        assert " " not in result
        # The sanitised result must only contain safe characters
        assert all(c.isalnum() or c in "-_." for c in result)

    def test_sanitise_tool_name_truncates_long_names(self) -> None:
        long_name = "a" * 200
        assert len(_sanitise_tool_name(long_name)) <= 128

    def test_sanitise_tool_name_handles_empty(self) -> None:
        assert _sanitise_tool_name("") == "unknown_tool"
        assert _sanitise_tool_name(None) == "unknown_tool"  # type: ignore

    def test_sanitise_input_truncates_at_limit(self) -> None:
        huge = "x" * 10_000
        result = _sanitise_input(huge)
        assert len(result) <= 2_048

    def test_sanitise_input_handles_non_string(self) -> None:
        result = _sanitise_input({"key": "value"})
        assert isinstance(result, str)
        assert len(result) > 0

    def test_hash_input_returns_16_char_hex(self) -> None:
        h = _hash_input("test input")
        assert len(h) == 16
        assert all(c in "0123456789abcdef" for c in h)

    def test_hash_input_is_deterministic(self) -> None:
        assert _hash_input("same") == _hash_input("same")

    def test_hash_input_differs_for_different_inputs(self) -> None:
        assert _hash_input("buy") != _hash_input("sell")


# ── Payload extraction tests ──────────────────────────────────────────────────

class TestPayloadExtraction:

    def test_extracts_numeric_amount(self) -> None:
        payload = _extract_payload("execute_trade", "amount=2400000 action=sell")
        assert "amount" in payload
        assert payload["amount"] == 2_400_000.0

    def test_extracts_after_hours_flag(self) -> None:
        payload = _extract_payload("execute_trade", "after_hours execution")
        assert payload.get("after_hours") is True

    def test_extracts_zone_information(self) -> None:
        payload = _extract_payload("set_curfew", "zone=ALL duration=48h")
        assert payload.get("zone") == "ALL"

    def test_input_hash_always_present(self) -> None:
        payload = _extract_payload("read_sensor", "data")
        assert "input_hash" in payload
        assert len(payload["input_hash"]) == 16

    def test_input_length_always_present(self) -> None:
        payload = _extract_payload("read_sensor", "short")
        assert "input_length" in payload


# ── Handler construction tests ────────────────────────────────────────────────

class TestHandlerConstruction:

    def test_rejects_non_engine(self) -> None:
        with pytest.raises(TypeError, match="AnalysisEngine"):
            AISeCCallbackHandler(engine="not_an_engine")  # type: ignore

    def test_agent_id_sanitised_at_construction(
        self, engine: AnalysisEngine
    ) -> None:
        handler = AISeCCallbackHandler(
            engine=engine,
            agent_id="agent; rm -rf /",
        )
        assert ";" not in handler.agent_id
        assert "rm" not in handler.agent_id

    def test_short_agent_id_replaced_with_default(
        self, engine: AnalysisEngine
    ) -> None:
        handler = AISeCCallbackHandler(engine=engine, agent_id="ab")
        assert handler.agent_id == "langchain_agent"

    def test_agent_id_is_read_only(
        self, trading_handler: AISeCCallbackHandler
    ) -> None:
        with pytest.raises(AttributeError):
            trading_handler.agent_id = "attacker"  # type: ignore

    def test_scenario_is_read_only(
        self, trading_handler: AISeCCallbackHandler
    ) -> None:
        with pytest.raises(AttributeError):
            trading_handler.scenario = Scenario.URBAN_AI  # type: ignore

    def test_repr_shows_safe_info(
        self, trading_handler: AISeCCallbackHandler
    ) -> None:
        r = repr(trading_handler)
        assert "test_trading_bot" in r
        assert "trading_ai" in r


# ── Security interception tests ───────────────────────────────────────────────

class TestSecurityInterception:

    def test_dangerous_tool_is_blocked(
        self, trading_handler: AISeCCallbackHandler
    ) -> None:
        with pytest.raises(AISeCSecurityError) as exc_info:
            trading_handler.on_tool_start(
                serialized=_serialized("manipulate_news_feed"),
                input_str="inject fake earnings data",
                run_id=_run_id(),
            )
        assert exc_info.value.decision in (Decision.BLOCK, Decision.ESCALATE)
        assert exc_info.value.risk_score > 0.0

    def test_large_trade_is_blocked(
        self, trading_handler: AISeCCallbackHandler
    ) -> None:
        with pytest.raises(AISeCSecurityError) as exc_info:
            trading_handler.on_tool_start(
                serialized=_serialized("execute_large_trade"),
                input_str="amount=2400000 action=sell",
                run_id=_run_id(),
            )
        err = exc_info.value
        assert err.decision in (Decision.BLOCK, Decision.ESCALATE,
                                 Decision.PENDING_REVIEW)

    def test_curfew_is_blocked(
        self, urban_handler: AISeCCallbackHandler
    ) -> None:
        with pytest.raises(AISeCSecurityError):
            urban_handler.on_tool_start(
                serialized=_serialized("set_curfew"),
                input_str="zone=ALL duration_hours=48",
                run_id=_run_id(),
            )

    def test_safe_tool_is_allowed(
        self, trading_handler: AISeCCallbackHandler
    ) -> None:
        # Should NOT raise — safe action must pass through
        trading_handler.on_tool_start(
            serialized=_serialized("read_market_data"),
            input_str="symbol=AAPL",
            run_id=_run_id(),
        )

    def test_safe_sensor_read_is_allowed(
        self, urban_handler: AISeCCallbackHandler
    ) -> None:
        urban_handler.on_tool_start(
            serialized=_serialized("read_sensor"),
            input_str="sensor_id=traffic_north_42",
            run_id=_run_id(),
        )

    def test_call_count_increments(
        self, trading_handler: AISeCCallbackHandler
    ) -> None:
        assert trading_handler.call_count == 0
        trading_handler.on_tool_start(
            serialized=_serialized("read_market_data"),
            input_str="symbol=AAPL",
            run_id=_run_id(),
        )
        assert trading_handler.call_count == 1

    def test_error_exception_blocks_action(
        self, trading_handler: AISeCCallbackHandler, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """
        If the engine itself raises unexpectedly — fail closed.
        AISec must never allow an action when it cannot analyse it.
        """
        def broken_analyse(*args, **kwargs):
            raise RuntimeError("Simulated engine failure")

        monkeypatch.setattr(
            trading_handler._AISeCCallbackHandler__engine,
            "analyse",
            broken_analyse,
        )
        with pytest.raises(AISeCSecurityError) as exc_info:
            trading_handler.on_tool_start(
                serialized=_serialized("read_market_data"),
                input_str="safe action",
                run_id=_run_id(),
            )
        # Must block — never allow on engine failure
        assert exc_info.value.decision == Decision.BLOCK
        assert "INTERCEPTOR-ERROR" in exc_info.value.rule_hits

    def test_injection_attempt_in_tool_name_is_sanitised(
        self, trading_handler: AISeCCallbackHandler
    ) -> None:
        """
        A crafted tool name with injection payload must not
        bypass our rule engine or cause unexpected behaviour.
        """
        # This should not raise an unexpected exception —
        # it should either block or allow based on sanitised name
        try:
            trading_handler.on_tool_start(
                serialized={"name": "read_data; DROP TABLE audit;--",
                             "id": ["tools"]},
                input_str="safe input",
                run_id=_run_id(),
            )
        except AISeCSecurityError:
            pass   # Blocked is fine
        # What matters is no unexpected exception escapes


# ── Thread safety tests ───────────────────────────────────────────────────────

class TestThreadSafety:

    def test_concurrent_safe_calls_all_complete(
        self, trading_handler: AISeCCallbackHandler
    ) -> None:
        """
        Multiple threads calling on_tool_start simultaneously
        must all complete without data corruption.
        """
        errors: list[Exception] = []

        def call_handler():
            try:
                trading_handler.on_tool_start(
                    serialized=_serialized("read_market_data"),
                    input_str="symbol=AAPL",
                    run_id=_run_id(),
                )
            except AISeCSecurityError:
                pass   # Blocked is acceptable
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=call_handler) for _ in range(20)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert errors == [], (
            f"Thread safety failure — {len(errors)} unexpected errors: "
            f"{[str(e) for e in errors[:3]]}"
        )

    def test_call_count_accurate_under_concurrency(
        self, trading_handler: AISeCCallbackHandler
    ) -> None:
        """Call count must be accurate even under concurrent access."""
        def call_handler():
            try:
                trading_handler.on_tool_start(
                    serialized=_serialized("read_market_data"),
                    input_str="AAPL",
                    run_id=_run_id(),
                )
            except AISeCSecurityError:
                pass

        n = 10
        threads = [threading.Thread(target=call_handler) for _ in range(n)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert trading_handler.call_count == n


# ── Audit log integration ─────────────────────────────────────────────────────

class TestAuditIntegration:

    def test_every_intercepted_call_is_logged(
        self,
        engine: AnalysisEngine,
        trading_handler: AISeCCallbackHandler,
    ) -> None:
        """Every tool call — safe or blocked — must appear in audit log."""
        calls = 5
        for _ in range(calls):
            try:
                trading_handler.on_tool_start(
                    serialized=_serialized("read_market_data"),
                    input_str="AAPL",
                    run_id=_run_id(),
                )
            except AISeCSecurityError:
                pass

        assert engine.audit_count() == calls

    def test_audit_chain_intact_after_interceptions(
        self,
        engine: AnalysisEngine,
        trading_handler: AISeCCallbackHandler,
    ) -> None:
        tools = [
            ("read_market_data",     "AAPL",          False),
            ("manipulate_news_feed", "fake data",     True),
            ("read_market_data",     "MSFT",          False),
            ("execute_large_trade",  "amount=5000000",True),
            ("read_market_data",     "GOOG",          False),
        ]
        for tool, inp, should_block in tools:
            try:
                trading_handler.on_tool_start(
                    serialized=_serialized(tool),
                    input_str=inp,
                    run_id=_run_id(),
                )
                assert not should_block, f"{tool} should have been blocked"
            except AISeCSecurityError:
                assert should_block, f"{tool} should NOT have been blocked"

        ok, errors = engine.verify_audit_chain()
        assert ok is True, f"Audit chain broken: {errors}"