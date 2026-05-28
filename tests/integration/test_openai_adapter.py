"""
Integration tests for the OpenAI function-calling adapter.

Tests do NOT require the OpenAI SDK to be installed.
Tool calls are represented as plain dicts matching the
OpenAI tool call structure.

Run with: pytest tests/integration/test_openai_adapter.py -v
"""

from __future__ import annotations

import threading
from pathlib import Path

import pytest

from aisec.integrations.openai_tools import (
    AISeCOpenAIInterceptor,
    AISeCOpenAISecurityError,
    BatchAnalysisResult,
    ToolCallResult,
    _build_payload,
    _extract_tool_call_fields,
    _hash_arguments,
    _parse_arguments_safely,
    _sanitise_tool_call_id,
    _validate_tool_name,
)
from aisec.core.engine import AnalysisEngine
from aisec.storage.models import Decision, Scenario

# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture
def engine(tmp_path: Path) -> AnalysisEngine:
    return AnalysisEngine(log_path=tmp_path / "openai_test.jsonl")


@pytest.fixture
def trading_interceptor(engine: AnalysisEngine) -> AISeCOpenAIInterceptor:
    return AISeCOpenAIInterceptor(
        engine=engine,
        scenario=Scenario.TRADING_AI,
        agent_id="test_trading_gpt4",
    )


@pytest.fixture
def urban_interceptor(engine: AnalysisEngine) -> AISeCOpenAIInterceptor:
    return AISeCOpenAIInterceptor(
        engine=engine,
        scenario=Scenario.URBAN_AI,
        agent_id="test_urban_gpt4",
    )


def _make_tool_call(
    name: str,
    arguments: dict,
    call_id: str = "call_test123",
) -> dict:
    """Build an OpenAI tool call dict for testing."""
    import json

    return {
        "id": call_id,
        "type": "function",
        "function": {
            "name": name,
            "arguments": json.dumps(arguments),
        },
    }


# ── Tool name validation tests ────────────────────────────────────────────────


class TestToolNameValidation:

    def test_accepts_valid_name(self) -> None:
        assert _validate_tool_name("execute_trade") == "execute_trade"

    def test_accepts_name_with_hyphen(self) -> None:
        assert _validate_tool_name("read-market-data") == "read-market-data"

    def test_rejects_name_with_spaces(self) -> None:
        with pytest.raises(ValueError):
            _validate_tool_name("execute trade")

    def test_rejects_sql_injection(self) -> None:
        with pytest.raises(ValueError):
            _validate_tool_name("execute; DROP TABLE audit;")

    def test_truncates_long_names(self) -> None:
        result = _validate_tool_name("a" * 100)
        assert len(result) <= 64

    def test_rejects_empty_name(self) -> None:
        with pytest.raises(ValueError):
            _validate_tool_name("")


# ── JSON argument parsing tests ───────────────────────────────────────────────


class TestArgumentParsing:

    def test_parses_valid_json(self) -> None:
        result = _parse_arguments_safely('{"amount": 5000, "symbol": "AAPL"}')
        assert result["amount"] == 5000
        assert result["symbol"] == "AAPL"

    def test_returns_empty_dict_for_malformed_json(self) -> None:
        result = _parse_arguments_safely("not valid json {{{")
        assert result == {}

    def test_returns_empty_dict_for_empty_string(self) -> None:
        assert _parse_arguments_safely("") == {}

    def test_truncates_before_parsing(self) -> None:
        huge = '{"key": "' + "x" * 10_000 + '"}'
        result = _parse_arguments_safely(huge)
        assert isinstance(result, dict)

    def test_handles_non_object_json(self) -> None:
        result = _parse_arguments_safely("[1, 2, 3]")
        assert isinstance(result, dict)

    def test_hash_is_16_chars(self) -> None:
        assert len(_hash_arguments('{"amount": 5000}')) == 16

    def test_hash_is_deterministic(self) -> None:
        s = '{"amount": 5000}'
        assert _hash_arguments(s) == _hash_arguments(s)


# ── Tool call field extraction tests ─────────────────────────────────────────


class TestToolCallExtraction:

    def test_extracts_from_dict(self) -> None:
        tc = _make_tool_call("execute_trade", {"amount": 5000})
        call_id, name, args = _extract_tool_call_fields(tc)
        assert call_id == "call_test123"
        assert name == "execute_trade"
        assert "5000" in args

    def test_extracts_from_object_with_attributes(self) -> None:
        """Test extraction from an object that mimics OpenAI SDK."""

        class MockFunction:
            name = "execute_trade"
            arguments = '{"amount": 5000}'

        class MockToolCall:
            id = "call_abc123"
            function = MockFunction()

        call_id, name, args = _extract_tool_call_fields(MockToolCall())
        assert call_id == "call_abc123"
        assert name == "execute_trade"

    def test_handles_missing_function_gracefully(self) -> None:
        """
        Missing function field should return empty strings, not crash.
        Graceful handling is safer than raising — the interceptor
        will then block the call due to invalid tool name.
        """
        call_id, func_name, arguments = _extract_tool_call_fields({"id": "x"})
        assert call_id == "x"
        assert isinstance(func_name, str)
        assert isinstance(arguments, str)


# ── Interceptor construction tests ────────────────────────────────────────────


class TestInterceptorConstruction:

    def test_rejects_non_engine(self) -> None:
        with pytest.raises(TypeError, match="AnalysisEngine"):
            AISeCOpenAIInterceptor(engine="not_an_engine")  # type: ignore

    def test_sanitises_agent_id(self, engine: AnalysisEngine) -> None:
        i = AISeCOpenAIInterceptor(engine=engine, agent_id="agent;DROP TABLE")
        assert ";" not in i.agent_id

    def test_short_id_replaced_with_default(self, engine: AnalysisEngine) -> None:
        i = AISeCOpenAIInterceptor(engine=engine, agent_id="ab")
        assert i.agent_id == "openai_agent"

    def test_agent_id_is_read_only(
        self, trading_interceptor: AISeCOpenAIInterceptor
    ) -> None:
        with pytest.raises(AttributeError):
            trading_interceptor.agent_id = "attacker"  # type: ignore

    def test_scenario_is_read_only(
        self, trading_interceptor: AISeCOpenAIInterceptor
    ) -> None:
        with pytest.raises(AttributeError):
            trading_interceptor.scenario = Scenario.URBAN_AI  # type: ignore

    def test_repr_shows_safe_info(
        self, trading_interceptor: AISeCOpenAIInterceptor
    ) -> None:
        r = repr(trading_interceptor)
        assert "test_trading_gpt4" in r
        assert "trading_ai" in r


# ── Single call analysis tests ────────────────────────────────────────────────


class TestSingleCallAnalysis:

    def test_safe_call_is_allowed(
        self, trading_interceptor: AISeCOpenAIInterceptor
    ) -> None:
        tc = _make_tool_call("read_market_data", {"symbol": "AAPL"})
        result = trading_interceptor.analyse_single_call(tc)
        assert result.allowed
        assert result.decision == Decision.ALLOW

    def test_dangerous_call_is_blocked(
        self, trading_interceptor: AISeCOpenAIInterceptor
    ) -> None:
        tc = _make_tool_call(
            "manipulate_news_feed",
            {"content": "fake earnings"},
        )
        with pytest.raises(AISeCOpenAISecurityError) as exc_info:
            trading_interceptor.analyse_single_call(tc)
        blocked = exc_info.value.blocked_calls
        assert len(blocked) == 1
        assert blocked[0].function_name == "manipulate_news_feed"

    def test_large_trade_is_blocked(
        self, trading_interceptor: AISeCOpenAIInterceptor
    ) -> None:
        tc = _make_tool_call(
            "execute_large_trade",
            {"amount": 2_400_000, "symbol": "AAPL"},
        )
        with pytest.raises(AISeCOpenAISecurityError):
            trading_interceptor.analyse_single_call(tc)

    def test_risk_limit_override_is_blocked(
        self, trading_interceptor: AISeCOpenAIInterceptor
    ) -> None:
        tc = _make_tool_call(
            "override_risk_limit",
            {"new_limit": 999_999_999},
        )
        with pytest.raises(AISeCOpenAISecurityError):
            trading_interceptor.analyse_single_call(tc)

    def test_curfew_is_blocked(self, urban_interceptor: AISeCOpenAIInterceptor) -> None:
        tc = _make_tool_call(
            "set_curfew",
            {"zone": "ALL", "duration_hours": 48},
        )
        with pytest.raises(AISeCOpenAISecurityError):
            urban_interceptor.analyse_single_call(tc)

    def test_sensor_read_is_allowed(
        self, urban_interceptor: AISeCOpenAIInterceptor
    ) -> None:
        tc = _make_tool_call("read_sensor", {"sensor_id": "traffic_42"})
        result = urban_interceptor.analyse_single_call(tc)
        assert result.allowed

    def test_no_raise_mode_returns_result_on_block(
        self, engine: AnalysisEngine
    ) -> None:
        interceptor = AISeCOpenAIInterceptor(
            engine=engine,
            scenario=Scenario.TRADING_AI,
            agent_id="no_raise_test",
            raise_on_block=False,
        )
        tc = _make_tool_call("manipulate_news_feed", {})
        result = interceptor.analyse_single_call(tc)
        assert result.blocked
        assert result.decision in (
            Decision.BLOCK,
            Decision.ESCALATE,
            Decision.PENDING_REVIEW,
        )

    def test_malformed_arguments_block_call(
        self, trading_interceptor: AISeCOpenAIInterceptor
    ) -> None:
        """Malformed JSON arguments must not crash AISec."""
        tc = {
            "id": "call_test",
            "type": "function",
            "function": {"name": "execute_trade", "arguments": "{malformed json{{{"},
        }
        # Should not crash — either allow or block gracefully
        try:
            result = trading_interceptor.analyse_single_call(tc)
            assert result is not None
        except AISeCOpenAISecurityError:
            pass  # Blocking malformed calls is also correct


# ── Batch analysis tests ──────────────────────────────────────────────────────


class TestBatchAnalysis:

    def test_all_safe_calls_allowed(
        self, trading_interceptor: AISeCOpenAIInterceptor
    ) -> None:
        calls = [
            _make_tool_call("read_market_data", {"symbol": "AAPL"}, "call_1"),
            _make_tool_call("read_market_data", {"symbol": "MSFT"}, "call_2"),
            _make_tool_call("read_market_data", {"symbol": "GOOG"}, "call_3"),
        ]
        batch = trading_interceptor.analyse_tool_calls(calls)
        assert not batch.any_blocked
        assert len(batch.allowed_calls) == 3
        assert batch.total == 3

    def test_mixed_batch_blocks_dangerous(
        self, trading_interceptor: AISeCOpenAIInterceptor
    ) -> None:
        calls = [
            _make_tool_call("read_market_data", {"symbol": "AAPL"}, "call_1"),
            _make_tool_call("manipulate_news_feed", {"content": "fake"}, "call_2"),
            _make_tool_call("read_market_data", {"symbol": "MSFT"}, "call_3"),
        ]
        with pytest.raises(AISeCOpenAISecurityError) as exc_info:
            trading_interceptor.analyse_tool_calls(calls)

        err = exc_info.value
        assert len(err.blocked_calls) >= 1
        blocked_names = [c.function_name for c in err.blocked_calls]
        assert "manipulate_news_feed" in blocked_names

    def test_all_calls_analysed_even_when_some_blocked(
        self, engine: AnalysisEngine
    ) -> None:
        """Every call must be analysed — blocked calls do not stop analysis."""
        interceptor = AISeCOpenAIInterceptor(
            engine=engine,
            scenario=Scenario.TRADING_AI,
            agent_id="batch_test",
            raise_on_block=False,
        )
        calls = [
            _make_tool_call("manipulate_news_feed", {}, "call_1"),
            _make_tool_call("read_market_data", {}, "call_2"),
            _make_tool_call("override_risk_limit", {}, "call_3"),
        ]
        batch = interceptor.analyse_tool_calls(calls)
        assert batch.total == 3
        assert engine.audit_count() == 3

    def test_empty_batch_returns_empty_result(
        self, trading_interceptor: AISeCOpenAIInterceptor
    ) -> None:
        batch = trading_interceptor.analyse_tool_calls([])
        assert batch.total == 0
        assert not batch.any_blocked

    def test_tool_call_id_preserved_in_result(
        self, trading_interceptor: AISeCOpenAIInterceptor
    ) -> None:
        calls = [
            _make_tool_call("read_market_data", {"symbol": "AAPL"}, "call_xyz_789"),
        ]
        batch = trading_interceptor.analyse_tool_calls(calls)
        assert batch.results[0].tool_call_id == "call_xyz_789"


# ── Monitoring tests ──────────────────────────────────────────────────────────


class TestMonitoring:

    def test_call_count_tracks_all_calls(
        self, trading_interceptor: AISeCOpenAIInterceptor
    ) -> None:
        calls = [_make_tool_call("read_market_data", {}, f"call_{i}") for i in range(5)]
        trading_interceptor.analyse_tool_calls(calls)
        assert trading_interceptor.call_count == 5

    def test_blocked_count_tracks_blocks(self, engine: AnalysisEngine) -> None:
        interceptor = AISeCOpenAIInterceptor(
            engine=engine,
            scenario=Scenario.TRADING_AI,
            agent_id="count_test",
            raise_on_block=False,
        )
        calls = [
            _make_tool_call("read_market_data", {}, "call_1"),
            _make_tool_call("manipulate_news_feed", {}, "call_2"),
        ]
        interceptor.analyse_tool_calls(calls)
        assert interceptor.blocked_count >= 1

    def test_block_rate_zero_when_no_calls(
        self, trading_interceptor: AISeCOpenAIInterceptor
    ) -> None:
        assert trading_interceptor.block_rate == 0.0


# ── Thread safety tests ───────────────────────────────────────────────────────


class TestThreadSafety:

    def test_concurrent_calls_no_corruption(
        self, trading_interceptor: AISeCOpenAIInterceptor
    ) -> None:
        errors: list[Exception] = []

        def analyse_call():
            try:
                tc = _make_tool_call("read_market_data", {"symbol": "AAPL"})
                trading_interceptor.analyse_single_call(tc)
            except AISeCOpenAISecurityError:
                pass
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=analyse_call) for _ in range(20)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert errors == [], f"Thread safety failure: {[str(e) for e in errors[:3]]}"

    def test_call_count_accurate_under_concurrency(
        self, trading_interceptor: AISeCOpenAIInterceptor
    ) -> None:
        n = 10

        def analyse():
            try:
                tc = _make_tool_call("read_market_data", {})
                trading_interceptor.analyse_single_call(tc)
            except AISeCOpenAISecurityError:
                pass

        threads = [threading.Thread(target=analyse) for _ in range(n)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert trading_interceptor.call_count == n


# ── Audit log tests ───────────────────────────────────────────────────────────


class TestAuditIntegration:

    def test_every_call_logged(
        self, engine: AnalysisEngine, trading_interceptor: AISeCOpenAIInterceptor
    ) -> None:
        calls = [
            _make_tool_call("read_market_data", {"symbol": "AAPL"}, "c1"),
            _make_tool_call("manipulate_news_feed", {"content": "x"}, "c2"),
            _make_tool_call("read_market_data", {"symbol": "MSFT"}, "c3"),
        ]
        try:
            trading_interceptor.analyse_tool_calls(calls)
        except AISeCOpenAISecurityError:
            pass

        assert engine.audit_count() == 3

    def test_audit_chain_intact_after_mixed_calls(self, engine: AnalysisEngine) -> None:
        interceptor = AISeCOpenAIInterceptor(
            engine=engine,
            scenario=Scenario.TRADING_AI,
            agent_id="audit_test",
            raise_on_block=False,
        )
        calls = [_make_tool_call("read_market_data", {}, f"c{i}") for i in range(5)] + [
            _make_tool_call("manipulate_news_feed", {}, "c_danger"),
        ]
        interceptor.analyse_tool_calls(calls)

        ok, errors = engine.verify_audit_chain()
        assert ok is True, f"Chain broken: {errors}"
