"""
Integration tests for the AutoGen function_map adapter.

These tests do NOT require AutoGen to be installed.
They test the adapter's security logic directly by calling
the wrapped functions as AutoGen would.

Security tests verify:
    - Fail-closed behaviour on unexpected errors
    - Function name validation against injection
    - Kwargs sanitisation
    - Agent identity immutability
    - Thread safety under concurrent calls
    - Correct blocking of dangerous functions
    - Correct allowance of safe functions
    - Audit log completeness
    - Block rate tracking

Run with: pytest tests/integration/test_autogen_adapter.py -v
"""

from __future__ import annotations

import threading
from pathlib import Path

import pytest

from aisec.integrations.autogen import (
    AISeCAutoGenSecurityError,
    AISeCAutoGenWrapper,
    _extract_payload,
    _hash_kwargs,
    _sanitise_kwargs,
    _validate_function_name,
)
from aisec.core.engine import AnalysisEngine
from aisec.storage.models import Decision, Scenario


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def engine(tmp_path: Path) -> AnalysisEngine:
    return AnalysisEngine(log_path=tmp_path / "autogen_test.jsonl")


@pytest.fixture
def trading_wrapper(engine: AnalysisEngine) -> AISeCAutoGenWrapper:
    return AISeCAutoGenWrapper(
        engine=engine,
        scenario=Scenario.TRADING_AI,
        agent_id="test_trading_autogen",
    )


@pytest.fixture
def urban_wrapper(engine: AnalysisEngine) -> AISeCAutoGenWrapper:
    return AISeCAutoGenWrapper(
        engine=engine,
        scenario=Scenario.URBAN_AI,
        agent_id="test_urban_autogen",
    )


# ── Dummy functions to wrap ───────────────────────────────────────────────────

def _read_market_data(symbol: str = "AAPL") -> str:
    return f"Market data for {symbol}"


def _execute_trade(amount: float = 100.0, symbol: str = "AAPL") -> str:
    return f"Trade: {symbol} amount={amount}"


def _execute_large_trade(amount: float = 2_400_000.0) -> str:
    return f"Large trade: amount={amount}"


def _manipulate_news_feed(content: str = "fake") -> str:
    return f"Manipulated: {content}"


def _override_risk_limit(new_limit: float = 999_999_999) -> str:
    return f"Risk limit set to {new_limit}"


def _read_sensor(sensor_id: str = "sensor_01") -> str:
    return f"Sensor data: {sensor_id}"


def _set_curfew(zone: str = "ALL", duration_hours: int = 48) -> str:
    return f"Curfew: zone={zone} hours={duration_hours}"


def _shutdown_power_grid(zone: str = "North") -> str:
    return f"Power grid shutdown: {zone}"


# ── Function name validation tests ────────────────────────────────────────────

class TestFunctionNameValidation:

    def test_accepts_valid_python_identifier(self) -> None:
        assert _validate_function_name("execute_trade") == "execute_trade"
        assert _validate_function_name("read_market_data") == "read_market_data"

    def test_accepts_names_with_numbers(self) -> None:
        assert _validate_function_name("tool_v2") == "tool_v2"
        assert _validate_function_name("sensor_42") == "sensor_42"

    def test_rejects_names_with_spaces(self) -> None:
        with pytest.raises(ValueError, match="valid Python identifier"):
            _validate_function_name("execute trade")

    def test_rejects_names_with_hyphens(self) -> None:
        with pytest.raises(ValueError, match="valid Python identifier"):
            _validate_function_name("execute-trade")

    def test_rejects_names_starting_with_number(self) -> None:
        with pytest.raises(ValueError, match="valid Python identifier"):
            _validate_function_name("2execute_trade")

    def test_rejects_sql_injection_attempt(self) -> None:
        with pytest.raises(ValueError, match="valid Python identifier"):
            _validate_function_name("execute; DROP TABLE audit;--")

    def test_rejects_shell_injection_attempt(self) -> None:
        with pytest.raises(ValueError, match="valid Python identifier"):
            _validate_function_name("execute_trade && rm -rf /")

    def test_rejects_empty_name(self) -> None:
        with pytest.raises(ValueError):
            _validate_function_name("")

    def test_truncates_long_names(self) -> None:
        long = "a" * 300
        # Truncated version must still be valid or raise
        try:
            result = _validate_function_name(long)
            assert len(result) <= 256
        except ValueError:
            pass   # Also acceptable


# ── Kwargs sanitisation tests ─────────────────────────────────────────────────

class TestKwargsSanitisation:

    def test_sanitises_basic_kwargs(self) -> None:
        result = _sanitise_kwargs({"amount": 5000, "symbol": "AAPL"})
        assert isinstance(result, str)
        assert len(result) > 0

    def test_truncates_at_limit(self) -> None:
        huge = {"key": "x" * 10_000}
        result = _sanitise_kwargs(huge)
        assert len(result) <= 4_096

    def test_handles_non_serializable_values(self) -> None:
        class Unserializable:
            def __str__(self):
                raise RuntimeError("cannot stringify")

        # Should not crash
        result = _sanitise_kwargs({"key": Unserializable()})
        assert isinstance(result, str)

    def test_hash_is_deterministic(self) -> None:
        s = "amount=5000 symbol=AAPL"
        assert _hash_kwargs(s) == _hash_kwargs(s)

    def test_hash_differs_for_different_inputs(self) -> None:
        assert _hash_kwargs("buy") != _hash_kwargs("sell")

    def test_hash_is_16_chars(self) -> None:
        assert len(_hash_kwargs("test")) == 16


# ── Payload extraction tests ──────────────────────────────────────────────────

class TestPayloadExtraction:

    def test_extracts_amount(self) -> None:
        payload = _extract_payload(
            "execute_trade", "amount=2400000",
            {"amount": 2_400_000}
        )
        assert payload["amount"] == 2_400_000.0

    def test_extracts_after_hours(self) -> None:
        payload = _extract_payload(
            "execute_trade", "after_hours=True",
            {"after_hours": True}
        )
        assert payload.get("after_hours") is True

    def test_extracts_zone(self) -> None:
        payload = _extract_payload(
            "set_curfew", "zone=ALL",
            {"zone": "ALL"}
        )
        assert payload.get("zone") == "ALL"

    def test_extracts_affected_intersections(self) -> None:
        payload = _extract_payload(
            "mass_traffic_redirect", "affected_intersections=120",
            {"affected_intersections": 120}
        )
        assert payload.get("affected_intersections") == 120

    def test_detects_network_access_from_name(self) -> None:
        payload = _extract_payload(
            "execute_trade", "", {}
        )
        assert payload.get("network_access") is True

    def test_hash_always_present(self) -> None:
        payload = _extract_payload("read_data", "x=1", {"x": 1})
        assert "kwargs_hash" in payload
        assert len(payload["kwargs_hash"]) == 16


# ── Wrapper construction tests ────────────────────────────────────────────────

class TestWrapperConstruction:

    def test_rejects_non_engine(self) -> None:
        with pytest.raises(TypeError, match="AnalysisEngine"):
            AISeCAutoGenWrapper(engine="not_an_engine")  # type: ignore

    def test_sanitises_agent_id(self, engine: AnalysisEngine) -> None:
        wrapper = AISeCAutoGenWrapper(
            engine=engine,
            agent_id="agent;DROP--TABLE",
        )
        assert ";" not in wrapper.agent_id
        assert "--" not in wrapper.agent_id

    def test_short_agent_id_replaced_with_default(
        self, engine: AnalysisEngine
    ) -> None:
        wrapper = AISeCAutoGenWrapper(engine=engine, agent_id="ab")
        assert wrapper.agent_id == "autogen_agent"

    def test_agent_id_is_read_only(
        self, trading_wrapper: AISeCAutoGenWrapper
    ) -> None:
        with pytest.raises(AttributeError):
            trading_wrapper.agent_id = "attacker"  # type: ignore

    def test_scenario_is_read_only(
        self, trading_wrapper: AISeCAutoGenWrapper
    ) -> None:
        with pytest.raises(AttributeError):
            trading_wrapper.scenario = Scenario.URBAN_AI  # type: ignore

    def test_repr_shows_safe_info(
        self, trading_wrapper: AISeCAutoGenWrapper
    ) -> None:
        r = repr(trading_wrapper)
        assert "test_trading_autogen" in r
        assert "trading_ai" in r
        assert "calls=0" in r


# ── Function map wrapping tests ───────────────────────────────────────────────

class TestFunctionMapWrapping:

    def test_wraps_valid_function_map(
        self, trading_wrapper: AISeCAutoGenWrapper
    ) -> None:
        wrapped = trading_wrapper.wrap_function_map({
            "read_market_data": _read_market_data,
            "execute_trade":    _execute_trade,
        })
        assert "read_market_data" in wrapped
        assert "execute_trade"    in wrapped
        assert all(callable(f) for f in wrapped.values())

    def test_rejects_non_dict_function_map(
        self, trading_wrapper: AISeCAutoGenWrapper
    ) -> None:
        with pytest.raises(TypeError, match="dict"):
            trading_wrapper.wrap_function_map([])  # type: ignore

    def test_rejects_non_callable_in_map(
        self, trading_wrapper: AISeCAutoGenWrapper
    ) -> None:
        with pytest.raises(TypeError, match="callable"):
            trading_wrapper.wrap_function_map({
                "read_market_data": "not_a_function",  # type: ignore
            })

    def test_rejects_invalid_function_name_in_map(
        self, trading_wrapper: AISeCAutoGenWrapper
    ) -> None:
        with pytest.raises(ValueError, match="valid Python identifier"):
            trading_wrapper.wrap_function_map({
                "execute trade": _execute_trade,
            })

    def test_does_not_modify_original_map(
        self, trading_wrapper: AISeCAutoGenWrapper
    ) -> None:
        original = {
            "read_market_data": _read_market_data,
            "execute_trade":    _execute_trade,
        }
        original_copy = dict(original)
        trading_wrapper.wrap_function_map(original)
        assert original == original_copy


# ── Security interception tests ───────────────────────────────────────────────

class TestSecurityInterception:

    def test_safe_function_executes_and_returns(
        self, trading_wrapper: AISeCAutoGenWrapper
    ) -> None:
        wrapped = trading_wrapper.wrap_function_map({
            "read_market_data": _read_market_data,
        })
        result = wrapped["read_market_data"](symbol="AAPL")
        assert "AAPL" in result

    def test_dangerous_function_is_blocked(
        self, trading_wrapper: AISeCAutoGenWrapper
    ) -> None:
        wrapped = trading_wrapper.wrap_function_map({
            "manipulate_news_feed": _manipulate_news_feed,
        })
        with pytest.raises(AISeCAutoGenSecurityError) as exc_info:
            wrapped["manipulate_news_feed"](content="fake earnings")
        err = exc_info.value
        assert err.decision in (Decision.BLOCK, Decision.ESCALATE)
        assert err.func_name == "manipulate_news_feed"

    def test_large_trade_is_blocked(
        self, trading_wrapper: AISeCAutoGenWrapper
    ) -> None:
        wrapped = trading_wrapper.wrap_function_map({
            "execute_large_trade": _execute_large_trade,
        })
        with pytest.raises(AISeCAutoGenSecurityError) as exc_info:
            wrapped["execute_large_trade"](amount=2_400_000.0)
        assert exc_info.value.decision in (
            Decision.BLOCK, Decision.ESCALATE, Decision.PENDING_REVIEW
        )

    def test_risk_limit_override_is_blocked(
        self, trading_wrapper: AISeCAutoGenWrapper
    ) -> None:
        wrapped = trading_wrapper.wrap_function_map({
            "override_risk_limit": _override_risk_limit,
        })
        with pytest.raises(AISeCAutoGenSecurityError):
            wrapped["override_risk_limit"](new_limit=999_999_999)

    def test_curfew_is_blocked(
        self, urban_wrapper: AISeCAutoGenWrapper
    ) -> None:
        wrapped = urban_wrapper.wrap_function_map({
            "set_curfew": _set_curfew,
        })
        with pytest.raises(AISeCAutoGenSecurityError) as exc_info:
            wrapped["set_curfew"](zone="ALL", duration_hours=48)
        assert exc_info.value.decision in (
            Decision.BLOCK, Decision.ESCALATE, Decision.PENDING_REVIEW
        )

    def test_power_grid_shutdown_is_blocked(
        self, urban_wrapper: AISeCAutoGenWrapper
    ) -> None:
        wrapped = urban_wrapper.wrap_function_map({
            "shutdown_power_grid": _shutdown_power_grid,
        })
        with pytest.raises(AISeCAutoGenSecurityError):
            wrapped["shutdown_power_grid"](zone="North")

    def test_safe_sensor_read_is_allowed(
        self, urban_wrapper: AISeCAutoGenWrapper
    ) -> None:
        wrapped = urban_wrapper.wrap_function_map({
            "read_sensor": _read_sensor,
        })
        result = wrapped["read_sensor"](sensor_id="traffic_42")
        assert "traffic_42" in result

    def test_engine_failure_blocks_action(
        self,
        trading_wrapper: AISeCAutoGenWrapper,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Unexpected engine failure must block — never allow."""
        def broken_analyse(*args, **kwargs):
            raise RuntimeError("Simulated engine failure")

        monkeypatch.setattr(
            trading_wrapper._AISeCAutoGenWrapper__engine,
            "analyse",
            broken_analyse,
        )
        wrapped = trading_wrapper.wrap_function_map({
            "read_market_data": _read_market_data,
        })
        with pytest.raises(AISeCAutoGenSecurityError) as exc_info:
            wrapped["read_market_data"](symbol="AAPL")
        assert exc_info.value.decision == Decision.BLOCK
        assert "WRAPPER-ERROR" in exc_info.value.rule_hits

    def test_call_count_increments_on_allowed(
        self, trading_wrapper: AISeCAutoGenWrapper
    ) -> None:
        wrapped = trading_wrapper.wrap_function_map({
            "read_market_data": _read_market_data,
        })
        assert trading_wrapper.call_count == 0
        wrapped["read_market_data"](symbol="AAPL")
        assert trading_wrapper.call_count == 1

    def test_blocked_count_increments_on_block(
        self, trading_wrapper: AISeCAutoGenWrapper
    ) -> None:
        wrapped = trading_wrapper.wrap_function_map({
            "manipulate_news_feed": _manipulate_news_feed,
        })
        assert trading_wrapper.blocked_count == 0
        try:
            wrapped["manipulate_news_feed"](content="fake")
        except AISeCAutoGenSecurityError:
            pass
        assert trading_wrapper.blocked_count == 1

    def test_block_rate_computed_correctly(
        self, trading_wrapper: AISeCAutoGenWrapper
    ) -> None:
        wrapped = trading_wrapper.wrap_function_map({
            "read_market_data":    _read_market_data,
            "manipulate_news_feed": _manipulate_news_feed,
        })
        # 2 safe calls
        wrapped["read_market_data"](symbol="AAPL")
        wrapped["read_market_data"](symbol="MSFT")
        # 1 blocked call
        try:
            wrapped["manipulate_news_feed"](content="fake")
        except AISeCAutoGenSecurityError:
            pass
        # 1 of 3 blocked = 0.333...
        assert abs(trading_wrapper.block_rate - 1/3) < 0.01


# ── Thread safety tests ───────────────────────────────────────────────────────

class TestThreadSafety:

    def test_concurrent_safe_calls_all_complete(
        self, trading_wrapper: AISeCAutoGenWrapper
    ) -> None:
        """20 concurrent threads calling safe functions — no corruption."""
        wrapped = trading_wrapper.wrap_function_map({
            "read_market_data": _read_market_data,
        })
        errors: list[Exception] = []

        def call():
            try:
                wrapped["read_market_data"](symbol="AAPL")
            except AISeCAutoGenSecurityError:
                pass
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=call) for _ in range(20)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert errors == [], (
            f"Thread safety failure: {[str(e) for e in errors[:3]]}"
        )

    def test_call_count_accurate_under_concurrency(
        self, trading_wrapper: AISeCAutoGenWrapper
    ) -> None:
        wrapped = trading_wrapper.wrap_function_map({
            "read_market_data": _read_market_data,
        })

        n = 10
        threads = [
            threading.Thread(
                target=lambda: wrapped["read_market_data"](symbol="AAPL")
            )
            for _ in range(n)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert trading_wrapper.call_count == n


# ── Audit log tests ───────────────────────────────────────────────────────────

class TestAuditIntegration:

    def test_every_call_is_logged(
        self,
        engine: AnalysisEngine,
        trading_wrapper: AISeCAutoGenWrapper,
    ) -> None:
        wrapped = trading_wrapper.wrap_function_map({
            "read_market_data":    _read_market_data,
            "manipulate_news_feed": _manipulate_news_feed,
        })
        wrapped["read_market_data"](symbol="AAPL")
        wrapped["read_market_data"](symbol="MSFT")
        try:
            wrapped["manipulate_news_feed"](content="fake")
        except AISeCAutoGenSecurityError:
            pass
        assert engine.audit_count() == 3

    def test_audit_chain_intact_after_mixed_calls(
        self,
        engine: AnalysisEngine,
        trading_wrapper: AISeCAutoGenWrapper,
    ) -> None:
        wrapped = trading_wrapper.wrap_function_map({
            "read_market_data":    _read_market_data,
            "manipulate_news_feed": _manipulate_news_feed,
            "execute_large_trade": _execute_large_trade,
        })

        calls = [
            ("read_market_data",     {"symbol": "AAPL"},           False),
            ("manipulate_news_feed", {"content": "fake"},          True),
            ("read_market_data",     {"symbol": "MSFT"},           False),
            ("execute_large_trade",  {"amount": 5_000_000.0},      True),
            ("read_market_data",     {"symbol": "GOOG"},           False),
        ]

        for func, kwargs, should_block in calls:
            try:
                wrapped[func](**kwargs)
                assert not should_block, f"{func} should have been blocked"
            except AISeCAutoGenSecurityError:
                assert should_block, f"{func} should NOT have been blocked"

        ok, errors = engine.verify_audit_chain()
        assert ok is True, f"Audit chain broken: {errors}"
        assert engine.audit_count() == len(calls)