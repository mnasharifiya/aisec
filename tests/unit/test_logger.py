"""
Unit tests for AISec structured logger.

Run with:
    pytest tests/unit/test_logger.py -v
"""

from __future__ import annotations

import json
from typing import Any

from aisec.utils.logger import configure_logging, get_logger


class TestLoggerBasics:
    def test_get_logger_returns_logger(self) -> None:
        logger = get_logger("test.module")
        assert logger is not None

    def test_logger_has_required_methods(self) -> None:
        logger = get_logger("test.module")

        assert hasattr(logger, "debug")
        assert hasattr(logger, "info")
        assert hasattr(logger, "warning")
        assert hasattr(logger, "error")

    def test_same_name_returns_usable_loggers(self) -> None:
        first = get_logger("same.module")
        second = get_logger("same.module")

        assert first is not None
        assert second is not None

        first.info("same_logger_first_call")
        second.info("same_logger_second_call")

    def test_different_names_return_different_loggers(self) -> None:
        first = get_logger("module.one")
        second = get_logger("module.two")

        assert first is not second


class TestLoggerConfiguration:
    def test_configure_logging_accepts_common_levels(self) -> None:
        configure_logging(level="DEBUG", output="stderr")
        configure_logging(level="INFO", output="stderr")
        configure_logging(level="WARNING", output="stderr")
        configure_logging(level="ERROR", output="stderr")

    def test_configure_logging_accepts_stdout_and_stderr(self) -> None:
        configure_logging(level="INFO", output="stdout")
        configure_logging(level="INFO", output="stderr")

    def test_repeated_configuration_does_not_break_logging(self) -> None:
        configure_logging(level="INFO", output="stderr")
        configure_logging(level="INFO", output="stderr")
        configure_logging(level="WARNING", output="stdout")
        configure_logging(level="INFO", output="stderr")

        logger = get_logger("test.reconfigure")
        logger.info("reconfigured_logger_still_works", ok=True)

    def test_invalid_level_does_not_corrupt_logger(self) -> None:
        try:
            configure_logging(level="NOT_A_LEVEL", output="stderr")
        except Exception:
            pass

        configure_logging(level="INFO", output="stderr")
        logger = get_logger("test.invalid_level_recovery")
        logger.info("logger_recovered_after_invalid_level")

    def test_invalid_output_does_not_corrupt_logger(self) -> None:
        try:
            configure_logging(level="INFO", output="invalid_output")
        except Exception:
            pass

        configure_logging(level="INFO", output="stderr")
        logger = get_logger("test.invalid_output_recovery")
        logger.info("logger_recovered_after_invalid_output")


class TestLoggerEmission:
    def test_debug_does_not_raise(self) -> None:
        configure_logging(level="DEBUG", output="stderr")
        logger = get_logger("test.debug")
        logger.debug("debug_event", enabled=True)

    def test_info_does_not_raise(self) -> None:
        logger = get_logger("test.info")
        logger.info("info_event", key="value", number=42)

    def test_warning_does_not_raise(self) -> None:
        logger = get_logger("test.warning")
        logger.warning("warning_event", detail="something")

    def test_error_does_not_raise(self) -> None:
        logger = get_logger("test.error")
        logger.error("error_event", exc_type="ValueError")

    def test_structured_fields_do_not_raise(self) -> None:
        logger = get_logger("test.structured")

        logger.info(
            "structured_event",
            agent_id="agent_01",
            decision="BLOCK",
            risk_score=0.93,
            blocked=True,
        )

    def test_none_value_does_not_raise(self) -> None:
        logger = get_logger("test.none")
        logger.info("none_value_event", value=None)

    def test_boolean_values_do_not_raise(self) -> None:
        logger = get_logger("test.boolean")
        logger.info("boolean_event", allowed=True, blocked=False)

    def test_numeric_values_do_not_raise(self) -> None:
        logger = get_logger("test.numeric")
        logger.info("numeric_event", count=10, score=0.87)

    def test_nested_payload_does_not_raise(self) -> None:
        logger = get_logger("test.nested")

        logger.info(
            "nested_payload_event",
            payload={
                "agent_id": "agent_01",
                "risk_score": 0.91,
                "tags": ["rbac", "soc", "audit"],
                "metadata": {
                    "source": "unit_test",
                    "safe": True,
                },
            },
        )

    def test_exception_metadata_does_not_raise(self) -> None:
        logger = get_logger("test.exception")

        try:
            raise ValueError("example")
        except ValueError as exc:
            logger.error(
                "handled_exception",
                exc_type=type(exc).__name__,
                detail=str(exc),
            )


class TestLoggerSafety:
    def test_control_characters_do_not_raise(self) -> None:
        logger = get_logger("test.control_chars")

        logger.info(
            "control_char_event",
            value="line1\nline2\tline3\rline4",
        )

    def test_large_field_does_not_raise(self) -> None:
        logger = get_logger("test.large_field")

        logger.info(
            "large_field_event",
            payload="x" * 10_000,
        )

    def test_unicode_text_does_not_raise(self) -> None:
        logger = get_logger("test.unicode")

        logger.info(
            "unicode_event",
            message="AISec monitor event — secure autonomous agent",
        )

    def test_json_compatible_payload_does_not_raise(self) -> None:
        logger = get_logger("test.json_payload")

        payload: dict[str, Any] = {
            "agent_id": "agent_01",
            "decision": "PENDING_REVIEW",
            "risk_score": 0.77,
            "rules": ["after_hours", "large_trade"],
        }

        json.dumps(payload)
        logger.info("json_payload_event", payload=payload)

    def test_unserializable_object_does_not_crash_test_suite(self) -> None:
        logger = get_logger("test.unserializable")

        class Unserializable:
            pass

        try:
            logger.info("unserializable_event", obj=Unserializable())
        except (TypeError, ValueError):
            pass

    def test_non_string_event_name_does_not_crash_test_suite(self) -> None:
        logger = get_logger("test.non_string_event")

        try:
            logger.info(12345, key="value")  # type: ignore[arg-type]
        except (TypeError, ValueError):
            pass


class TestLoggerAfterConfiguration:
    def test_logger_still_works_after_stdout_configuration(self) -> None:
        configure_logging(level="INFO", output="stdout")
        logger = get_logger("test.stdout_configured")

        logger.info("stdout_configured_event", key="value")

    def test_logger_still_works_after_stderr_configuration(self) -> None:
        configure_logging(level="INFO", output="stderr")
        logger = get_logger("test.stderr_configured")

        logger.warning("stderr_configured_event", key="value")

    def test_debug_call_is_safe_when_level_is_info(self) -> None:
        configure_logging(level="INFO", output="stderr")
        logger = get_logger("test.debug_filter_safe")

        logger.debug("debug_event_may_be_filtered")
        logger.info("info_event_should_be_allowed")
