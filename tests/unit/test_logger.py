"""
Unit tests for the structured JSON logger.
Run with: pytest tests/unit/test_logger.py -v
"""

from __future__ import annotations

import json
import logging
import io

import pytest

from aisec.utils.logger import (
    JSONFormatter,
    StructuredLogger,
    configure_logging,
    get_logger,
)


class TestJSONFormatter:

    def _make_record(self, msg: str, level: int = logging.INFO,
                     **kwargs) -> logging.LogRecord:
        record = logging.LogRecord(
            name="aisec.test", level=level,
            pathname="", lineno=0, msg=msg,
            args=(), exc_info=None,
        )
        for k, v in kwargs.items():
            setattr(record, f"_aisec_{k}", v)
        return record

    def test_output_is_valid_json(self) -> None:
        formatter = JSONFormatter()
        record    = self._make_record("test message")
        output    = formatter.format(record)
        parsed    = json.loads(output)
        assert parsed["msg"] == "test message"

    def test_required_fields_present(self) -> None:
        formatter = JSONFormatter()
        record    = self._make_record("hello")
        parsed    = json.loads(formatter.format(record))
        assert "ts"        in parsed
        assert "level"     in parsed
        assert "component" in parsed
        assert "msg"       in parsed

    def test_structured_fields_included(self) -> None:
        formatter = JSONFormatter()
        record    = self._make_record("decision", decision="BLOCK",
                                      risk_score=0.94)
        parsed    = json.loads(formatter.format(record))
        assert parsed["decision"]   == "BLOCK"
        assert parsed["risk_score"] == 0.94

    def test_level_name_is_correct(self) -> None:
        formatter = JSONFormatter()
        record    = self._make_record("warn msg", level=logging.WARNING)
        parsed    = json.loads(formatter.format(record))
        assert parsed["level"] == "WARNING"

    def test_output_is_single_line(self) -> None:
        formatter = JSONFormatter()
        record    = self._make_record("single line test")
        output    = formatter.format(record)
        assert "\n" not in output


class TestStructuredLogger:

    def _capture_output(self, logger_name: str) -> tuple[StructuredLogger, io.StringIO]:
        buf     = io.StringIO()
        handler = logging.StreamHandler(buf)
        handler.setFormatter(JSONFormatter())
        raw     = logging.getLogger(logger_name)
        raw.addHandler(handler)
        raw.setLevel(logging.DEBUG)
        raw.propagate = False
        return StructuredLogger(logger_name), buf

    def test_info_message_captured(self) -> None:
        log, buf = self._capture_output("test.info")
        log.info("action_evaluated", decision="ALLOW", risk=0.12)
        output = buf.getvalue()
        assert output.strip()
        parsed = json.loads(output.strip())
        assert parsed["msg"]      == "action_evaluated"
        assert parsed["decision"] == "ALLOW"
        assert parsed["risk"]     == 0.12

    def test_warning_level_correct(self) -> None:
        log, buf = self._capture_output("test.warning")
        log.warning("high_risk", risk=0.85)
        parsed = json.loads(buf.getvalue().strip())
        assert parsed["level"] == "WARNING"

    def test_error_level_correct(self) -> None:
        log, buf = self._capture_output("test.error")
        log.error("engine_failed", detail="something went wrong")
        parsed = json.loads(buf.getvalue().strip())
        assert parsed["level"]  == "ERROR"
        assert parsed["detail"] == "something went wrong"

    def test_multiple_fields_all_present(self) -> None:
        log, buf = self._capture_output("test.multi")
        log.info("pipeline_complete",
                 agent="trading_bot", action="execute_trade",
                 risk=0.45, decision="PENDING_REVIEW", rules=["TRADING-004"])
        parsed = json.loads(buf.getvalue().strip())
        assert parsed["agent"]    == "trading_bot"
        assert parsed["action"]   == "execute_trade"
        assert parsed["risk"]     == 0.45
        assert parsed["decision"] == "PENDING_REVIEW"
        assert parsed["rules"]    == ["TRADING-004"]

    def test_debug_message_below_info_not_emitted(self) -> None:
        buf     = io.StringIO()
        handler = logging.StreamHandler(buf)
        handler.setFormatter(JSONFormatter())
        raw     = logging.getLogger("test.debug_filter")
        raw.addHandler(handler)
        raw.setLevel(logging.INFO)   # INFO level — DEBUG should be filtered
        raw.propagate = False
        log = StructuredLogger("test.debug_filter")
        log.debug("this should not appear")
        assert buf.getvalue().strip() == ""