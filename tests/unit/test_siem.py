"""
Unit tests for CEF/SIEM export.
Run with: pytest tests/unit/test_siem.py -v
"""

from __future__ import annotations

from pathlib import Path

import pytest

from aisec.integrations.siem import (
    CEFFormatter,
    SIEMExporter,
    _sanitise_cef_value,
    _sanitise_cef_header,
)
from aisec.storage.models import Decision


@pytest.fixture
def formatter() -> CEFFormatter:
    return CEFFormatter()


def _result_kwargs(**overrides) -> dict:
    base = dict(
        event_id="test-evt-001",
        agent_id="trading_bot_v1",
        action_type="execute_large_trade",
        decision=Decision.BLOCK,
        risk_score=0.94,
        rule_hits=["TRADING-001"],
        scenario="trading_ai",
        explanation="Large trade blocked",
        timestamp="2025-05-03T22:14:05+00:00",
    )
    base.update(overrides)
    return base


class TestCEFSanitisation:

    def test_sanitises_equals_sign(self) -> None:
        assert "\\=" in _sanitise_cef_value("key=value")

    def test_sanitises_backslash(self) -> None:
        assert "\\\\" in _sanitise_cef_value("path\\file")

    def test_sanitises_newline(self) -> None:
        assert "\\n" in _sanitise_cef_value("line1\nline2")

    def test_sanitises_pipe_in_header(self) -> None:
        assert "\\|" in _sanitise_cef_header("vendor|product")

    def test_truncates_long_values(self) -> None:
        assert len(_sanitise_cef_value("x" * 1000)) <= 500

    def test_safe_string_unchanged(self) -> None:
        assert _sanitise_cef_value("safe_string_123") == "safe_string_123"


class TestCEFFormatter:

    def test_format_result_returns_string(self, formatter: CEFFormatter) -> None:
        line = formatter.format_result(**_result_kwargs())
        assert isinstance(line, str)
        assert len(line) > 0

    def test_format_result_starts_with_cef(self, formatter: CEFFormatter) -> None:
        line = formatter.format_result(**_result_kwargs())
        assert line.startswith("CEF:0")

    def test_format_result_contains_vendor(self, formatter: CEFFormatter) -> None:
        line = formatter.format_result(**_result_kwargs())
        assert "AISec" in line

    def test_format_result_contains_decision(self, formatter: CEFFormatter) -> None:
        line = formatter.format_result(**_result_kwargs(decision=Decision.BLOCK))
        assert "BLOCK" in line

    def test_format_result_contains_agent_id(self, formatter: CEFFormatter) -> None:
        line = formatter.format_result(**_result_kwargs())
        assert "trading_bot_v1" in line

    def test_format_result_severity_10_for_block(self, formatter: CEFFormatter) -> None:
        line = formatter.format_result(**_result_kwargs(decision=Decision.BLOCK))
        # CEF header has 7 pipe-separated fields, severity is field 7
        parts = line.split("|")
        assert parts[6] == "10"

    def test_format_result_severity_1_for_allow(self, formatter: CEFFormatter) -> None:
        line = formatter.format_result(**_result_kwargs(decision=Decision.ALLOW))
        parts = line.split("|")
        assert parts[6] == "1"

    def test_format_result_contains_risk_score(self, formatter: CEFFormatter) -> None:
        line = formatter.format_result(**_result_kwargs(risk_score=0.94))
        assert "cn1=94" in line  # 0.94 * 100 = 94

    def test_format_result_contains_rule_hits(self, formatter: CEFFormatter) -> None:
        line = formatter.format_result(
            **_result_kwargs(rule_hits=["TRADING-001", "TRADING-002"])
        )
        assert "TRADING-001" in line
        assert "TRADING-002" in line

    def test_format_result_no_newlines(self, formatter: CEFFormatter) -> None:
        line = formatter.format_result(**_result_kwargs())
        assert "\n" not in line
        assert "\r" not in line

    def test_format_temporal_alert(self, formatter: CEFFormatter) -> None:
        line = formatter.format_temporal_alert(
            agent_id="bot",
            threat="BURST_ATTACK",
            severity="CRITICAL",
            description="50 events in 10s",
            timestamp="2025-05-03T22:14:05+00:00",
        )
        assert "CEF:0" in line
        assert "BURST_ATTACK" in line
        assert "CRITICAL" in line

    def test_format_result_injection_in_agent_id(self, formatter: CEFFormatter) -> None:
        line = formatter.format_result(
            **_result_kwargs(agent_id="bot|evil=injection\nattack")
        )
        # Pipe and newline must be escaped or removed
        assert "\n" not in line

    def test_format_result_injection_in_explanation(
        self, formatter: CEFFormatter
    ) -> None:
        line = formatter.format_result(
            **_result_kwargs(explanation="safe\n|evil=injection")
        )
        assert "\n" not in line


class TestSIEMExporter:

    def test_write_line_creates_file(self, tmp_path: Path) -> None:
        exporter = SIEMExporter(output_path=tmp_path / "siem.log")
        exporter.write_line("CEF:0|AISec|test|1.0|001|test|5|msg=test")
        assert (tmp_path / "siem.log").exists()

    def test_write_line_appends(self, tmp_path: Path) -> None:
        exporter = SIEMExporter(output_path=tmp_path / "siem.log")
        exporter.write_line("CEF:0|AISec|test|1.0|001|line1|5|msg=one")
        exporter.write_line("CEF:0|AISec|test|1.0|001|line2|5|msg=two")
        lines = (tmp_path / "siem.log").read_text().strip().splitlines()
        assert len(lines) == 2

    def test_export_audit_log_writes_cef_lines(self, tmp_path: Path) -> None:
        from aisec.core.engine import AnalysisEngine
        from aisec.storage.models import Event, Scenario

        engine = AnalysisEngine(log_path=tmp_path / "audit.jsonl")
        exporter = SIEMExporter(output_path=tmp_path / "siem.log")

        for action in ["read_market_data", "manipulate_news_feed"]:
            engine.analyse(
                Event(
                    action_type=action,
                    agent_id="test_bot",
                    target="MARKET",
                    scenario=Scenario.TRADING_AI,
                )
            )

        written = exporter.export_audit_log(engine._logger)
        assert written >= 2

        content = (tmp_path / "siem.log").read_text()
        assert "CEF:0" in content
        assert "AISec" in content

    def test_export_returns_count(self, tmp_path: Path) -> None:
        from aisec.core.engine import AnalysisEngine
        from aisec.storage.models import Event, Scenario

        engine = AnalysisEngine(log_path=tmp_path / "audit2.jsonl")
        exporter = SIEMExporter(output_path=tmp_path / "siem2.log")

        engine.analyse(
            Event(
                action_type="read_market_data",
                agent_id="bot",
                target="NYSE",
                scenario=Scenario.TRADING_AI,
            )
        )

        count = exporter.export_audit_log(engine._logger)
        assert count >= 1
