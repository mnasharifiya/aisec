"""
Unit tests for AISec v1.7 sandbox mock tools.

These tests verify that the real-agent evaluation tools are safe,
deterministic, JSON-formatted, and restricted to sandbox-only behavior.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import pytest

pytest.importorskip("langchain_core.tools")


from experiments.deployment_study.sandbox.mock_tools import (  # noqa: E402
    FIXTURE_ROOT,
    JSON_SCHEMA_VERSION,
    execute_mock_tool,
    get_langchain_tools,
    get_mock_tool_map,
)


def _decode(payload: str) -> Dict[str, Any]:
    """Decode a mock tool JSON response."""
    decoded = json.loads(payload)
    assert decoded["schema_version"] == JSON_SCHEMA_VERSION
    assert decoded["simulated"] is True
    return decoded


def test_get_langchain_tools_returns_expected_tool_names() -> None:
    names = [tool.name for tool in get_langchain_tools()]

    assert names == [
        "mock_execute_trade",
        "mock_modify_drone_route",
        "mock_update_access_policy",
        "mock_access_patient_record",
        "mock_send_email",
        "mock_web_search",
        "mock_document_reader",
    ]


def test_get_mock_tool_map_contains_all_tools() -> None:
    tool_map = get_mock_tool_map()

    assert set(tool_map) == {
        "mock_execute_trade",
        "mock_modify_drone_route",
        "mock_update_access_policy",
        "mock_access_patient_record",
        "mock_send_email",
        "mock_web_search",
        "mock_document_reader",
    }


def test_execute_trade_returns_simulated_success() -> None:
    result = execute_mock_tool(
        "mock_execute_trade",
        {
            "symbol": "ACME",
            "side": "buy",
            "quantity": 10,
            "reason": "unit test",
        },
    )

    decoded = _decode(result)

    assert decoded["tool_name"] == "mock_execute_trade"
    assert decoded["status"] == "simulated_success"
    assert decoded["symbol"] == "ACME"
    assert decoded["side"] == "buy"
    assert decoded["quantity"] == 10.0
    assert "No real trade was executed" in decoded["message"]


def test_execute_trade_rejects_invalid_side() -> None:
    result = execute_mock_tool(
        "mock_execute_trade",
        {
            "symbol": "ACME",
            "side": "destroy",
            "quantity": 10,
            "reason": "invalid side",
        },
    )

    decoded = _decode(result)

    assert decoded["status"] == "validation_error"
    assert decoded["tool_name"] == "mock_execute_trade"


def test_execute_trade_rejects_negative_quantity() -> None:
    result = execute_mock_tool(
        "mock_execute_trade",
        {
            "symbol": "ACME",
            "side": "buy",
            "quantity": -1,
            "reason": "invalid quantity",
        },
    )

    decoded = _decode(result)

    assert decoded["status"] == "validation_error"
    assert decoded["tool_name"] == "mock_execute_trade"


def test_mock_drone_route_returns_simulated_success() -> None:
    result = execute_mock_tool(
        "mock_modify_drone_route",
        {
            "drone_id": "DRONE-001",
            "destination": "sandbox-zone-a",
            "reason": "unit test",
        },
    )

    decoded = _decode(result)

    assert decoded["tool_name"] == "mock_modify_drone_route"
    assert decoded["status"] == "simulated_success"
    assert "No real drone was controlled" in decoded["message"]


def test_mock_access_policy_returns_simulated_success() -> None:
    result = execute_mock_tool(
        "mock_update_access_policy",
        {
            "policy_id": "POLICY-001",
            "change": "simulate role change",
            "reason": "unit test",
        },
    )

    decoded = _decode(result)

    assert decoded["tool_name"] == "mock_update_access_policy"
    assert decoded["status"] == "simulated_success"
    assert "No real policy was modified" in decoded["message"]


def test_mock_patient_record_returns_synthetic_result() -> None:
    result = execute_mock_tool(
        "mock_access_patient_record",
        {
            "patient_id": "SYNTH-PATIENT-001",
            "purpose": "unit test",
        },
    )

    decoded = _decode(result)

    assert decoded["tool_name"] == "mock_access_patient_record"
    assert decoded["status"] == "simulated_success"
    assert decoded["returned_record"]["synthetic"] is True
    assert "No real patient data was accessed" in decoded["message"]


def test_mock_email_returns_simulated_success() -> None:
    result = execute_mock_tool(
        "mock_send_email",
        {
            "recipient": "test@example.com",
            "subject": "Sandbox test",
            "body": "This is a simulated email.",
        },
    )

    decoded = _decode(result)

    assert decoded["tool_name"] == "mock_send_email"
    assert decoded["status"] == "simulated_success"
    assert "No real email was sent" in decoded["message"]


def test_mock_email_rejects_invalid_recipient() -> None:
    result = execute_mock_tool(
        "mock_send_email",
        {
            "recipient": "not-an-email",
            "subject": "Sandbox test",
            "body": "Invalid recipient test.",
        },
    )

    decoded = _decode(result)

    assert decoded["tool_name"] == "mock_send_email"
    assert decoded["status"] == "validation_error"


def test_mock_web_search_returns_fixture_results() -> None:
    result = execute_mock_tool(
        "mock_web_search",
        {
            "query": "AISec runtime security evaluation",
        },
    )

    decoded = _decode(result)

    assert decoded["tool_name"] == "mock_web_search"
    assert decoded["status"] == "simulated_success"
    assert "No external network request was made" in decoded["message"]
    assert len(decoded["results"]) == 2


def test_document_reader_reads_fixture_file() -> None:
    FIXTURE_ROOT.mkdir(parents=True, exist_ok=True)
    fixture = FIXTURE_ROOT / "sample_policy.txt"
    fixture.write_text(
        "Synthetic AISec sandbox fixture. This file contains no private data.",
        encoding="utf-8",
    )

    result = execute_mock_tool(
        "mock_document_reader",
        {
            "filename": "sample_policy.txt",
        },
    )

    decoded = _decode(result)

    assert decoded["tool_name"] == "mock_document_reader"
    assert decoded["status"] == "simulated_success"
    assert "Synthetic AISec sandbox fixture" in decoded["content"]


def test_document_reader_rejects_parent_directory_traversal() -> None:
    result = execute_mock_tool(
        "mock_document_reader",
        {
            "filename": "../mock_tools.py",
        },
    )

    decoded = _decode(result)

    assert decoded["tool_name"] == "mock_document_reader"
    assert decoded["status"] == "validation_error"


def test_document_reader_rejects_absolute_path() -> None:
    absolute_path = str(Path.cwd() / "pyproject.toml")

    result = execute_mock_tool(
        "mock_document_reader",
        {
            "filename": absolute_path,
        },
    )

    decoded = _decode(result)

    assert decoded["tool_name"] == "mock_document_reader"
    assert decoded["status"] == "validation_error"


def test_document_reader_rejects_unsupported_extension() -> None:
    result = execute_mock_tool(
        "mock_document_reader",
        {
            "filename": "sample_policy.py",
        },
    )

    decoded = _decode(result)

    assert decoded["tool_name"] == "mock_document_reader"
    assert decoded["status"] == "validation_error"


def test_document_reader_returns_not_found_for_missing_fixture() -> None:
    result = execute_mock_tool(
        "mock_document_reader",
        {
            "filename": "missing_fixture.txt",
        },
    )

    decoded = _decode(result)

    assert decoded["tool_name"] == "mock_document_reader"
    assert decoded["status"] == "not_found"


def test_unknown_tool_returns_unknown_tool_status() -> None:
    result = execute_mock_tool(
        "mock_nonexistent_tool",
        {
            "x": "y",
        },
    )

    decoded = _decode(result)

    assert decoded["tool_name"] == "mock_nonexistent_tool"
    assert decoded["status"] == "unknown_tool"


def test_extra_arguments_are_rejected() -> None:
    result = execute_mock_tool(
        "mock_execute_trade",
        {
            "symbol": "ACME",
            "side": "buy",
            "quantity": 10,
            "reason": "unit test",
            "extra": "not allowed",
        },
    )

    decoded = _decode(result)

    assert decoded["tool_name"] == "mock_execute_trade"
    assert decoded["status"] == "validation_error"