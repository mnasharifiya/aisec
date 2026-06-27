"""
Unit tests for AISec v1.7 LangChain/Groq real-agent collector.

These tests verify the tool-call collection layer without making any
external Groq API calls. The collector must be importable, deterministic,
JSON-safe, and capable of extracting proposed tool calls without executing
sandbox tools.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import pytest
from langchain_core.messages import AIMessage

from experiments.deployment_study.agents.langchain_agent import (
    DEFAULT_FRAMEWORK,
    DEFAULT_MODEL,
    DEFAULT_PROTOCOL_VERSION,
    DEFAULT_PROVIDER,
    ProposedToolCall,
    RealAgentProposalResult,
    ToolCallCollectorConfig,
    build_default_system_prompt,
    build_system_fingerprint,
    canonical_json,
    extract_provider_metadata_summary,
    extract_response_content_summary,
    extract_tool_calls,
    make_json_safe,
    normalize_tool_call,
    safe_text,
    sanitize_prompt,
    stable_prompt_hash,
    stable_sha256,
    tool_schema_fingerprint,
)
from experiments.deployment_study.sandbox.mock_tools import get_langchain_tools


def test_stable_sha256_is_deterministic() -> None:
    first = stable_sha256("aisec")
    second = stable_sha256("aisec")

    assert first == second
    assert len(first) == 64


def test_stable_prompt_hash_is_sha256() -> None:
    prompt_hash = stable_prompt_hash("test prompt")

    assert len(prompt_hash) == 64
    assert prompt_hash == stable_prompt_hash("test prompt")


def test_safe_text_collapses_whitespace_and_truncates() -> None:
    value = "  hello     world  "
    assert safe_text(value) == "hello world"

    long_value = "a" * 2000
    shortened = safe_text(long_value, max_length=10)

    assert shortened == "aaaaaaaaaa...[truncated]"


def test_sanitize_prompt_returns_bounded_clean_text() -> None:
    assert sanitize_prompt("  hello   world  ") == "hello world"


def test_make_json_safe_handles_nested_non_json_values() -> None:
    payload = {
        "path": Path("example.txt"),
        "error": ValueError("bad value"),
        "items": {1, 2},
    }

    safe = make_json_safe(payload)

    assert safe["path"] == "example.txt"
    assert safe["error"]["error_type"] == "ValueError"
    assert "bad value" in safe["error"]["message"]
    assert sorted(safe["items"]) == [1, 2]


def test_canonical_json_is_deterministic() -> None:
    left = {"b": 2, "a": 1}
    right = {"a": 1, "b": 2}

    assert canonical_json(left) == canonical_json(right)


def test_tool_schema_fingerprint_is_stable() -> None:
    tools = get_langchain_tools()

    first = tool_schema_fingerprint(tools)
    second = tool_schema_fingerprint(tools)

    assert first == second
    assert len(first) == 64


def test_tool_collector_config_defaults() -> None:
    config = ToolCallCollectorConfig()

    assert config.model_id == DEFAULT_MODEL
    assert config.model_provider == DEFAULT_PROVIDER
    assert config.protocol_version == DEFAULT_PROTOCOL_VERSION
    assert config.framework == DEFAULT_FRAMEWORK


def test_tool_collector_config_from_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GROQ_MODEL", "llama-3.3-70b-versatile")
    monkeypatch.setenv("GROQ_TEMPERATURE", "0.0")
    monkeypatch.setenv("GROQ_MAX_RETRIES", "3")
    monkeypatch.setenv("GROQ_TIMEOUT_SECONDS", "45")
    monkeypatch.setenv("AISEC_PROTOCOL_VERSION", "1.7.0-test")

    config = ToolCallCollectorConfig.from_environment()

    assert config.model_id == "llama-3.3-70b-versatile"
    assert config.temperature == 0.0
    assert config.max_retries == 3
    assert config.timeout_seconds == 45
    assert config.protocol_version == "1.7.0-test"


def test_normalize_mapping_tool_call() -> None:
    call = normalize_tool_call(
        {
            "name": "mock_execute_trade",
            "args": {"symbol": "ACME", "side": "buy", "quantity": 10},
            "id": "call-123",
        },
        index=2,
    )

    assert isinstance(call, ProposedToolCall)
    assert call.name == "mock_execute_trade"
    assert call.args["symbol"] == "ACME"
    assert call.args["quantity"] == 10
    assert call.call_id == "call-123"
    assert call.index == 2


def test_normalize_tool_call_handles_non_mapping_args() -> None:
    call = normalize_tool_call(
        {
            "name": "mock_execute_trade",
            "args": "not-json-args",
            "id": "call-raw",
        }
    )

    assert call.name == "mock_execute_trade"
    assert call.args["_raw_args"] == "not-json-args"


def test_extract_tool_calls_from_ai_message() -> None:
    message = AIMessage(
        content="",
        tool_calls=[
            {
                "name": "mock_send_email",
                "args": {
                    "recipient": "test@example.com",
                    "subject": "Synthetic test",
                    "body": "Hello",
                },
                "id": "call-email-1",
            }
        ],
    )

    calls = extract_tool_calls(message)

    assert len(calls) == 1
    assert calls[0].name == "mock_send_email"
    assert calls[0].args["recipient"] == "test@example.com"
    assert calls[0].call_id == "call-email-1"


def test_extract_response_content_summary() -> None:
    message = AIMessage(content="  synthetic   response  ")

    assert extract_response_content_summary(message) == "synthetic response"


def test_extract_provider_metadata_summary_excludes_logprobs() -> None:
    message = AIMessage(
        content="",
        response_metadata={
            "model_name": "llama-3.3-70b-versatile",
            "finish_reason": "tool_calls",
            "token_usage": {"total_tokens": 10},
            "system_fingerprint": "fingerprint",
            "logprobs": {"should": "not be exported"},
            "private_extra": "do not export",
        },
    )

    summary = extract_provider_metadata_summary(message)

    assert summary["model_name"] == "llama-3.3-70b-versatile"
    assert summary["finish_reason"] == "tool_calls"
    assert summary["token_usage"]["total_tokens"] == 10
    assert summary["system_fingerprint"] == "fingerprint"
    assert "logprobs" not in summary
    assert "private_extra" not in summary


def test_build_default_system_prompt_mentions_sandbox() -> None:
    prompt = build_default_system_prompt()

    assert "sandbox" in prompt.lower()
    assert "no real-world" in prompt.lower()


def test_build_system_fingerprint_has_expected_keys() -> None:
    fingerprint = build_system_fingerprint()

    assert "python_version" in fingerprint
    assert "platform" in fingerprint


def test_real_agent_proposal_result_public_record_and_json() -> None:
    result = RealAgentProposalResult(
        study_run_id="study-1",
        task_id="task-1",
        task_group="A",
        repetition_id=0,
        prompt_hash=stable_prompt_hash("prompt"),
        sanitized_prompt="prompt",
        model_provider="groq",
        model_id="llama-3.3-70b-versatile",
        temperature=0.0,
        protocol_version="1.7.0",
        framework="langchain",
        timestamp_utc="2026-06-25T00:00:00+00:00",
        tool_schema_hash="a" * 64,
        raw_response_type="AIMessage",
        response_content_summary="",
        proposed_tool_calls=[
            ProposedToolCall(
                name="mock_execute_trade",
                args={"symbol": "ACME", "side": "buy", "quantity": 10},
                call_id="call-1",
                index=0,
            )
        ],
        provider_metadata_summary={"finish_reason": "tool_calls"},
        system_fingerprint={"python_version": "3.x"},
    )

    public_record = result.to_public_record()
    encoded = result.to_json()
    decoded: Dict[str, Any] = json.loads(encoded)

    assert result.proposed_tool_call_count == 1
    assert public_record["study_run_id"] == "study-1"
    assert decoded["proposed_tool_calls"][0]["name"] == "mock_execute_trade"


def test_collector_missing_api_key_raises_clean_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import experiments.deployment_study.agents.langchain_agent as langchain_agent

    # Q1/CI-standard isolation:
    # This test must verify behavior when GROQ_API_KEY is truly unavailable.
    # A developer's local .env file must not affect this test.
    monkeypatch.setattr(langchain_agent, "load_optional_env", lambda: None)
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.delenv("GROQ_MODEL", raising=False)

    with pytest.raises(
        langchain_agent.RealAgentConfigurationError,
        match="GROQ_API_KEY",
    ):
        langchain_agent.LangChainGroqToolCallCollector(
            config=langchain_agent.ToolCallCollectorConfig()
        )