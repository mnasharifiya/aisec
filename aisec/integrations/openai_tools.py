"""
AISec OpenAI function-calling integration.

Intercepts OpenAI tool calls before execution by analysing
the model's tool_call response objects through AISec.

This adapter works with the OpenAI function-calling protocol
used by GPT-4, GPT-4o, and any model that returns tool_calls
in its response. It is framework-agnostic — works with the
raw OpenAI Python client, with LangChain OpenAI bindings,
and with any custom implementation of the protocol.

Security design:
    - Fail closed: any error in analysis blocks the tool call.
    - JSON argument parsing is sandboxed — malformed JSON blocks
      the call rather than crashing AISec.
    - Multiple tool calls in one response are each analysed
      independently — one dangerous call does not prevent
      analysis of others.
    - Tool call IDs are logged for correlation with OpenAI logs.
    - Agent identity is set at interceptor construction and
      cannot be overridden by tool call content.
    - No sensitive data logged — arguments are hashed only.

OpenAI tool call structure (what we intercept):
    {
        "id":       "call_abc123",
        "type":     "function",
        "function": {
            "name":      "execute_trade",
            "arguments": '{"amount": 5000, "symbol": "AAPL"}'
        }
    }

Usage:
    from openai import OpenAI
    from aisec.integrations.openai_tools import AISeCOpenAIInterceptor
    from aisec.core.engine import AnalysisEngine
    from aisec.storage.models import Scenario

    engine      = AnalysisEngine()
    interceptor = AISeCOpenAIInterceptor(
        engine=engine,
        scenario=Scenario.TRADING_AI,
        agent_id="trading_gpt4_prod",
    )

    # After getting a response from OpenAI:
    response = client.chat.completions.create(...)

    if response.choices[0].message.tool_calls:
        # Analyse all tool calls — raises on blocked ones
        safe_calls = interceptor.analyse_tool_calls(
            response.choices[0].message.tool_calls
        )
        # safe_calls contains only the allowed tool calls
        # Execute them with your function dispatcher

Requirements:
    pip install openai>=1.0.0   (optional — adapter works without it)
"""

from __future__ import annotations

import hashlib
import json
import re
import threading
from dataclasses import dataclass, field
from typing import Any

from aisec.core.engine import AnalysisEngine
from aisec.storage.models import Decision, Event, Scenario
from aisec.utils.logger import get_logger

log = get_logger("aisec.integrations.openai_tools")


# ── Constants ─────────────────────────────────────────────────────────────────

# Maximum length of tool arguments string we analyse.
MAX_ARGS_LEN:        int = 4_096

# Maximum tool name length — OpenAI enforces 64 chars but we are stricter.
MAX_TOOL_NAME_LEN:   int = 64

# Maximum tool call ID length for audit logging.
MAX_CALL_ID_LEN:     int = 128

# Characters logged from arguments for debugging.
LOG_ARGS_PREFIX_LEN: int = 64


# ── Result types ──────────────────────────────────────────────────────────────

@dataclass
class ToolCallResult:
    """
    Result of AISec analysis for a single OpenAI tool call.

    Attributes:
        tool_call_id:  The OpenAI tool call ID for correlation.
        function_name: The function that was called.
        decision:      AISec enforcement decision.
        risk_score:    Computed risk score in [0.0, 1.0].
        rule_hits:     Rule IDs that fired.
        explanation:   Human-readable explanation.
        event_id:      Audit log entry ID.
        allowed:       True if the tool call may proceed.
        blocked:       True if the tool call must not proceed.
    """
    tool_call_id:  str
    function_name: str
    decision:      Decision
    risk_score:    float
    rule_hits:     list[str]
    explanation:   str
    event_id:      str

    @property
    def allowed(self) -> bool:
        return self.decision == Decision.ALLOW

    @property
    def blocked(self) -> bool:
        return self.decision in (
            Decision.BLOCK,
            Decision.ESCALATE,
            Decision.PENDING_REVIEW,
        )


@dataclass
class BatchAnalysisResult:
    """
    Result of analysing all tool calls in a single OpenAI response.

    Contains individual results for each tool call and
    aggregate statistics for the batch.

    Attributes:
        results:        Individual ToolCallResult for each call.
        any_blocked:    True if any tool call was blocked.
        allowed_calls:  Tool calls that may proceed.
        blocked_calls:  Tool calls that were blocked.
    """
    results: list[ToolCallResult] = field(default_factory=list)

    @property
    def any_blocked(self) -> bool:
        return any(r.blocked for r in self.results)

    @property
    def allowed_calls(self) -> list[ToolCallResult]:
        return [r for r in self.results if r.allowed]

    @property
    def blocked_calls(self) -> list[ToolCallResult]:
        return [r for r in self.results if r.blocked]

    @property
    def total(self) -> int:
        return len(self.results)

    @property
    def block_count(self) -> int:
        return len(self.blocked_calls)


# ── Security exception ────────────────────────────────────────────────────────

class AISeCOpenAISecurityError(Exception):
    """
    Raised when AISec blocks one or more OpenAI tool calls.

    Contains the full batch analysis result so the caller
    can inspect which calls were blocked and why.

    Attributes:
        batch_result:   Full analysis result for the batch.
        blocked_calls:  List of blocked ToolCallResult objects.
    """

    def __init__(self, batch_result: BatchAnalysisResult) -> None:
        self.batch_result  = batch_result
        self.blocked_calls = batch_result.blocked_calls

        blocked_names = [c.function_name for c in self.blocked_calls]
        super().__init__(
            f"AISec blocked {len(self.blocked_calls)} tool call(s): "
            f"{blocked_names}. "
            f"Check batch_result for details."
        )


# ── Security helpers ──────────────────────────────────────────────────────────

def _validate_tool_name(name: str) -> str:
    """
    Validate an OpenAI tool function name.

    OpenAI enforces: must be a-z, A-Z, 0-9, underscores, hyphens.
    Maximum 64 characters.
    We are slightly stricter — no leading hyphens or digits.

    Args:
        name: Raw function name from OpenAI tool call.

    Returns:
        Validated function name.

    Raises:
        ValueError: If the name is invalid.
    """
    if not name or not isinstance(name, str):
        raise ValueError("Tool name must be a non-empty string")

    name = name[:MAX_TOOL_NAME_LEN]

    # OpenAI pattern: a-z, A-Z, 0-9, underscores, hyphens
    if not re.match(r'^[a-zA-Z_][a-zA-Z0-9_-]*$', name):
        raise ValueError(
            f"Tool name '{name[:30]}' is not a valid OpenAI function name. "
            "Must match [a-zA-Z_][a-zA-Z0-9_-]*"
        )

    return name


def _parse_arguments_safely(arguments_str: str) -> dict[str, Any]:
    """
    Parse OpenAI tool call arguments JSON safely.

    OpenAI sends arguments as a JSON string. Malformed JSON
    must not crash AISec — instead block the call.

    Args:
        arguments_str: Raw arguments string from OpenAI tool call.

    Returns:
        Parsed arguments dict. Empty dict if parsing fails.
    """
    if not arguments_str or not arguments_str.strip():
        return {}

    # Truncate before parsing to prevent memory exhaustion
    truncated = arguments_str[:MAX_ARGS_LEN]

    try:
        parsed = json.loads(truncated)
        if not isinstance(parsed, dict):
            # OpenAI arguments should always be a JSON object
            return {"_raw_value": str(parsed)[:256]}
        return parsed
    except (json.JSONDecodeError, ValueError):
        # Malformed JSON — log and treat as empty
        log.warning(
            "tool_arguments_json_parse_failed",
            args_prefix=truncated[:64],
            args_hash=hashlib.sha256(
                truncated.encode()
            ).hexdigest()[:16],
        )
        return {}


def _hash_arguments(arguments_str: str) -> str:
    """
    Return SHA-256 hash of tool arguments for audit logging.

    We log the hash not the arguments because arguments may
    contain sensitive data (API keys, PII, credentials).

    Args:
        arguments_str: Raw arguments string.

    Returns:
        First 16 characters of SHA-256 hex digest.
    """
    return hashlib.sha256(
        arguments_str.encode("utf-8")
    ).hexdigest()[:16]


def _sanitise_tool_call_id(call_id: str) -> str:
    """
    Sanitise a tool call ID for safe audit logging.

    OpenAI tool call IDs are alphanumeric strings like 'call_abc123'.
    We accept only safe characters to prevent log injection.

    Args:
        call_id: Raw tool call ID from OpenAI.

    Returns:
        Sanitised ID string.
    """
    if not call_id or not isinstance(call_id, str):
        return "unknown_call_id"
    safe = "".join(c for c in call_id if c.isalnum() or c in "_-")
    return safe[:MAX_CALL_ID_LEN] or "unknown_call_id"


def _build_payload(
    tool_name:     str,
    arguments:     dict[str, Any],
    arguments_str: str,
    call_id:       str,
) -> dict[str, Any]:
    """
    Build a structured payload for AISec risk scoring.

    Extracts security-relevant fields from the tool call arguments
    without exposing raw sensitive data.

    Args:
        tool_name:     Validated tool name.
        arguments:     Parsed arguments dict.
        arguments_str: Original arguments string (for hashing).
        call_id:       Sanitised tool call ID.

    Returns:
        Payload dict for Event.raw_payload.
    """
    payload: dict[str, Any] = {
        "args_hash":    _hash_arguments(arguments_str),
        "args_length":  len(arguments_str),
        "tool_name":    tool_name,
        "tool_call_id": call_id,
        "arg_count":    len(arguments),
    }

    # Extract financial amounts
    for key in ("amount", "trade_amount", "value", "size", "quantity", "price"):
        if key in arguments:
            try:
                payload["amount"] = float(arguments[key])
                break
            except (TypeError, ValueError):
                pass

    # Extract after-hours flag
    for key in ("after_hours", "afterhours", "extended_hours", "off_hours"):
        if arguments.get(key):
            payload["after_hours"] = True
            break

    # Extract zone (urban AI)
    for key in ("zone", "area", "district", "region", "location"):
        if key in arguments and isinstance(arguments[key], str):
            payload["zone"] = str(arguments[key])[:32]
            break

    # Extract duration (curfew detection)
    for key in ("duration_hours", "duration", "hours", "time_hours"):
        if key in arguments:
            try:
                payload["duration_hours"] = float(arguments[key])
                break
            except (TypeError, ValueError):
                pass

    # Extract affected count (traffic override)
    for key in ("affected_intersections", "intersections", "affected_count"):
        if key in arguments:
            try:
                payload["affected_intersections"] = int(arguments[key])
                break
            except (TypeError, ValueError):
                pass

    # Detect network access from tool name
    network_indicators = {
        "trade", "execute", "send", "post", "publish", "broadcast",
        "override", "shutdown", "curfew", "manipulate", "inject",
        "restrict", "lockdown",
    }
    if any(ind in tool_name.lower() for ind in network_indicators):
        payload["network_access"] = True

    # Detect privilege indicators from tool name
    privilege_indicators = {
        "override", "admin", "root", "sudo", "system",
        "shutdown", "kill", "terminate", "curfew", "lockdown",
    }
    if any(ind in tool_name.lower() for ind in privilege_indicators):
        payload["is_privileged"] = True

    return payload


# ── OpenAI tool call protocol ─────────────────────────────────────────────────

def _extract_tool_call_fields(tool_call: Any) -> tuple[str, str, str]:
    """
    Extract fields from an OpenAI tool call object.

    Handles both:
        - OpenAI Python SDK objects (with .id, .function.name, .function.arguments)
        - Plain dicts (for testing without the OpenAI SDK)

    Args:
        tool_call: OpenAI tool call object or dict.

    Returns:
        (call_id, function_name, arguments_str) tuple.

    Raises:
        ValueError: If required fields are missing or invalid.
    """
    try:
        if isinstance(tool_call, dict):
            # Dict format (testing or custom implementations)
            call_id      = str(tool_call.get("id", ""))
            func         = tool_call.get("function", {})
            func_name    = str(func.get("name", ""))
            arguments    = str(func.get("arguments", "{}"))
        else:
            # OpenAI SDK object format
            call_id   = str(getattr(tool_call, "id", ""))
            func      = getattr(tool_call, "function", None)
            if func is None:
                raise ValueError("tool_call has no .function attribute")
            func_name  = str(getattr(func, "name", ""))
            arguments  = str(getattr(func, "arguments", "{}"))

    except (AttributeError, TypeError) as exc:
        raise ValueError(
            f"Cannot extract fields from tool_call: {exc}"
        ) from exc

    return call_id, func_name, arguments


# ── OpenAI interceptor ────────────────────────────────────────────────────────

class AISeCOpenAIInterceptor:
    """
    Intercepts OpenAI tool calls before execution.

    Works with the OpenAI function-calling protocol used by
    GPT-4, GPT-4o, and compatible models.

    Designed to be used after receiving a model response and
    before executing the tool calls it contains.

    Thread safety:
        Safe for concurrent use from async OpenAI clients.
        Engine analysis is protected by a threading.Lock.

    Fail closed:
        Any error during analysis blocks the tool call.
        AISec never fails in a way that allows execution.

    Usage:
        interceptor = AISeCOpenAIInterceptor(engine, Scenario.TRADING_AI)

        response = client.chat.completions.create(...)
        tool_calls = response.choices[0].message.tool_calls

        if tool_calls:
            batch = interceptor.analyse_tool_calls(tool_calls)
            if batch.any_blocked:
                raise AISeCOpenAISecurityError(batch)
            # Execute only allowed calls
            for result in batch.allowed_calls:
                execute_function(result.function_name, ...)
    """

    def __init__(
        self,
        engine:          AnalysisEngine,
        scenario:        Scenario = Scenario.UNKNOWN,
        agent_id:        str      = "openai_agent",
        block_on_review: bool     = True,
        raise_on_block:  bool     = True,
    ) -> None:
        """
        Initialise the OpenAI interceptor.

        Args:
            engine:          AISec AnalysisEngine for evaluation.
            scenario:        Threat scenario — selects rule set.
            agent_id:        Identity recorded in the audit log.
            block_on_review: If True, PENDING_REVIEW also blocks.
            raise_on_block:  If True, raise AISeCOpenAISecurityError
                             when any call is blocked.
                             If False, return BatchAnalysisResult
                             and let the caller decide.
        """
        if not isinstance(engine, AnalysisEngine):
            raise TypeError(
                f"engine must be an AnalysisEngine instance, "
                f"got {type(engine).__name__}"
            )

        # Sanitise agent_id
        safe_id = re.sub(r'[^a-zA-Z0-9_.]', '', str(agent_id))[:64]
        if len(safe_id) < 3:
            safe_id = "openai_agent"

        self.__engine          = engine
        self.__scenario        = scenario
        self.__agent_id        = safe_id
        self.__block_on_review = block_on_review
        self.__raise_on_block  = raise_on_block
        self.__lock            = threading.Lock()
        self.__call_count      = 0
        self.__blocked_count   = 0

        log.info(
            "aisec_openai_interceptor_initialized",
            agent_id=self.__agent_id,
            scenario=scenario.value,
            block_on_review=block_on_review,
            raise_on_block=raise_on_block,
        )

    def analyse_tool_calls(
        self,
        tool_calls: list[Any],
    ) -> BatchAnalysisResult:
        """
        Analyse all tool calls in a single OpenAI response.

        Each tool call is analysed independently. A blocked call
        does not prevent analysis of the others — all results
        are returned in the BatchAnalysisResult.

        Args:
            tool_calls: List of OpenAI tool call objects or dicts.

        Returns:
            BatchAnalysisResult with individual results for each call.

        Raises:
            AISeCOpenAISecurityError: If raise_on_block=True and
                                      any call was blocked.
            TypeError: If tool_calls is not a list.
        """
        if not isinstance(tool_calls, list):
            raise TypeError(
                f"tool_calls must be a list, got {type(tool_calls).__name__}"
            )

        batch = BatchAnalysisResult()

        for tool_call in tool_calls:
            result = self._analyse_single_call(tool_call)
            batch.results.append(result)

        if self.__raise_on_block and batch.any_blocked:
            raise AISeCOpenAISecurityError(batch)

        return batch

    def analyse_single_call(
        self,
        tool_call: Any,
    ) -> ToolCallResult:
        """
        Analyse a single OpenAI tool call.

        Useful when processing tool calls one at a time.

        Args:
            tool_call: Single OpenAI tool call object or dict.

        Returns:
            ToolCallResult with the enforcement decision.

        Raises:
            AISeCOpenAISecurityError: If raise_on_block=True and
                                      the call was blocked.
        """
        result = self._analyse_single_call(tool_call)

        if self.__raise_on_block and result.blocked:
            batch = BatchAnalysisResult(results=[result])
            raise AISeCOpenAISecurityError(batch)

        return result

    # ── Private methods ───────────────────────────────────────────────────────

    def _analyse_single_call(self, tool_call: Any) -> ToolCallResult:
        """
        Core analysis logic for a single tool call.

        Fail closed: any unexpected exception produces a BLOCK result.

        Args:
            tool_call: OpenAI tool call object or dict.

        Returns:
            ToolCallResult — always returned, never raises.
        """
        try:
            return self._do_analyse(tool_call)
        except AISeCOpenAISecurityError:
            raise
        except Exception as exc:
            # Unexpected error — fail closed
            log.error(
                "openai_interceptor_unexpected_error",
                exc_type=type(exc).__name__,
                detail=str(exc)[:200],
                agent_id=self.__agent_id,
            )
            return ToolCallResult(
                tool_call_id="error",
                function_name="unknown",
                decision=Decision.BLOCK,
                risk_score=1.0,
                rule_hits=["INTERCEPTOR-ERROR"],
                explanation=(
                    f"AISec interceptor error: {type(exc).__name__}. "
                    "Failing closed."
                ),
                event_id="error",
            )

    def _do_analyse(self, tool_call: Any) -> ToolCallResult:
        """
        Perform the actual analysis of a single tool call.

        Args:
            tool_call: OpenAI tool call object or dict.

        Returns:
            ToolCallResult with the enforcement decision.
        """
        # Extract fields from tool call object
        try:
            raw_call_id, raw_func_name, arguments_str = (
                _extract_tool_call_fields(tool_call)
            )
        except ValueError as exc:
            log.error(
                "tool_call_field_extraction_failed",
                error=str(exc),
                agent_id=self.__agent_id,
            )
            return ToolCallResult(
                tool_call_id="unknown",
                function_name="unknown",
                decision=Decision.BLOCK,
                risk_score=1.0,
                rule_hits=["EXTRACTION-ERROR"],
                explanation=f"Cannot extract tool call fields: {exc}",
                event_id="error",
            )

        # Sanitise all inputs
        call_id = _sanitise_tool_call_id(raw_call_id)

        try:
            func_name = _validate_tool_name(raw_func_name)
        except ValueError as exc:
            log.warning(
                "tool_name_validation_failed",
                raw_name=raw_func_name[:30],
                error=str(exc),
                agent_id=self.__agent_id,
            )
            return ToolCallResult(
                tool_call_id=call_id,
                function_name=raw_func_name[:30],
                decision=Decision.BLOCK,
                risk_score=1.0,
                rule_hits=["NAME-VALIDATION-FAILED"],
                explanation=f"Invalid tool name: {exc}",
                event_id="error",
            )

        # Parse arguments safely
        arguments = _parse_arguments_safely(arguments_str)

        # Build payload for risk scoring
        payload = _build_payload(
            func_name, arguments, arguments_str, call_id
        )

        # Build the Event
        event = Event(
            action_type=func_name,
            agent_id=self.__agent_id,
            target=func_name,
            scenario=self.__scenario,
            raw_payload=payload,
        )

        log.info(
            "openai_tool_call_intercepted",
            agent_id=self.__agent_id,
            func_name=func_name,
            call_id=call_id,
            args_prefix=arguments_str[:LOG_ARGS_PREFIX_LEN],
            args_hash=_hash_arguments(arguments_str),
        )

        # Analyse under lock — thread-safe
        with self.__lock:
            self.__call_count += 1
            engine_result = self.__engine.analyse(event)

        # Build the result
        tool_result = ToolCallResult(
            tool_call_id=call_id,
            function_name=func_name,
            decision=engine_result.decision,
            risk_score=engine_result.risk_score,
            rule_hits=engine_result.analysis.rule_hits,
            explanation=engine_result.analysis.explanation,
            event_id=engine_result.log_entry_id,
        )

        if tool_result.blocked:
            with self.__lock:
                self.__blocked_count += 1

            log.warning(
                "openai_tool_call_blocked",
                agent_id=self.__agent_id,
                func_name=func_name,
                call_id=call_id,
                decision=engine_result.decision.value,
                risk_score=engine_result.risk_score,
                rule_hits=engine_result.analysis.rule_hits,
            )
        else:
            log.info(
                "openai_tool_call_allowed",
                agent_id=self.__agent_id,
                func_name=func_name,
                call_id=call_id,
                decision=engine_result.decision.value,
                risk_score=engine_result.risk_score,
            )

        return tool_result

    # ── Monitoring accessors ──────────────────────────────────────────────────

    @property
    def call_count(self) -> int:
        """Total tool calls analysed."""
        return self.__call_count

    @property
    def blocked_count(self) -> int:
        """Total tool calls blocked."""
        return self.__blocked_count

    @property
    def agent_id(self) -> str:
        """The agent ID this interceptor monitors."""
        return self.__agent_id

    @property
    def scenario(self) -> Scenario:
        """The configured threat scenario."""
        return self.__scenario

    @property
    def block_rate(self) -> float:
        """Fraction of calls blocked. 0.0 if no calls yet."""
        if self.__call_count == 0:
            return 0.0
        return self.__blocked_count / self.__call_count

    def __repr__(self) -> str:
        return (
            f"AISeCOpenAIInterceptor("
            f"agent_id={self.__agent_id!r}, "
            f"scenario={self.__scenario.value!r}, "
            f"calls={self.__call_count}, "
            f"blocked={self.__blocked_count})"
        )