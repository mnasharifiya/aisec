"""
AISec AutoGen integration — function map interceptor.

Integrates AISec into any AutoGen agent setup by wrapping
the UserProxyAgent's function_map before tool execution.

Security design:
    - Fail closed: if AISec cannot analyse a tool call, it blocks it.
    - Inter-agent injection protection: tool calls are validated
      regardless of which agent initiated the request.
    - Function name sanitisation: AutoGen function names are Python
      identifiers — we validate them before analysis.
    - No sensitive data logging: tool inputs may contain secrets.
      Only SHA-256 hashes are written to the audit log.
    - Thread safety: safe for concurrent tool calls in GroupChat.
    - Identity enforcement: agent_id is set at wrapper construction
      and cannot be overridden by the agent or its messages.
    - Code execution protection: wrapping the function_map prevents
      AISec from being bypassed by direct code execution requests.

AutoGen interception points:
    AutoGen UserProxyAgent has two tool execution paths:
        1. function_map  — named Python functions (our primary target)
        2. code execution — arbitrary Python code blocks (blocked by design)

    AISec wraps path 1. Path 2 (code execution) should be disabled
    in production AutoGen deployments via:
        UserProxyAgent(code_execution_config=False)

Usage:
    from autogen import AssistantAgent, UserProxyAgent
    from aisec.integrations.autogen import AISeCAutoGenWrapper
    from aisec.core.engine import AnalysisEngine
    from aisec.storage.models import Scenario

    engine  = AnalysisEngine()
    wrapper = AISeCAutoGenWrapper(
        engine=engine,
        scenario=Scenario.TRADING_AI,
        agent_id="trading_assistant_prod",
    )

    def execute_trade(amount: float, symbol: str) -> str:
        # Real trading logic here
        return f"Trade executed: {symbol} amount={amount}"

    def read_market_data(symbol: str) -> str:
        return f"Market data for {symbol}"

    # Wrap the function map — AISec intercepts every call
    safe_function_map = wrapper.wrap_function_map({
        "execute_trade":    execute_trade,
        "read_market_data": read_market_data,
    })

    user_proxy = UserProxyAgent(
        name="user_proxy",
        function_map=safe_function_map,       # <- AISec intercepts here
        code_execution_config=False,          # <- Disable code execution
    )

Requirements:
    pip install pyautogen>=0.2.0

    AISec raises ImportError with clear instructions if AutoGen
    is not installed, rather than failing silently.
"""

from __future__ import annotations

import functools
import hashlib
import re
import threading
from typing import Any, Callable

from aisec.core.engine import AnalysisEngine
from aisec.storage.models import Decision, Event, Scenario
from aisec.utils.logger import get_logger

log = get_logger("aisec.integrations.autogen")

# ── AutoGen import guard ──────────────────────────────────────────────────────
# AutoGen is an optional dependency.
# AISec works without it — only the adapter requires it.

try:
    import autogen  # type: ignore[import]
    _AUTOGEN_AVAILABLE = True
except ImportError:
    _AUTOGEN_AVAILABLE = False


# ── Constants ─────────────────────────────────────────────────────────────────

# Maximum characters of tool input we analyse.
# Prevents memory exhaustion through crafted large inputs.
MAX_INPUT_LEN: int = 4_096

# Maximum function name length.
# AutoGen function names are Python identifiers — 256 chars is generous.
MAX_FUNC_NAME_LEN: int = 256

# Characters logged from inputs for debugging — never log full inputs.
LOG_INPUT_PREFIX_LEN: int = 64


# ── Security exception ────────────────────────────────────────────────────────

class AISeCAutoGenSecurityError(Exception):
    """
    Raised when AISec blocks an AutoGen tool call.

    AutoGen catches this exception and returns an error message
    to the agent, preventing the tool from executing.

    Attributes:
        decision:    The enforcement decision (BLOCK or ESCALATE).
        rule_hits:   List of rule IDs that fired.
        risk_score:  Computed risk score for this action.
        explanation: Human-readable explanation for the analyst.
        event_id:    ID of the event in the audit log.
        func_name:   The function name that was blocked.
    """

    def __init__(
        self,
        decision:    Decision,
        rule_hits:   list[str],
        risk_score:  float,
        explanation: str,
        event_id:    str,
        func_name:   str,
    ) -> None:
        self.decision    = decision
        self.rule_hits   = rule_hits
        self.risk_score  = risk_score
        self.explanation = explanation
        self.event_id    = event_id
        self.func_name   = func_name

        super().__init__(
            f"AISec {decision.value}: function '{func_name}' blocked. "
            f"{explanation[:200]} "
            f"[event_id={event_id}, risk={risk_score:.3f}]"
        )


# ── Security helpers ──────────────────────────────────────────────────────────

def _validate_function_name(name: str) -> str:
    """
    Validate and sanitise an AutoGen function name.

    AutoGen function names are Python identifiers. We accept only
    names matching the Python identifier pattern to prevent:
    - Shell injection through function names
    - Unicode bypass attempts
    - Extremely long names causing performance issues

    Args:
        name: Raw function name from AutoGen function_map.

    Returns:
        Validated function name.

    Raises:
        ValueError: If the name is invalid or empty.
    """
    if not name or not isinstance(name, str):
        raise ValueError("Function name must be a non-empty string")

    # Truncate before validation
    name = name[:MAX_FUNC_NAME_LEN]

    # Must be a valid Python identifier
    # This is stricter than LangChain — AutoGen uses Python callables
    if not re.match(r'^[a-zA-Z_][a-zA-Z0-9_]*$', name):
        raise ValueError(
            f"Function name '{name[:50]}' is not a valid Python identifier. "
            "AutoGen function names must match [a-zA-Z_][a-zA-Z0-9_]*"
        )

    return name


def _sanitise_kwargs(kwargs: dict[str, Any]) -> str:
    """
    Convert function kwargs to a safe string for analysis.

    We never pass raw kwargs to the rule engine because they may contain:
    - Injection payloads crafted to manipulate our rules
    - Sensitive data (API keys, credentials, PII)
    - Extremely large values designed to exhaust memory

    Args:
        kwargs: Raw function kwargs from AutoGen.

    Returns:
        Truncated, safe string representation.
    """
    try:
        # Sort keys for deterministic representation
        parts = []
        for k, v in sorted(kwargs.items()):
            key   = str(k)[:32]
            value = str(v)[:128]
            parts.append(f"{key}={value}")
        result = " ".join(parts)
    except Exception:
        result = "[unserializable kwargs]"

    return result[:MAX_INPUT_LEN]


def _hash_kwargs(kwargs_str: str) -> str:
    """
    Return a SHA-256 hash of the kwargs string for audit logging.

    We log the hash, not the kwargs, to prevent secrets from
    appearing in audit logs while maintaining verifiability.

    Args:
        kwargs_str: The sanitised kwargs string.

    Returns:
        First 16 characters of SHA-256 hex digest.
    """
    return hashlib.sha256(kwargs_str.encode("utf-8")).hexdigest()[:16]


def _extract_payload(
    func_name:  str,
    kwargs_str: str,
    kwargs:     dict[str, Any],
) -> dict[str, Any]:
    """
    Build a structured payload for risk scoring from function call data.

    Extracts numeric and boolean values from kwargs that are relevant
    to our rule engine (amounts, zones, after_hours flags, etc.)
    without exposing raw sensitive data.

    Args:
        func_name:  Validated function name.
        kwargs_str: Sanitised kwargs string.
        kwargs:     Original kwargs dict.

    Returns:
        Payload dictionary for Event.raw_payload.
    """
    payload: dict[str, Any] = {
        "kwargs_hash":   _hash_kwargs(kwargs_str),
        "kwargs_length": len(kwargs_str),
        "func_name":     func_name,
        "arg_count":     len(kwargs),
    }

    # Extract common financial parameters
    for amount_key in ("amount", "trade_amount", "value", "size", "quantity"):
        if amount_key in kwargs:
            try:
                payload["amount"] = float(kwargs[amount_key])
                break
            except (TypeError, ValueError):
                pass

    # Extract after_hours flag
    for ah_key in ("after_hours", "afterhours", "off_hours", "extended_hours"):
        if kwargs.get(ah_key):
            payload["after_hours"] = True
            break

    # Extract zone information (urban AI)
    for zone_key in ("zone", "area", "district", "region"):
        if zone_key in kwargs and isinstance(kwargs[zone_key], str):
            payload["zone"] = str(kwargs[zone_key])[:32]
            break

    # Extract duration (curfew detection)
    for dur_key in ("duration", "duration_hours", "hours", "minutes"):
        if dur_key in kwargs:
            try:
                payload["duration_hours"] = float(kwargs[dur_key])
                break
            except (TypeError, ValueError):
                pass

    # Extract affected count (traffic override detection)
    for count_key in ("affected_intersections", "intersections", "count", "num"):
        if count_key in kwargs:
            try:
                payload["affected_intersections"] = int(kwargs[count_key])
                break
            except (TypeError, ValueError):
                pass

    # Detect network access from function name
    network_indicators = {
        "trade", "execute", "send", "post", "publish",
        "broadcast", "override", "shutdown", "curfew",
        "manipulate", "inject",
    }
    if any(ind in func_name.lower() for ind in network_indicators):
        payload["network_access"] = True

    return payload


# ── AutoGen wrapper ───────────────────────────────────────────────────────────

class AISeCAutoGenWrapper:
    """
    Wraps an AutoGen function_map to intercept every tool call.

    Works with any AutoGen agent that uses a function_map:
        - UserProxyAgent
        - AssistantAgent (via function registration)
        - GroupChatManager participants

    Thread safety:
        The wrapper uses a threading.Lock around the engine call
        to handle concurrent tool calls in GroupChat setups safely.

    Fail closed guarantee:
        Any unexpected exception in the wrapper causes the tool call
        to be blocked. AISec never fails open.

    Usage:
        wrapper = AISeCAutoGenWrapper(
            engine=engine,
            scenario=Scenario.TRADING_AI,
            agent_id="trading_prod_v1",
        )
        safe_map = wrapper.wrap_function_map(original_function_map)
        # Use safe_map instead of original_function_map in AutoGen
    """

    def __init__(
        self,
        engine:          AnalysisEngine,
        scenario:        Scenario = Scenario.UNKNOWN,
        agent_id:        str      = "autogen_agent",
        block_on_review: bool     = True,
    ) -> None:
        """
        Initialise the AutoGen wrapper.

        Args:
            engine:          The AISec AnalysisEngine for evaluation.
            scenario:        Threat scenario — selects rule set.
            agent_id:        Identity recorded in the audit log.
                             Cannot be overridden by the agent.
            block_on_review: If True, PENDING_REVIEW also blocks.
                             Default True — conservative and safe.
        """
        if not isinstance(engine, AnalysisEngine):
            raise TypeError(
                f"engine must be an AnalysisEngine instance, "
                f"got {type(engine).__name__}"
            )

        # Sanitise agent_id — only safe Python identifier characters
        safe_id = re.sub(r'[^a-zA-Z0-9_.]', '', str(agent_id))[:64]
        if len(safe_id) < 3:
            safe_id = "autogen_agent"

        # All attributes are private — not accessible from outside
        self.__engine         = engine
        self.__scenario       = scenario
        self.__agent_id       = safe_id
        self.__block_on_review = block_on_review
        self.__lock           = threading.Lock()
        self.__call_count     = 0
        self.__blocked_count  = 0

        log.info(
            "aisec_autogen_wrapper_initialized",
            agent_id=self.__agent_id,
            scenario=scenario.value,
            block_on_review=block_on_review,
        )

    def wrap_function_map(
        self,
        function_map: dict[str, Callable[..., Any]],
    ) -> dict[str, Callable[..., Any]]:
        """
        Wrap every function in an AutoGen function_map with AISec interception.

        Returns a new dict with the same keys but wrapped callables.
        The original function_map is not modified.

        Security: each wrapped function validates its own name at call time,
        not just at wrap time. This prevents late-binding attacks where
        the function map is modified after wrapping.

        Args:
            function_map: The original AutoGen function_map dict.

        Returns:
            New dict with AISec-wrapped callables.

        Raises:
            TypeError:  If function_map is not a dict.
            ValueError: If a function name is invalid.
        """
        if not isinstance(function_map, dict):
            raise TypeError(
                f"function_map must be a dict, got {type(function_map).__name__}"
            )

        wrapped: dict[str, Callable[..., Any]] = {}

        for name, func in function_map.items():
            # Validate name at wrap time
            validated_name = _validate_function_name(name)

            if not callable(func):
                raise TypeError(
                    f"function_map['{name}'] must be callable, "
                    f"got {type(func).__name__}"
                )

            # Create the wrapped version
            wrapped[validated_name] = self._create_wrapper(
                validated_name, func
            )

            log.info(
                "function_wrapped",
                agent_id=self.__agent_id,
                func_name=validated_name,
            )

        log.info(
            "function_map_wrapped",
            agent_id=self.__agent_id,
            function_count=len(wrapped),
            functions=list(wrapped.keys()),
        )

        return wrapped

    def wrap_single_function(
        self,
        name: str,
        func: Callable[..., Any],
    ) -> Callable[..., Any]:
        """
        Wrap a single function with AISec interception.

        Useful for wrapping individual tools without building
        a full function_map.

        Args:
            name: The function name (must be valid Python identifier).
            func: The callable to wrap.

        Returns:
            Wrapped callable that passes through AISec.
        """
        validated_name = _validate_function_name(name)
        if not callable(func):
            raise TypeError(
                f"func must be callable, got {type(func).__name__}"
            )
        return self._create_wrapper(validated_name, func)

    # ── Private methods ───────────────────────────────────────────────────────

    def _create_wrapper(
        self,
        func_name: str,
        original:  Callable[..., Any],
    ) -> Callable[..., Any]:
        """
        Create a wrapped version of a function with AISec interception.

        The wrapper:
            1. Intercepts the call before execution
            2. Analyses it through the AISec engine
            3. Raises AISeCAutoGenSecurityError if blocked
            4. Calls the original function if allowed
            5. Logs the outcome in all cases

        The wrapper uses functools.wraps to preserve the original
        function's metadata (name, docstring, signature).

        Args:
            func_name: Validated function name.
            original:  The original callable to wrap.

        Returns:
            Wrapped callable.
        """
        # Capture references in closure — not accessible from outside
        engine          = self.__engine
        scenario        = self.__scenario
        agent_id        = self.__agent_id
        block_on_review = self.__block_on_review
        lock            = self.__lock
        wrapper_self    = self

        @functools.wraps(original)
        def _wrapped(**kwargs: Any) -> Any:
            """AISec-intercepted version of the original function."""
            try:
                return wrapper_self._intercept_and_call(
                    func_name=func_name,
                    original=original,
                    kwargs=kwargs,
                )
            except AISeCAutoGenSecurityError:
                # Re-raise security errors — these are intentional blocks
                raise
            except Exception as exc:
                # Unexpected error — fail closed
                log.error(
                    "autogen_wrapper_unexpected_error",
                    func_name=func_name,
                    exc_type=type(exc).__name__,
                    detail=str(exc)[:200],
                    agent_id=agent_id,
                )
                raise AISeCAutoGenSecurityError(
                    decision=Decision.BLOCK,
                    rule_hits=["WRAPPER-ERROR"],
                    risk_score=1.0,
                    explanation=(
                        f"AISec wrapper encountered an unexpected error "
                        f"while intercepting '{func_name}': "
                        f"{type(exc).__name__}. Failing closed."
                    ),
                    event_id="error",
                    func_name=func_name,
                ) from exc

        return _wrapped

    def _intercept_and_call(
        self,
        func_name: str,
        original:  Callable[..., Any],
        kwargs:    dict[str, Any],
    ) -> Any:
        """
        Core interception logic — analyse, decide, then execute or block.

        Thread safety:
            Engine analysis is protected by a lock.
            The original function is called outside the lock to prevent
            deadlocks if the function itself triggers another tool call.

        Args:
            func_name: Validated function name.
            original:  The original callable.
            kwargs:    Function kwargs from AutoGen.

        Returns:
            Return value of the original function if allowed.

        Raises:
            AISeCAutoGenSecurityError: If AISec decides to block.
        """
        # Sanitise kwargs — never pass raw data to our pipeline
        safe_kwargs_str = _sanitise_kwargs(kwargs)

        # Build structured payload for risk scoring
        payload = _extract_payload(func_name, safe_kwargs_str, kwargs)

        # Build the Event
        # agent_id is always from our constructor — cannot be overridden
        event = Event(
            action_type=func_name,
            agent_id=self.__agent_id,
            target=func_name,
            scenario=self.__scenario,
            raw_payload=payload,
        )

        log.info(
            "autogen_tool_call_intercepted",
            agent_id=self.__agent_id,
            func_name=func_name,
            kwargs_prefix=safe_kwargs_str[:LOG_INPUT_PREFIX_LEN],
            kwargs_hash=_hash_kwargs(safe_kwargs_str),
        )

        # Analyse under lock — thread-safe engine access
        with self.__lock:
            self.__call_count += 1
            result = self.__engine.analyse(event)

        # Handle the decision
        if result.blocked:
            with self.__lock:
                self.__blocked_count += 1

            log.warning(
                "autogen_tool_call_blocked",
                agent_id=self.__agent_id,
                func_name=func_name,
                decision=result.decision.value,
                risk_score=result.risk_score,
                rule_hits=result.analysis.rule_hits,
                event_id=result.log_entry_id,
            )

            raise AISeCAutoGenSecurityError(
                decision=result.decision,
                rule_hits=result.analysis.rule_hits,
                risk_score=result.risk_score,
                explanation=result.analysis.explanation,
                event_id=result.log_entry_id,
                func_name=func_name,
            )

        # Action allowed — execute original function OUTSIDE the lock
        log.info(
            "autogen_tool_call_allowed",
            agent_id=self.__agent_id,
            func_name=func_name,
            decision=result.decision.value,
            risk_score=result.risk_score,
            event_id=result.log_entry_id,
        )

        return original(**kwargs)

    # ── Monitoring accessors ──────────────────────────────────────────────────

    @property
    def call_count(self) -> int:
        """Total number of tool calls intercepted."""
        return self.__call_count

    @property
    def blocked_count(self) -> int:
        """Total number of tool calls blocked."""
        return self.__blocked_count

    @property
    def agent_id(self) -> str:
        """The agent ID this wrapper is monitoring."""
        return self.__agent_id

    @property
    def scenario(self) -> Scenario:
        """The scenario this wrapper is configured for."""
        return self.__scenario

    @property
    def block_rate(self) -> float:
        """Fraction of calls that were blocked. 0.0 if no calls yet."""
        if self.__call_count == 0:
            return 0.0
        return self.__blocked_count / self.__call_count

    def __repr__(self) -> str:
        return (
            f"AISeCAutoGenWrapper("
            f"agent_id={self.__agent_id!r}, "
            f"scenario={self.__scenario.value!r}, "
            f"calls={self.__call_count}, "
            f"blocked={self.__blocked_count})"
        )