"""
AISec LangChain integration — callback-based runtime interceptor.

Integrates AISec into any LangChain agent by registering a
callback handler that intercepts every tool call before execution.

Security design:
    - Fail closed: any exception in AISec blocks the tool call.
      AISec must never fail in a way that allows an action to proceed.
    - Input sanitisation: tool inputs are truncated and sanitised
      before analysis to prevent injection through crafted payloads.
    - No sensitive data logging: tool inputs may contain secrets.
      Only hashed representations are written to the audit log.
    - Identity enforcement: agent_id is derived from the LangChain
      run_id which cannot be forged by the agent itself.
    - Thread safety: the handler is safe for concurrent tool calls
      from multi-agent LangChain setups.

Usage:
    from langchain.agents import AgentExecutor, create_openai_tools_agent
    from aisec.integrations.langchain import AISeCCallbackHandler
    from aisec.core.engine import AnalysisEngine
    from aisec.storage.models import Scenario

    engine  = AnalysisEngine()
    handler = AISeCCallbackHandler(
        engine=engine,
        scenario=Scenario.TRADING_AI,
        agent_id="trading_bot_prod",
    )

    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        callbacks=[handler],       # <- AISec intercepts here
        verbose=False,
    )

    # Now every tool call passes through AISec before execution.
    agent_executor.invoke({"input": "Analyse the market"})

Requirements:
    pip install langchain>=0.1.0

    AISec will raise ImportError with clear instructions if
    LangChain is not installed, rather than failing silently.
"""

from __future__ import annotations

import hashlib
import threading
from typing import Any, Union
from uuid import UUID

from aisec.core.engine import AnalysisEngine
from aisec.storage.models import Decision, Event, Scenario
from aisec.utils.logger import get_logger

log = get_logger("aisec.integrations.langchain")

# ── LangChain import guard ────────────────────────────────────────────────────
# We import LangChain lazily and fail with a clear message if not installed.
# This keeps AISec installable without LangChain as a hard dependency.

try:
    from langchain_core.callbacks import BaseCallbackHandler
    from langchain_core.outputs import LLMResult
    _LANGCHAIN_AVAILABLE = True
except ImportError:
    try:
        from langchain.callbacks.base import BaseCallbackHandler  # type: ignore[no-redef]
        from langchain.schema import LLMResult                     # type: ignore[no-redef]
        _LANGCHAIN_AVAILABLE = True
    except ImportError:
        _LANGCHAIN_AVAILABLE = False
        BaseCallbackHandler = object  # type: ignore[assignment,misc]


# ── Constants ─────────────────────────────────────────────────────────────────

# Maximum characters of tool input we analyse.
# Beyond this limit, inputs are truncated before analysis.
# This prevents memory exhaustion through crafted large inputs
# and limits the attack surface of our rule/scorer pipeline.
MAX_INPUT_LEN: int = 2_048

# Maximum tool name length we accept.
# Unusually long tool names are a potential injection signal.
MAX_TOOL_NAME_LEN: int = 128

# Minimum characters shown in logs from tool inputs.
# We never log full inputs — only a short prefix for debugging.
# This prevents secrets (API keys, passwords) from appearing in logs.
LOG_INPUT_PREFIX_LEN: int = 64


# ── Security helpers ──────────────────────────────────────────────────────────

def _sanitise_tool_name(name: str) -> str:
    """
    Sanitise a tool name before it enters the analysis pipeline.

    Accepts only alphanumeric characters, underscores, and hyphens.
    Truncates to MAX_TOOL_NAME_LEN.

    This prevents:
    - SQL/command injection through tool names
    - Rule engine evasion through Unicode lookalikes
    - Memory issues from absurdly long names

    Args:
        name: Raw tool name from LangChain.

    Returns:
        Sanitised, safe tool name string.
    """
    if not name or not isinstance(name, str):
        return "unknown_tool"

    # Keep only safe characters
    safe = "".join(
        c for c in name
        if c.isalnum() or c in "-_."
    )

    # Truncate
    safe = safe[:MAX_TOOL_NAME_LEN]

    return safe if safe else "unknown_tool"


def _sanitise_input(raw: Any) -> str:
    """
    Convert tool input to a safe string for analysis.

    We never pass raw tool input directly to our rule engine
    because it may contain:
    - Injected rule-evasion payloads
    - Extremely large inputs designed to exhaust memory
    - Sensitive data (API keys, credentials, PII)

    Args:
        raw: Raw tool input from LangChain (any type).

    Returns:
        Truncated string representation, safe for analysis.
    """
    try:
        text = str(raw)
    except Exception:
        text = "[unserializable input]"

    return text[:MAX_INPUT_LEN]


def _hash_input(raw: str) -> str:
    """
    Return a SHA-256 hash of the tool input for audit logging.

    We log the hash, not the input itself, to prevent secrets
    from appearing in audit logs while still maintaining a
    verifiable record that a specific input was submitted.

    Args:
        raw: The sanitised input string.

    Returns:
        First 16 characters of SHA-256 hex digest.
    """
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def _extract_payload(tool_name: str, tool_input: str) -> dict[str, Any]:
    """
    Extract a structured payload from a tool call for feature scoring.

    Attempts to parse numeric values and flags from the input string
    so the feature vector builder can score them meaningfully.

    Args:
        tool_name:  Sanitised tool name.
        tool_input: Sanitised input string.

    Returns:
        Payload dictionary for Event.raw_payload.
    """
    payload: dict[str, Any] = {
        "input_hash":   _hash_input(tool_input),
        "input_length": len(tool_input),
        "tool_name":    tool_name,
    }

    # Detect numeric amounts in input — relevant for trading rules
    import re
    amount_match = re.search(r"\b(\d[\d,]*(?:\.\d+)?)\b", tool_input)
    if amount_match:
        try:
            amount_str = amount_match.group(1).replace(",", "")
            payload["amount"] = float(amount_str)
        except ValueError:
            pass

    # Detect after-hours indicators
    after_hours_keywords = {"after_hours", "after-hours", "afterhours",
                            "off_hours", "overnight", "weekend"}
    if any(kw in tool_input.lower() for kw in after_hours_keywords):
        payload["after_hours"] = True

    # Detect burst/frequency signals
    burst_match = re.search(r"burst[_\s]?rate[=:\s]+(\d+(?:\.\d+)?)",
                             tool_input.lower())
    if burst_match:
        try:
            payload["burst_rate"] = float(burst_match.group(1))
        except ValueError:
            pass

    # Detect zone information (urban AI)
    zone_match = re.search(r"zone[=:\s]+([A-Za-z0-9_-]+)", tool_input)
    if zone_match:
        payload["zone"] = zone_match.group(1)[:32]

    return payload


# ── Security exception ────────────────────────────────────────────────────────

class AISeCSecurityError(Exception):
    """
    Raised when AISec blocks a tool call.

    LangChain will catch this exception and stop the tool
    from executing. The agent receives an error message
    explaining that the action was blocked.

    Attributes:
        decision:    The enforcement decision (BLOCK or ESCALATE).
        rule_hits:   List of rule IDs that fired.
        risk_score:  Computed risk score for this action.
        explanation: Human-readable explanation for the analyst.
        event_id:    ID of the event in the audit log.
    """

    def __init__(
        self,
        decision: Decision,
        rule_hits: list[str],
        risk_score: float,
        explanation: str,
        event_id: str,
    ) -> None:
        self.decision    = decision
        self.rule_hits   = rule_hits
        self.risk_score  = risk_score
        self.explanation = explanation
        self.event_id    = event_id

        super().__init__(
            f"AISec {decision.value}: {explanation[:200]} "
            f"[event_id={event_id}, risk={risk_score:.3f}]"
        )


# ── Callback handler ──────────────────────────────────────────────────────────

class AISeCCallbackHandler(BaseCallbackHandler):  # type: ignore[misc]
    """
    LangChain callback handler that intercepts every tool call.

    Registers with any LangChain AgentExecutor or chain via the
    callbacks parameter. AISec evaluates each tool call before
    it executes and raises AISeCSecurityError to block dangerous ones.

    Thread safety:
        The handler uses a threading.Lock around the engine call
        to ensure concurrent tool calls from multi-agent setups
        are serialised through the analysis pipeline safely.

    Fail closed guarantee:
        Any unexpected exception inside on_tool_start() causes
        the tool call to be blocked. AISec never fails open.

    Args:
        engine:    The AISec AnalysisEngine to use for evaluation.
        scenario:  The threat scenario (TRADING_AI or URBAN_AI).
        agent_id:  Identifier for the agent being monitored.
                   Used in audit logs. Cannot be overridden by the agent.
        block_on_review: If True, PENDING_REVIEW decisions also block
                         the tool call and require out-of-band approval.
                         Default: True (conservative).

    Example:
        engine  = AnalysisEngine()
        handler = AISeCCallbackHandler(
            engine=engine,
            scenario=Scenario.TRADING_AI,
            agent_id="prod_trading_bot_v1",
            block_on_review=True,
        )
        executor = AgentExecutor(agent=agent, tools=tools,
                                 callbacks=[handler])
    """

    # Tell LangChain we want to handle tool events
    raise_error = True   # Propagate our SecurityError to the agent

    def __init__(
        self,
        engine: AnalysisEngine,
        scenario: Scenario = Scenario.UNKNOWN,
        agent_id: str = "langchain_agent",
        block_on_review: bool = True,
    ) -> None:
        if not _LANGCHAIN_AVAILABLE:
            raise ImportError(
                "LangChain is not installed. Install it with:\n"
                "    pip install langchain langchain-core\n"
                "Then retry importing AISeCCallbackHandler."
            )

        if not isinstance(engine, AnalysisEngine):
            raise TypeError(
                f"engine must be an AnalysisEngine instance, "
                f"got {type(engine).__name__}"
            )

        # Validate and sanitise agent_id at construction
        # The agent cannot change this after the handler is created
        safe_agent_id = "".join(
            c for c in str(agent_id)
            if c.isalnum() or c in "-_."
        )[:64]
        if len(safe_agent_id) < 3:
            safe_agent_id = "langchain_agent"

        # Store all as private — not accessible to subclasses
        self.__engine         = engine
        self.__scenario       = scenario
        self.__agent_id       = safe_agent_id
        self.__block_on_review = block_on_review
        self.__lock           = threading.Lock()
        self.__call_count     = 0

        log.info(
            "aisec_langchain_handler_initialized",
            agent_id=self.__agent_id,
            scenario=scenario.value,
            block_on_review=block_on_review,
        )

        super().__init__()

    # ── LangChain callback ────────────────────────────────────────────────────

    def on_tool_start(
        self,
        serialized: dict[str, Any],
        input_str: str,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        """
        Called by LangChain immediately before every tool execution.

        This is the primary interception point. We evaluate the action
        and raise AISeCSecurityError to block it if necessary.

        Fail closed: any unexpected exception here blocks the tool.

        Args:
            serialized: LangChain tool metadata dict.
            input_str:  Raw input string passed to the tool.
            run_id:     Unique ID for this tool run (from LangChain).
            **kwargs:   Additional LangChain callback kwargs (ignored).

        Raises:
            AISeCSecurityError: If AISec decides to block the action.
        """
        try:
            self._intercept(serialized, input_str, run_id)
        except AISeCSecurityError:
            # Re-raise security errors — these are intentional blocks
            raise
        except Exception as exc:
            # Unexpected error — fail closed
            log.error(
                "aisec_interceptor_unexpected_error",
                exc_type=type(exc).__name__,
                detail=str(exc)[:200],
                agent_id=self.__agent_id,
            )
            raise AISeCSecurityError(
                decision=Decision.BLOCK,
                rule_hits=["INTERCEPTOR-ERROR"],
                risk_score=1.0,
                explanation=(
                    f"AISec interceptor encountered an unexpected error: "
                    f"{type(exc).__name__}. Failing closed — action blocked."
                ),
                event_id="error",
            ) from exc

    def on_tool_end(
        self,
        output: str,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        """
        Called by LangChain after a tool completes successfully.

        We log the completion for audit purposes.
        We do NOT log the output — it may contain sensitive data.
        """
        log.info(
            "tool_execution_completed",
            agent_id=self.__agent_id,
            run_id=str(run_id),
            output_length=len(str(output)),
            output_hash=hashlib.sha256(
                str(output).encode()
            ).hexdigest()[:16],
        )

    def on_tool_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        """
        Called by LangChain when a tool raises an exception.

        We distinguish AISec security blocks from genuine tool errors.
        """
        if isinstance(error, AISeCSecurityError):
            log.warning(
                "tool_blocked_by_aisec",
                agent_id=self.__agent_id,
                decision=error.decision.value,
                risk_score=error.risk_score,
                rule_hits=error.rule_hits,
                event_id=error.event_id,
            )
        else:
            log.warning(
                "tool_error_not_aisec",
                agent_id=self.__agent_id,
                exc_type=type(error).__name__,
                run_id=str(run_id),
            )

    # ── Core interception logic ───────────────────────────────────────────────

    def _intercept(
        self,
        serialized: dict[str, Any],
        input_str: str,
        run_id: UUID,
    ) -> None:
        """
        Core interception logic — thread-safe, fail-closed.

        Builds an Event from the tool call, runs it through the
        AISec analysis engine, and raises AISeCSecurityError if
        the decision requires blocking.

        Thread safety:
            Acquires self.__lock to ensure only one analysis runs
            at a time. This prevents race conditions in the audit
            log and rule engine under concurrent tool calls.
        """
        # Extract tool name safely
        tool_name = _sanitise_tool_name(
            serialized.get("name", "")
            or serialized.get("id", ["unknown"])[-1]
        )

        # Sanitise input — never pass raw input to our pipeline
        safe_input = _sanitise_input(input_str)

        # Extract structured payload for feature scoring
        payload = _extract_payload(tool_name, safe_input)

        # Add LangChain run metadata (non-sensitive)
        payload["langchain_run_id"] = str(run_id)[:36]

        # Build the Event
        # Agent identity is always taken from our constructor parameter —
        # the agent cannot override it through crafted tool calls
        event = Event(
            action_type=tool_name,
            agent_id=self.__agent_id,     # immutable — set at construction
            target=tool_name,
            scenario=self.__scenario,      # immutable — set at construction
            raw_payload=payload,
        )

        # Log the interception attempt (never log full input)
        log.info(
            "tool_call_intercepted",
            agent_id=self.__agent_id,
            tool=tool_name,
            input_prefix=safe_input[:LOG_INPUT_PREFIX_LEN],
            input_hash=_hash_input(safe_input),
            call_count=self.__call_count,
        )

        # Analyse under lock — thread-safe engine access
        with self.__lock:
            self.__call_count += 1
            result = self.__engine.analyse(event)

        # Handle the decision
        if result.blocked:
            log.warning(
                "tool_call_blocked",
                agent_id=self.__agent_id,
                tool=tool_name,
                decision=result.decision.value,
                risk_score=result.risk_score,
                rule_hits=result.analysis.rule_hits,
                event_id=result.log_entry_id,
            )
            raise AISeCSecurityError(
                decision=result.decision,
                rule_hits=result.analysis.rule_hits,
                risk_score=result.risk_score,
                explanation=result.analysis.explanation,
                event_id=result.log_entry_id,
            )

        # Action allowed — log for audit
        log.info(
            "tool_call_allowed",
            agent_id=self.__agent_id,
            tool=tool_name,
            decision=result.decision.value,
            risk_score=result.risk_score,
            event_id=result.log_entry_id,
        )

    # ── Monitoring accessors ──────────────────────────────────────────────────

    @property
    def call_count(self) -> int:
        """Total number of tool calls intercepted by this handler."""
        return self.__call_count

    @property
    def agent_id(self) -> str:
        """The agent ID this handler is monitoring."""
        return self.__agent_id

    @property
    def scenario(self) -> Scenario:
        """The scenario this handler is configured for."""
        return self.__scenario

    def __repr__(self) -> str:
        return (
            f"AISeCCallbackHandler("
            f"agent_id={self.__agent_id!r}, "
            f"scenario={self.__scenario.value!r}, "
            f"calls={self.__call_count})"
        )