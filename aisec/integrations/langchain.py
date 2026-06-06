"""
AISec LangChain integration — callback-based runtime interceptor.

Integrates AISec into LangChain agents by registering a callback
handler that intercepts tool calls before execution.

Security design:
    - Fail closed: any exception inside AISec blocks the tool call.
    - Input sanitisation: tool names and inputs are bounded and cleaned.
    - Prompt-injection detection: tool inputs are analysed before execution.
    - No sensitive data logging: raw tool inputs are never logged.
    - Auditability: AISec events include hashes and security metadata.
    - Identity enforcement: agent_id is fixed at handler construction.
    - Thread safety: engine analysis is protected by a lock.

Usage:
    from langchain.agents import AgentExecutor
    from aisec.integrations.langchain import AISeCCallbackHandler
    from aisec.core.engine import AnalysisEngine
    from aisec.storage.models import Scenario

    engine = AnalysisEngine()
    handler = AISeCCallbackHandler(
        engine=engine,
        scenario=Scenario.TRADING_AI,
        agent_id="trading_bot_prod",
    )

    executor = AgentExecutor(
        agent=agent,
        tools=tools,
        callbacks=[handler],
        verbose=False,
    )

    executor.invoke({"input": "Analyse the market"})

Requirements:
    pip install langchain langchain-core
"""

from __future__ import annotations

import hashlib
import re
import threading
from typing import Any
from uuid import UUID

from aisec.core.engine import AnalysisEngine
from aisec.security.prompt_injection import (
    InputSource,
    PromptInjectionContext,
    PromptInjectionDetector,
    RecommendedAction,
)
from aisec.storage.models import Decision, Event, Scenario
from aisec.utils.logger import get_logger

log = get_logger("aisec.integrations.langchain")


# ── LangChain import guard ────────────────────────────────────────────────────

try:
    from langchain_core.callbacks import BaseCallbackHandler
    from langchain_core.outputs import LLMResult

    _LANGCHAIN_AVAILABLE = True
except ImportError:
    try:
        from langchain.callbacks.base import BaseCallbackHandler  # type: ignore[no-redef]
        from langchain.schema import LLMResult  # type: ignore[no-redef]

        _LANGCHAIN_AVAILABLE = True
    except ImportError:
        _LANGCHAIN_AVAILABLE = False
        BaseCallbackHandler = object  # type: ignore[assignment,misc]


# ── Constants ─────────────────────────────────────────────────────────────────

MAX_INPUT_LEN: int = 2_048
MAX_TOOL_NAME_LEN: int = 128


# ── Security helpers ──────────────────────────────────────────────────────────


def _sanitise_tool_name(name: str) -> str:
    """
    Sanitise a LangChain tool name before it enters AISec.

    Allows only alphanumeric characters, underscores, and dots.
    Hyphens are removed to preserve the original AISec adapter
    test contract.
    """
    if not name or not isinstance(name, str):
        return "unknown_tool"

    safe = "".join(c for c in name if c.isalnum() or c in "_.")[:MAX_TOOL_NAME_LEN]

    return safe if safe else "unknown_tool"


def _extract_tool_name(serialized: dict[str, Any]) -> str:
    """
    Extract a tool name from LangChain serialized metadata safely.

    LangChain versions may expose tool identity in slightly different fields.
    """
    raw_name = serialized.get("name")

    if not raw_name:
        raw_id = serialized.get("id")
        if isinstance(raw_id, list) and raw_id:
            raw_name = str(raw_id[-1])
        elif raw_id:
            raw_name = str(raw_id)
        else:
            raw_name = "unknown_tool"

    return _sanitise_tool_name(str(raw_name))


def _sanitise_input(raw: Any) -> str:
    """
    Convert tool input to a bounded string for analysis.

    The returned value may still contain sensitive text, so it must not be
    logged directly. Use hashes for audit/logging.
    """
    try:
        text = str(raw)
    except Exception:
        text = "[unserializable input]"

    return text[:MAX_INPUT_LEN]


def _hash_input(raw: str) -> str:
    """Return short SHA-256 digest for privacy-preserving correlation."""
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def _extract_payload(tool_name: str, tool_input: str) -> dict[str, Any]:
    """
    Extract structured security-relevant payload from a tool call.

    This intentionally stores hashes and derived signals, not raw input.
    """
    payload: dict[str, Any] = {
        "input_hash": _hash_input(tool_input),
        "input_length": len(tool_input),
        "tool_name": tool_name,
    }

    amount_match = re.search(r"\b(\d[\d,]*(?:\.\d+)?)\b", tool_input)
    if amount_match:
        try:
            amount_str = amount_match.group(1).replace(",", "")
            payload["amount"] = float(amount_str)
        except ValueError:
            pass

    after_hours_keywords = {
        "after_hours",
        "after-hours",
        "afterhours",
        "off_hours",
        "overnight",
        "weekend",
    }
    if any(keyword in tool_input.lower() for keyword in after_hours_keywords):
        payload["after_hours"] = True

    burst_match = re.search(
        r"burst[_\s]?rate[=:\s]+(\d+(?:\.\d+)?)",
        tool_input.lower(),
    )
    if burst_match:
        try:
            payload["burst_rate"] = float(burst_match.group(1))
        except ValueError:
            pass

    zone_match = re.search(r"zone[=:\s]+([A-Za-z0-9_-]+)", tool_input)
    if zone_match:
        payload["zone"] = zone_match.group(1)[:32]

    return payload


# ── Security exception ────────────────────────────────────────────────────────


class AISeCSecurityError(Exception):
    """
    Raised when AISec blocks a LangChain tool call.
    """

    def __init__(
        self,
        decision: Decision,
        rule_hits: list[str],
        risk_score: float,
        explanation: str,
        event_id: str,
    ) -> None:
        self.decision = decision
        self.rule_hits = rule_hits
        self.risk_score = risk_score
        self.explanation = explanation
        self.event_id = event_id

        super().__init__(
            f"AISec {decision.value}: {explanation[:200]} "
            f"[event_id={event_id}, risk={risk_score:.3f}]"
        )


# ── Callback handler ──────────────────────────────────────────────────────────


class AISeCCallbackHandler(BaseCallbackHandler):  # type: ignore[misc]
    """
    LangChain callback handler that intercepts every tool call.

    AISec evaluates each tool call before execution. Dangerous calls are
    blocked by raising AISeCSecurityError.

    Security invariant:
        prompt injection + dangerous tool context must never execute.
    """

    raise_error = True

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

        safe_agent_id = "".join(c for c in str(agent_id) if c.isalnum() or c in "_.-")[
            :64
        ]

        if len(safe_agent_id) < 3:
            safe_agent_id = "langchain_agent"

        self.__engine = engine
        self.__scenario = scenario
        self.__agent_id = safe_agent_id
        self.__block_on_review = block_on_review
        self.__lock = threading.Lock()
        self.__call_count = 0
        self.__blocked_count = 0
        self.__injection_detector = PromptInjectionDetector()

        log.info(
            "aisec_langchain_handler_initialized",
            agent_id=self.__agent_id,
            scenario=scenario.value,
            block_on_review=block_on_review,
        )

        super().__init__()

    # ── LangChain callbacks ───────────────────────────────────────────────────

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

        Fail closed: any unexpected exception blocks the tool call.
        """
        try:
            self._intercept(serialized, input_str, run_id)
        except AISeCSecurityError:
            raise
        except Exception as exc:
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
                    f"AISec interceptor encountered unexpected error: "
                    f"{type(exc).__name__}. Failing closed."
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
        Called after tool execution.

        Does not log raw output because tool output may contain secrets.
        """
        output_text = str(output)

        log.info(
            "tool_execution_completed",
            agent_id=self.__agent_id,
            run_id=str(run_id),
            output_length=len(output_text),
            output_hash=hashlib.sha256(output_text.encode("utf-8")).hexdigest()[:16],
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
        Called when a tool raises an exception.
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
        Build an AISec event from a LangChain tool call, run analysis,
        and block if AISec or the prompt-injection detector requires it.
        """
        tool_name = _extract_tool_name(serialized)
        safe_input = _sanitise_input(input_str)

        payload = _extract_payload(tool_name, safe_input)
        payload["langchain_run_id"] = str(run_id)[:36]

        injection_context = PromptInjectionContext(
            source=InputSource.TOOL_ARGUMENTS,
            tool_name=tool_name,
            scenario=self.__scenario.value,
        )

        injection_result = self.__injection_detector.analyse(
            safe_input,
            context=injection_context,
        )

        injection_forces_block = False
        injection_forces_review = False

        if injection_result.is_injection:
            payload["injection_detected"] = True
            payload["injection_type"] = (
                injection_result.injection_type.value
                if injection_result.injection_type
                else "unknown"
            )
            payload["injection_severity"] = injection_result.severity.value
            payload["injection_confidence"] = injection_result.confidence
            payload["injection_risk_score"] = injection_result.risk_score
            payload["injection_recommended_action"] = (
                injection_result.recommended_action.value
            )
            payload["injection_rule_hits"] = injection_result.matched_rule_ids

            payload["keyword_risk_score"] = max(
                float(payload.get("keyword_risk_score", 0.0)),
                injection_result.risk_score,
            )

            injection_forces_block = (
                injection_result.recommended_action == RecommendedAction.BLOCK
            )
            injection_forces_review = (
                injection_result.recommended_action == RecommendedAction.REVIEW
            )

            log.warning(
                "prompt_injection_detected_in_langchain_tool_input",
                agent_id=self.__agent_id,
                tool=tool_name,
                injection_type=payload["injection_type"],
                severity=payload["injection_severity"],
                confidence=injection_result.confidence,
                risk_score=injection_result.risk_score,
                recommended_action=payload["injection_recommended_action"],
                matched_rule_ids=injection_result.matched_rule_ids,
                input_hash=_hash_input(safe_input),
                text_hash=injection_result.text_hash,
            )
        else:
            payload["injection_detected"] = False

        event = Event(
            action_type=tool_name,
            agent_id=self.__agent_id,
            target=tool_name,
            scenario=self.__scenario,
            raw_payload=payload,
        )

        log.info(
            "tool_call_intercepted",
            agent_id=self.__agent_id,
            tool=tool_name,
            input_hash=_hash_input(safe_input),
            input_length=len(safe_input),
            call_count=self.__call_count,
            injection_detected=payload["injection_detected"],
        )

        with self.__lock:
            self.__call_count += 1
            result = self.__engine.analyse(event)

        should_block = result.blocked
        forced_decision = result.decision
        forced_rule_hits = list(result.analysis.rule_hits)
        forced_risk_score = result.risk_score
        forced_explanation = result.analysis.explanation

        if injection_forces_block:
            should_block = True
            forced_decision = Decision.BLOCK
            forced_rule_hits = ["PROMPT-INJECTION"] + injection_result.matched_rule_ids
            forced_risk_score = max(
                0.95, result.risk_score, injection_result.risk_score
            )
            forced_explanation = (
                "Prompt injection detected in LangChain tool input. "
                f"type={payload.get('injection_type')}, "
                f"severity={payload.get('injection_severity')}, "
                f"confidence={injection_result.confidence:.2f}. "
                "Failing closed before tool execution."
            )

        elif injection_forces_review and self.__block_on_review:
            should_block = True
            forced_decision = Decision.PENDING_REVIEW
            forced_rule_hits = [
                "PROMPT-INJECTION-REVIEW"
            ] + injection_result.matched_rule_ids
            forced_risk_score = max(
                0.75, result.risk_score, injection_result.risk_score
            )
            forced_explanation = (
                "Prompt injection indicators detected in LangChain tool input. "
                f"type={payload.get('injection_type')}, "
                f"severity={payload.get('injection_severity')}, "
                f"confidence={injection_result.confidence:.2f}. "
                "Tool call requires human review."
            )

        if should_block:
            with self.__lock:
                self.__blocked_count += 1

            log.warning(
                "tool_call_blocked",
                agent_id=self.__agent_id,
                tool=tool_name,
                decision=forced_decision.value,
                risk_score=forced_risk_score,
                rule_hits=forced_rule_hits,
                event_id=result.log_entry_id,
                injection_detected=payload["injection_detected"],
            )

            raise AISeCSecurityError(
                decision=forced_decision,
                rule_hits=forced_rule_hits,
                risk_score=forced_risk_score,
                explanation=forced_explanation,
                event_id=result.log_entry_id,
            )

        log.info(
            "tool_call_allowed",
            agent_id=self.__agent_id,
            tool=tool_name,
            decision=result.decision.value,
            risk_score=result.risk_score,
            event_id=result.log_entry_id,
            injection_detected=payload["injection_detected"],
        )

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
        """The monitored agent ID."""
        return self.__agent_id

    @property
    def scenario(self) -> Scenario:
        """The configured AISec scenario."""
        return self.__scenario

    @property
    def block_rate(self) -> float:
        """Fraction of intercepted calls blocked."""
        if self.__call_count == 0:
            return 0.0
        return self.__blocked_count / self.__call_count

    def __repr__(self) -> str:
        return (
            f"AISeCCallbackHandler("
            f"agent_id={self.__agent_id!r}, "
            f"scenario={self.__scenario.value!r}, "
            f"calls={self.__call_count}, "
            f"blocked={self.__blocked_count})"
        )
