"""
aisec/core/interceptor.py
─────────────────────────────────────────────────────────────────────────────
Runtime action interceptor — the first and most critical chokepoint in AISec.

DESIGN PHILOSOPHY
─────────────────
Every action an AI agent wants to execute must pass through this module.
Nothing bypasses it. Nothing runs before it clears the action.

The interceptor is deliberately minimal. Its only job is:
    1. Receive a raw action from an AI agent.
    2. Stamp it with a unique ID and precise UTC timestamp.
    3. Validate that it is structurally complete and safe to process.
    4. Hand it to the engine.
    5. Enforce the decision — ALLOW passes through, everything else stops.
    6. Log every outcome, including the ones that never reached the engine.

It does NOT score. It does NOT make decisions. It does NOT apply rules.
Those are the engine's responsibilities. This module's only responsibility
is capture and enforcement. Separation of concerns is not optional here.

SECURITY PROPERTIES
───────────────────
• Fail-closed: any exception during interception → action is BLOCKED,
  never silently passed through.
• Immutable capture: the raw action dict is deep-copied on entry.
  The agent cannot mutate it after submission.
• Complete audit: every interception attempt is logged — including
  malformed, rejected, and errored ones. There are no silent failures.
• No trust of caller: the interceptor assumes the agent is potentially
  compromised. All fields are validated. No field is trusted by default.

THREADING
─────────
The interceptor is thread-safe. Multiple agents can submit actions
concurrently. Each interception is an independent, stateless operation
that acquires no shared locks. The engine below it is responsible for
its own thread safety.

USAGE
─────
    from aisec.core.interceptor import Interceptor
    from aisec.core.engine import AnalysisEngine
    from aisec.storage.audit import AuditLogger

    logger      = AuditLogger()
    engine      = AnalysisEngine(audit_logger=logger)
    interceptor = Interceptor(engine=engine, audit_logger=logger)

    result = interceptor.intercept(
        agent_id    = "trading_bot_01",
        action_type = "execute_trade",
        params      = {"symbol": "AAPL", "quantity": 500, "side": "buy"},
        scenario    = "trading_ai",
    )

    if result.allowed:
        execute_the_trade()

Author : AISec Project
"""

from __future__ import annotations

import copy
import logging
import threading
import uuid
from dataclasses import dataclass, field
from typing import Any

from aisec.storage.audit import AuditLogger
from aisec.storage.models import (
    AnalysisResult,
    Decision,
    Event,
    Scenario,
    Severity,
)
from aisec.utils.time import now_utc as utc_now

log = logging.getLogger(__name__)


# ── Constants ─────────────────────────────────────────────────────────────────

# Hard ceiling on how large a single params dict may be (number of keys).
# Anything larger is almost certainly a programming error or an attack.
MAX_PARAM_KEYS: int = 64

# Hard ceiling on string length for action_type and agent_id fields.
MAX_FIELD_LENGTH: int = 128

# Valid scenario values — derived from the Scenario enum.
VALID_SCENARIOS: frozenset[str] = frozenset(s.value for s in Scenario)


# ── Interception result ───────────────────────────────────────────────────────

@dataclass(frozen=True, slots=True)
class InterceptionResult:
    """
    Everything AISec knows about a single interception attempt.

    Frozen so callers cannot mutate it after the fact.
    The agent checks `allowed` and acts on it. Period.
    """

    # Unique ID for this interception — matches the event_id in the audit log.
    interception_id: str

    # The decision reached. ALLOW means the action may proceed.
    decision: Decision

    # The full analysis result from the engine, or None if the action
    # was rejected before reaching the engine.
    analysis: AnalysisResult | None

    # Human-readable explanation of why this decision was reached.
    reason: str

    # Precise UTC timestamp of when interception was completed.
    timestamp: str

    # Whether the action is permitted to proceed.
    @property
    def allowed(self) -> bool:
        return self.decision == Decision.ALLOW

    # Whether the action was blocked before reaching the engine
    # (malformed input, validation failure, internal error).
    @property
    def pre_engine_rejection(self) -> bool:
        return self.analysis is None


# ── Validation errors ─────────────────────────────────────────────────────────

class InterceptionError(Exception):
    """
    Raised when an action is structurally invalid and cannot be processed.

    This is not a security alert — it signals a programming error in the
    agent submitting the action. The interceptor still logs it and blocks.
    """


# ── The interceptor ───────────────────────────────────────────────────────────

class Interceptor:
    """
    Runtime action interceptor — the mandatory gateway for all AI agent actions.

    Every action submitted to AISec enters here and only here.

    The interceptor is stateless between calls. It holds references to the
    engine and audit logger but maintains no per-action state of its own.
    """

    def __init__(
        self,
        engine: Any,           # AnalysisEngine — typed as Any to avoid circular import
        audit_logger: AuditLogger,
        *,
        strict_mode: bool = True,
    ) -> None:
        """
        Initialise the interceptor.

        Args:
            engine:       The analysis engine. Must expose .analyse(event) -> AnalysisResult.
            audit_logger: The audit logger. Every outcome is written here.
            strict_mode:  If True (default), any validation failure immediately
                          blocks the action. If False, mild validation issues
                          produce warnings but allow processing to continue.
                          Never set False in production.
        """
        if engine is None:
            raise ValueError("Interceptor requires a real engine. Received None.")
        if audit_logger is None:
            raise ValueError("Interceptor requires a real audit logger. Received None.")

        self._engine       = engine
        self._audit        = audit_logger
        self._strict       = strict_mode
        self._lock         = threading.Lock()   # protects _stats only
        self._stats: dict[str, int] = {
            "total":    0,
            "allowed":  0,
            "blocked":  0,
            "reviewed": 0,
            "escalated":0,
            "errors":   0,
        }

    # ── Public interface ──────────────────────────────────────────────────────

    def intercept(
        self,
        agent_id:    str,
        action_type: str,
        params:      dict[str, Any],
        scenario:    str,
    ) -> InterceptionResult:
        """
        Intercept one AI agent action.

        This is the only public method agents should call. It is synchronous,
        blocking, and fail-closed. If anything goes wrong, the action is blocked.

        Args:
            agent_id:    Identifier of the AI agent submitting the action.
                         Must be a non-empty string ≤ MAX_FIELD_LENGTH chars.
            action_type: The type of action being requested.
                         Must be a non-empty string ≤ MAX_FIELD_LENGTH chars.
            params:      Action parameters. Immutable copy is taken on entry.
                         Must be a dict with ≤ MAX_PARAM_KEYS keys.
            scenario:    The operating scenario ("trading_ai" or "urban_ai").
                         Must match a valid Scenario value.

        Returns:
            InterceptionResult with decision and full audit trail.
            Check `.allowed` to determine if the action may proceed.

        Raises:
            Never. All exceptions are caught internally and result in a BLOCK.
        """
        interception_id = str(uuid.uuid4())
        timestamp       = utc_now()

        try:
            # Step 1 — Deep copy params immediately. Agent cannot mutate them.
            safe_params = copy.deepcopy(params)

            # Step 2 — Validate all inputs.
            self._validate(agent_id, action_type, safe_params, scenario)

            # Step 3 — Build the canonical Event.
            event = Event(
                event_id    = interception_id,
                agent_id    = agent_id.strip(),
                action_type = action_type.strip(),
                params      = safe_params,
                scenario    = Scenario(scenario),
                timestamp   = timestamp,
            )

            # Step 4 — Send to analysis engine.
            analysis: AnalysisResult = self._engine.analyse(event)

            # Step 5 — Build result.
            result = InterceptionResult(
                interception_id = interception_id,
                decision        = analysis.decision,
                analysis        = analysis,
                reason          = analysis.explanation,
                timestamp       = timestamp,
            )

        except InterceptionError as exc:
            # Validation failure — block immediately, log it.
            log.warning("Interception validation failure [%s]: %s", interception_id, exc)
            result = self._blocked_result(
                interception_id = interception_id,
                timestamp       = timestamp,
                reason          = f"Validation failure: {exc}",
            )
            self._audit_pre_engine_rejection(
                interception_id = interception_id,
                agent_id        = agent_id if isinstance(agent_id, str) else "INVALID",
                action_type     = action_type if isinstance(action_type, str) else "INVALID",
                scenario        = scenario if isinstance(scenario, str) else "INVALID",
                reason          = str(exc),
                timestamp       = timestamp,
            )

        except Exception as exc:
            # Internal error — fail closed, never fail open.
            log.error(
                "Interceptor internal error [%s]: %s",
                interception_id, exc, exc_info=True
            )
            result = self._blocked_result(
                interception_id = interception_id,
                timestamp       = timestamp,
                reason          = "Internal interceptor error — action blocked for safety.",
            )
            self._audit_pre_engine_rejection(
                interception_id = interception_id,
                agent_id        = agent_id if isinstance(agent_id, str) else "INVALID",
                action_type     = action_type if isinstance(action_type, str) else "INVALID",
                scenario        = scenario if isinstance(scenario, str) else "INVALID",
                reason          = f"Internal error: {type(exc).__name__}",
                timestamp       = timestamp,
            )
            with self._lock:
                self._stats["errors"] += 1

        # Step 6 — Update stats.
        self._update_stats(result.decision)

        # Step 7 — Log the enforcement outcome.
        self._log_enforcement(result, agent_id if isinstance(agent_id, str) else "INVALID")

        return result

    # ── Statistics ────────────────────────────────────────────────────────────

    def stats(self) -> dict[str, int]:
        """Return a snapshot of interception statistics (thread-safe)."""
        with self._lock:
            return dict(self._stats)

    def reset_stats(self) -> None:
        """Reset statistics counters. Intended for testing only."""
        with self._lock:
            for key in self._stats:
                self._stats[key] = 0

    # ── Private helpers ───────────────────────────────────────────────────────

    def _validate(
        self,
        agent_id:    Any,
        action_type: Any,
        params:      Any,
        scenario:    Any,
    ) -> None:
        """
        Validate all fields of an incoming action.

        Raises InterceptionError on any validation failure.
        All checks must pass before the event is constructed.
        """
        # agent_id
        if not isinstance(agent_id, str):
            raise InterceptionError(
                f"agent_id must be str, got {type(agent_id).__name__}"
            )
        if not agent_id.strip():
            raise InterceptionError("agent_id must not be empty or whitespace.")
        if len(agent_id) > MAX_FIELD_LENGTH:
            raise InterceptionError(
                f"agent_id exceeds maximum length of {MAX_FIELD_LENGTH} characters."
            )

        # action_type
        if not isinstance(action_type, str):
            raise InterceptionError(
                f"action_type must be str, got {type(action_type).__name__}"
            )
        if not action_type.strip():
            raise InterceptionError("action_type must not be empty or whitespace.")
        if len(action_type) > MAX_FIELD_LENGTH:
            raise InterceptionError(
                f"action_type exceeds maximum length of {MAX_FIELD_LENGTH} characters."
            )

        # params
        if not isinstance(params, dict):
            raise InterceptionError(
                f"params must be a dict, got {type(params).__name__}"
            )
        if len(params) > MAX_PARAM_KEYS:
            raise InterceptionError(
                f"params has {len(params)} keys — exceeds maximum of {MAX_PARAM_KEYS}. "
                "This is almost certainly a programming error."
            )

        # scenario
        if not isinstance(scenario, str):
            raise InterceptionError(
                f"scenario must be str, got {type(scenario).__name__}"
            )
        if scenario not in VALID_SCENARIOS:
            raise InterceptionError(
                f"Unknown scenario '{scenario}'. "
                f"Valid scenarios: {sorted(VALID_SCENARIOS)}"
            )

    def _blocked_result(
        self,
        interception_id: str,
        timestamp:       str,
        reason:          str,
    ) -> InterceptionResult:
        """Construct a pre-engine BLOCK result."""
        return InterceptionResult(
            interception_id = interception_id,
            decision        = Decision.BLOCK,
            analysis        = None,
            reason          = reason,
            timestamp       = timestamp,
        )

    def _update_stats(self, decision: Decision) -> None:
        """Update internal counters (thread-safe)."""
        mapping = {
            Decision.ALLOW:          "allowed",
            Decision.BLOCK:          "blocked",
            Decision.PENDING_REVIEW: "reviewed",
            Decision.ESCALATE:       "escalated",
        }
        with self._lock:
            self._stats["total"] += 1
            key = mapping.get(decision)
            if key:
                self._stats[key] += 1

    def _log_enforcement(self, result: InterceptionResult, agent_id: str) -> None:
        """
        Write enforcement outcome to the Python logger.

        This is a secondary log for operator visibility — the primary audit
        record is written by the engine or the pre-engine rejection handler.
        """
        level = logging.INFO if result.allowed else logging.WARNING
        log.log(
            level,
            "INTERCEPT [%s] agent=%s decision=%s reason=%s",
            result.interception_id,
            agent_id,
            result.decision.value,
            result.reason,
        )

    def _audit_pre_engine_rejection(
        self,
        interception_id: str,
        agent_id:        str,
        action_type:     str,
        scenario:        str,
        reason:          str,
        timestamp:       str,
    ) -> None:
        """
        Write a pre-engine rejection to the audit chain.

        Actions that never reach the engine still get an audit record.
        There are no silent failures in AISec.
        """
        try:
            self._audit.log(
                record_type = "pre_engine_rejection",
                payload     = {
                    "interception_id": interception_id,
                    "agent_id":        agent_id,
                    "action_type":     action_type,
                    "scenario":        scenario,
                    "decision":        Decision.BLOCK.value,
                    "reason":          reason,
                    "timestamp":       timestamp,
                },
            )
        except Exception as exc:
            # Audit logger failure is serious but must not raise here —
            # the block decision is already made.
            log.critical(
                "AUDIT LOGGER FAILED during pre-engine rejection [%s]: %s",
                interception_id, exc, exc_info=True,
            )