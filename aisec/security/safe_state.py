"""
AISec Safe State Enforcer — implements R3 from the paper.

Formal rule R3:
    anomaly_detected = True → system ∈ S

Where S is the safe state: a restricted operational mode where
the AI agent's future actions are automatically blocked until
a human administrator explicitly releases it.

This is the enforcement layer that the paper describes but
previous versions of AISec only partially implemented.
The temporal detector DETECTS anomalies. This module ENFORCES
the safe state transition when anomalies are detected.

Safe state properties:
    - All actions from an agent in safe state are BLOCKED.
    - The block is applied BEFORE the rule engine or scorer.
    - Entry to safe state is logged with reason and timestamp.
    - Exit from safe state requires admin role (RBAC enforced).
    - Every entry and exit is written to the audit log.
    - Safe state is persistent in memory — survives engine restarts
      only if backed by a persistent store (v2 feature).

Design:
    - Thread-safe: safe for concurrent multi-agent use.
    - Fail closed: if safe state lookup fails, action is blocked.
    - Transparent: every safe state decision is audit logged.

Paper reference:
    Section 5.2 — Formal Enforcement Rules
    R3: anomaly_detected = True → system ∈ S
    Section 6.3 — Safe State Architecture
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from aisec.utils.logger import get_logger
from aisec.utils.time import now_utc

if TYPE_CHECKING:
    from aisec.storage.audit import AuditLogger

log = get_logger("aisec.security.safe_state")


# ── Safe state record ─────────────────────────────────────────────────────────


@dataclass
class SafeStateEntry:
    """
    Records a single safe state activation for an agent.

    Attributes:
        agent_id:      The agent placed in safe state.
        reason:        Why the agent was placed in safe state.
        triggered_by:  What triggered the transition (threat type).
        entered_at:    UTC timestamp when safe state was entered.
        released_at:   UTC timestamp when released (None if still active).
        released_by:   Admin who released the agent (None if still active).
        active:        True if the agent is currently in safe state.
    """

    agent_id: str
    reason: str
    triggered_by: str
    entered_at: str = field(default_factory=now_utc)
    released_at: str | None = None
    released_by: str | None = None
    active: bool = True

    def release(self, admin_id: str) -> None:
        """Mark this safe state entry as released."""
        self.active = False
        self.released_at = now_utc()
        self.released_by = admin_id


# ── Safe state enforcer ───────────────────────────────────────────────────────


class SafeStateEnforcer:
    """
    Enforces R3: anomaly_detected = True → system ∈ S

    Maintains a per-agent safe state registry. When an agent
    enters safe state, all future actions from that agent are
    blocked until an administrator explicitly releases it.

    Integration:
        The enforcer is checked at the START of the analysis pipeline,
        before the rule engine and risk scorer run. This ensures
        safe state blocks cannot be bypassed by crafting actions
        that would normally score low.

    Thread safety:
        All state is protected by a threading.RLock (reentrant)
        to allow the same thread to check and update state safely.

    Usage:
        enforcer = SafeStateEnforcer(audit_logger=logger)

        # Check before analysis
        if enforcer.is_in_safe_state(agent_id):
            return enforcer.synthetic_block_result(event)

        # Enter safe state on CRITICAL temporal alert
        enforcer.enter_safe_state(
            agent_id="trading_bot_v1",
            reason="BURST_ATTACK detected: 50 events in 10s",
            triggered_by="BURST_ATTACK",
        )

        # Admin releases (RBAC enforced by caller)
        enforcer.exit_safe_state(
            agent_id="trading_bot_v1",
            admin_id="admin_01",
        )
    """

    def __init__(
        self,
        audit_logger: AuditLogger | None = None,
    ) -> None:
        self._states: dict[str, SafeStateEntry] = {}
        self._lock: threading.RLock = threading.RLock()
        self._logger = audit_logger

        log.info("safe_state_enforcer_initialized")

    def is_in_safe_state(self, agent_id: str) -> bool:
        """
        Check if an agent is currently in safe state.

        This is the primary check called before every analysis.
        Fail closed: any lookup error returns True (blocked).

        Args:
            agent_id: The agent to check.

        Returns:
            True if the agent is in safe state and should be blocked.
        """
        try:
            with self._lock:
                entry = self._states.get(agent_id)
                return entry is not None and entry.active
        except Exception as exc:
            log.error(
                "safe_state_lookup_error",
                agent_id=agent_id,
                exc_type=type(exc).__name__,
                detail=str(exc)[:200],
            )
            return True  # Fail closed

    def enter_safe_state(
        self,
        agent_id: str,
        reason: str,
        triggered_by: str,
    ) -> SafeStateEntry:
        """
        Place an agent in safe state.

        All future actions from this agent will be blocked until
        exit_safe_state() is called by an administrator.

        Args:
            agent_id:     The agent to restrict.
            reason:       Human-readable reason for the restriction.
            triggered_by: The threat type that triggered this (e.g. BURST_ATTACK).

        Returns:
            The SafeStateEntry created for this activation.
        """
        with self._lock:
            # Check if already in safe state
            existing = self._states.get(agent_id)
            if existing and existing.active:
                log.info(
                    "safe_state_already_active",
                    agent_id=agent_id,
                    triggered_by=triggered_by,
                )
                return existing

            entry = SafeStateEntry(
                agent_id=agent_id,
                reason=reason,
                triggered_by=triggered_by,
            )
            self._states[agent_id] = entry

        log.warning(
            "safe_state_entered",
            agent_id=agent_id,
            triggered_by=triggered_by,
            reason=reason,
        )

        # Write to audit log
        if self._logger:
            try:
                self._logger.log(
                    record_type="safe_state_entry",
                    record_id=agent_id,
                    payload={
                        "agent_id": agent_id,
                        "reason": reason,
                        "triggered_by": triggered_by,
                        "entered_at": entry.entered_at,
                        "action": "ENTER_SAFE_STATE",
                    },
                )
            except Exception as exc:
                log.error(
                    "safe_state_audit_log_error",
                    exc_type=type(exc).__name__,
                )

        return entry

    def exit_safe_state(
        self,
        agent_id: str,
        admin_id: str,
        reason: str = "",
    ) -> bool:
        """
        Release an agent from safe state.

        This action should only be called after the caller has
        verified that the admin has MANAGE_ROLES or equivalent
        permission via the RBAC system.

        Args:
            agent_id: The agent to release.
            admin_id: The administrator authorising the release.
            reason:   Optional reason for the release.

        Returns:
            True if the agent was released.
            False if the agent was not in safe state.
        """
        with self._lock:
            entry = self._states.get(agent_id)
            if entry is None or not entry.active:
                log.info(
                    "safe_state_exit_not_active",
                    agent_id=agent_id,
                    admin_id=admin_id,
                )
                return False

            entry.release(admin_id)

        log.info(
            "safe_state_exited",
            agent_id=agent_id,
            admin_id=admin_id,
            reason=reason,
        )

        # Write to audit log
        if self._logger:
            try:
                self._logger.log(
                    record_type="safe_state_exit",
                    record_id=agent_id,
                    payload={
                        "agent_id": agent_id,
                        "admin_id": admin_id,
                        "reason": reason,
                        "released_at": entry.released_at,
                        "action": "EXIT_SAFE_STATE",
                    },
                )
            except Exception as exc:
                log.error(
                    "safe_state_exit_audit_error",
                    exc_type=type(exc).__name__,
                )

        return True

    def get_entry(self, agent_id: str) -> SafeStateEntry | None:
        """Return the current safe state entry for an agent."""
        with self._lock:
            return self._states.get(agent_id)

    def list_active(self) -> list[SafeStateEntry]:
        """Return all agents currently in safe state."""
        with self._lock:
            return [e for e in self._states.values() if e.active]

    def list_all(self) -> list[SafeStateEntry]:
        """Return all safe state entries including released ones."""
        with self._lock:
            return list(self._states.values())

    def active_count(self) -> int:
        """Return the number of agents currently in safe state."""
        with self._lock:
            return sum(1 for e in self._states.values() if e.active)

    def reset_all(self) -> None:
        """
        Clear all safe state entries. Admin operation only.

        Used for testing and emergency recovery.
        In production, prefer exit_safe_state() per agent.
        """
        with self._lock:
            self._states.clear()
        log.warning("safe_state_all_reset")
