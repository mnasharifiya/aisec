"""
AISec temporal anomaly detector.

Analyses sequences of AI agent actions over time to detect
attack patterns that are invisible when actions are evaluated
in isolation.

Core insight:
    A single trade at $999,999 is allowed (below $1M rule).
    Ten trades at $999,999 each in 60 seconds is an attack.
    Only temporal analysis can detect the second case.

Threat patterns detected:

    BURST_ATTACK
        An agent submits an unusually high number of actions
        in a short time window. Indicates automated execution
        attempting to overwhelm human oversight.
        Threshold: > N actions in T seconds.

    THRESHOLD_PROBING
        An agent repeatedly submits actions scoring just below
        the BLOCK threshold (0.75-0.80). Indicates systematic
        exploration of our detection boundaries.
        Threshold: > K actions scoring in [0.65, 0.80] in window.

    ESCALATING_RISK
        An agent's average risk score increases monotonically
        over the last N actions. Indicates progressive boundary
        testing before a major attack.
        Threshold: average risk increases by > 0.15 over window.

    CUMULATIVE_EXPOSURE
        The sum of financial amounts in the window exceeds
        a threshold even though no single trade triggered rules.
        Threshold: sum of amounts > $5,000,000 in 60 seconds.

    REPEATED_BLOCK_EVASION
        An agent's action was blocked but it keeps submitting
        similar actions with minor variations.
        Threshold: > 2 similar action types after a block.

Design principles:
    - Per-agent sliding windows — each agent is tracked independently.
    - Thread-safe — safe for concurrent multi-agent use.
    - Configurable thresholds — operators can tune per environment.
    - Memory-bounded — windows are capped to prevent unbounded growth.
    - Fail open for temporal analysis — if temporal analysis fails,
      the base engine decision stands. Temporal is additive.

Paper reference:
    Section 8 — Behavioural Monitoring and Anomaly Detection.
    Section 9 — Temporal Correlation and Sequence Analysis.
"""

from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Deque

from aisec.core.engine import EngineResult
from aisec.utils.logger import get_logger

log = get_logger("aisec.core.temporal")


# ── Threat pattern types ──────────────────────────────────────────────────────


class TemporalThreat(Enum):
    """Classification of detected temporal threat patterns."""

    BURST_ATTACK = auto()
    THRESHOLD_PROBING = auto()
    ESCALATING_RISK = auto()
    CUMULATIVE_EXPOSURE = auto()
    REPEATED_BLOCK_EVASION = auto()


# ── Configuration ─────────────────────────────────────────────────────────────


@dataclass
class TemporalConfig:
    """
    Configuration for temporal anomaly detection thresholds.

    All thresholds have documented rationale. Change only with
    empirical justification and updated tests.

    Attributes:
        window_seconds:
            Size of the sliding time window in seconds.
            Events older than this are discarded.
            Default: 60 seconds.

        burst_threshold:
            Number of actions in window_seconds that triggers
            BURST_ATTACK detection.
            Default: 20 actions (more than one every 3 seconds
            sustained for a minute is suspicious for any AI agent).

        probe_threshold:
            Number of actions scoring in [probe_score_min, probe_score_max]
            that triggers THRESHOLD_PROBING detection.
            Default: 5 actions.

        probe_score_min / probe_score_max:
            Score range that indicates boundary probing.
            [0.65, 0.80] is just below our BLOCK threshold of 0.80.

        escalation_delta:
            Minimum increase in average risk score over the window
            to trigger ESCALATING_RISK.
            Default: 0.15 (15 percentage point increase).

        escalation_min_events:
            Minimum events in window before escalation is evaluated.
            Prevents false positives from single-event windows.

        cumulative_amount_threshold:
            Maximum total financial amount (sum of all trades in window)
            before CUMULATIVE_EXPOSURE is triggered.
            Default: $5,000,000.

        evasion_threshold:
            Number of similar actions after a block before
            REPEATED_BLOCK_EVASION is triggered.
            Default: 2.

        max_window_size:
            Maximum number of events stored per agent window.
            Prevents unbounded memory growth.
            Default: 1000 events.
    """

    window_seconds: float = 60.0
    burst_threshold: int = 20
    probe_threshold: int = 5
    probe_score_min: float = 0.65
    probe_score_max: float = 0.80
    escalation_delta: float = 0.15
    escalation_min_events: int = 5
    cumulative_amount_threshold: float = 5_000_000.0
    evasion_threshold: int = 2
    max_window_size: int = 1_000


# Default configuration — used when no custom config is provided
DEFAULT_CONFIG = TemporalConfig()


# ── Temporal alert ────────────────────────────────────────────────────────────


@dataclass
class TemporalAlert:
    """
    Alert raised when a temporal threat pattern is detected.

    Attributes:
        agent_id:     The agent exhibiting the suspicious pattern.
        threat:       The type of threat pattern detected.
        severity:     Severity level: "HIGH" or "CRITICAL".
        description:  Human-readable description for analysts.
        evidence:     Quantitative evidence supporting the detection.
        timestamp:    UTC timestamp when the alert was raised.
    """

    agent_id: str
    threat: TemporalThreat
    severity: str
    description: str
    evidence: dict[str, float | int | str]
    timestamp: float = field(default_factory=time.time)

    def __str__(self) -> str:
        return (
            f"[{self.severity}] {self.threat.name} — "
            f"agent={self.agent_id}: {self.description}"
        )


# ── Event record ──────────────────────────────────────────────────────────────


@dataclass
class WindowEvent:
    """
    A single event record stored in the agent's sliding window.

    Attributes:
        action_type: The action type of the event.
        risk_score:  The risk score assigned by the engine.
        was_blocked: True if the engine blocked this event.
        amount:      Financial amount from payload (0.0 if not applicable).
        timestamp:   Unix timestamp when the event was analysed.
    """

    action_type: str
    risk_score: float
    was_blocked: bool
    amount: float
    timestamp: float


# ── Agent window ──────────────────────────────────────────────────────────────


class AgentWindow:
    """
    Sliding time window of recent events for a single agent.

    Automatically expires events older than window_seconds.
    Bounded to max_window_size to prevent memory exhaustion.

    Thread safety: all methods protected by an internal lock.
    """

    def __init__(self, config: TemporalConfig) -> None:
        self._config: TemporalConfig = config
        self._events: Deque[WindowEvent] = deque()
        self._lock: threading.Lock = threading.Lock()

    def add(self, event: WindowEvent) -> None:
        """Add a new event to the window and expire old events."""
        with self._lock:
            self._events.append(event)
            self._expire()
            # Enforce max window size
            while len(self._events) > self._config.max_window_size:
                self._events.popleft()

    def get_recent(self) -> list[WindowEvent]:
        """Return all events currently in the window."""
        with self._lock:
            self._expire()
            return list(self._events)

    def size(self) -> int:
        """Return the current window size."""
        with self._lock:
            self._expire()
            return len(self._events)

    def clear(self) -> None:
        """Clear all events from the window."""
        with self._lock:
            self._events.clear()

    def _expire(self) -> None:
        """Remove events older than window_seconds. Must hold lock."""
        cutoff = time.monotonic() - self._config.window_seconds
        while self._events and self._events[0].timestamp < cutoff:
            self._events.popleft()


# ── Temporal anomaly detector ─────────────────────────────────────────────────


class TemporalAnomalyDetector:
    """
    Detects temporal threat patterns across sequences of AI agent actions.

    Maintains a per-agent sliding window of recent events and
    analyses them after each new event to detect patterns that
    are invisible when events are evaluated in isolation.

    Integration with the analysis pipeline:
        The temporal detector is called AFTER the base engine
        produces its result. If the temporal detector raises an
        alert, the alert is logged and returned to the caller.
        The base engine decision is never reversed by temporal
        analysis — temporal alerts are additive signals.

    Usage:
        detector = TemporalAnomalyDetector()

        # After each engine.analyse() call:
        alerts = detector.update(engine_result)
        for alert in alerts:
            print(alert)
            audit_logger.log("temporal_alert", alert.agent_id, {...})

    Thread safety:
        Safe for concurrent use across multiple agents.
        Each agent has its own window with its own lock.
        The detector's agent registry is also lock-protected.
    """

    def __init__(self, config: TemporalConfig = DEFAULT_CONFIG) -> None:
        self._config: TemporalConfig = config
        self._windows: dict[str, AgentWindow] = {}
        self._lock: threading.Lock = threading.Lock()

    def update(self, result: EngineResult) -> list[TemporalAlert]:
        """
        Update the agent's window with a new event and check for threats.

        This is the primary entry point. Call it after every
        engine.analyse() call to maintain temporal state.

        Args:
            result: The EngineResult from engine.analyse().

        Returns:
            List of TemporalAlert objects. Empty if no threats detected.
            Never raises — temporal failures do not affect base decisions.
        """
        try:
            return self._do_update(result)
        except Exception as exc:
            # Temporal analysis must never crash the main pipeline
            log.error(
                "temporal_detector_error",
                exc_type=type(exc).__name__,
                detail=str(exc)[:200],
                agent_id=result.event.agent_id,
            )
            return []

    def get_window(self, agent_id: str) -> list[WindowEvent]:
        """
        Return the current window for an agent.

        Args:
            agent_id: The agent to query.

        Returns:
            List of WindowEvent objects in chronological order.
            Empty list if the agent has no history.
        """
        window = self._get_or_create_window(agent_id)
        return window.get_recent()

    def window_size(self, agent_id: str) -> int:
        """Return the current window size for an agent."""
        window = self._get_or_create_window(agent_id)
        return window.size()

    def reset_agent(self, agent_id: str) -> None:
        """
        Clear the window for a specific agent.

        Used when an agent is restarted or re-authenticated.
        """
        window = self._get_or_create_window(agent_id)
        window.clear()
        log.info("temporal_window_reset", agent_id=agent_id)

    def reset_all(self) -> None:
        """Clear all agent windows."""
        with self._lock:
            for window in self._windows.values():
                window.clear()
        log.info("temporal_all_windows_reset")

    # ── Private methods ───────────────────────────────────────────────────────

    def _do_update(self, result: EngineResult) -> list[TemporalAlert]:
        """Core update logic — always returns a list, never raises."""
        agent_id = result.event.agent_id

        # Build window event from engine result
        amount = result.event.raw_payload.get("amount", 0.0)
        try:
            amount = float(amount)
        except (TypeError, ValueError):
            amount = 0.0

        window_event = WindowEvent(
            action_type=result.event.action_type,
            risk_score=result.risk_score,
            was_blocked=result.blocked,
            amount=max(0.0, amount),
            timestamp=time.monotonic(),
        )

        # Add to window
        window = self._get_or_create_window(agent_id)
        window.add(window_event)

        # Analyse for threats
        events = window.get_recent()
        alerts = []

        alerts.extend(self._check_burst_attack(agent_id, events))
        alerts.extend(self._check_threshold_probing(agent_id, events))
        alerts.extend(self._check_escalating_risk(agent_id, events))
        alerts.extend(self._check_cumulative_exposure(agent_id, events))
        alerts.extend(self._check_repeated_block_evasion(agent_id, events))

        for alert in alerts:
            log.warning(
                "temporal_threat_detected",
                agent_id=agent_id,
                threat=alert.threat.name,
                severity=alert.severity,
                description=alert.description,
                evidence=alert.evidence,
            )

        return alerts

    def _get_or_create_window(self, agent_id: str) -> AgentWindow:
        """Get or create the sliding window for an agent."""
        with self._lock:
            if agent_id not in self._windows:
                self._windows[agent_id] = AgentWindow(self._config)
            return self._windows[agent_id]

    def _check_burst_attack(
        self,
        agent_id: str,
        events: list[WindowEvent],
    ) -> list[TemporalAlert]:
        """
        Detect BURST_ATTACK: too many actions in the time window.

        Trigger: event count > burst_threshold.
        """
        count = len(events)
        if count > self._config.burst_threshold:
            return [
                TemporalAlert(
                    agent_id=agent_id,
                    threat=TemporalThreat.BURST_ATTACK,
                    severity="HIGH",
                    description=(
                        f"Agent submitted {count} actions in "
                        f"{self._config.window_seconds:.0f}s window "
                        f"(threshold: {self._config.burst_threshold}). "
                        "Possible automated attack bypassing human oversight."
                    ),
                    evidence={
                        "event_count": count,
                        "threshold": self._config.burst_threshold,
                        "window_seconds": self._config.window_seconds,
                    },
                )
            ]
        return []

    def _check_threshold_probing(
        self,
        agent_id: str,
        events: list[WindowEvent],
    ) -> list[TemporalAlert]:
        """
        Detect THRESHOLD_PROBING: repeated actions near the block threshold.

        Trigger: > probe_threshold actions with score in [probe_score_min, probe_score_max].
        """
        probe_events = [
            e
            for e in events
            if self._config.probe_score_min
            <= e.risk_score
            <= self._config.probe_score_max
        ]
        count = len(probe_events)
        if count > self._config.probe_threshold:
            avg_score = sum(e.risk_score for e in probe_events) / count
            return [
                TemporalAlert(
                    agent_id=agent_id,
                    threat=TemporalThreat.THRESHOLD_PROBING,
                    severity="HIGH",
                    description=(
                        f"Agent has {count} actions scoring between "
                        f"{self._config.probe_score_min:.2f} and "
                        f"{self._config.probe_score_max:.2f} "
                        f"(average: {avg_score:.3f}). "
                        "Possible systematic detection boundary exploration."
                    ),
                    evidence={
                        "probe_event_count": count,
                        "avg_probe_score": round(avg_score, 4),
                        "probe_range_min": self._config.probe_score_min,
                        "probe_range_max": self._config.probe_score_max,
                        "threshold": self._config.probe_threshold,
                    },
                )
            ]
        return []

    def _check_escalating_risk(
        self,
        agent_id: str,
        events: list[WindowEvent],
    ) -> list[TemporalAlert]:
        """
        Detect ESCALATING_RISK: monotonically increasing risk scores.

        Compares average risk in the first half of the window
        against average risk in the second half.
        Trigger: second half average > first half average + escalation_delta.
        """
        if len(events) < self._config.escalation_min_events:
            return []

        mid = len(events) // 2
        first_half = events[:mid]
        second_half = events[mid:]

        avg_first = sum(e.risk_score for e in first_half) / len(first_half)
        avg_second = sum(e.risk_score for e in second_half) / len(second_half)
        delta = avg_second - avg_first

        if delta >= self._config.escalation_delta:
            return [
                TemporalAlert(
                    agent_id=agent_id,
                    threat=TemporalThreat.ESCALATING_RISK,
                    severity="CRITICAL",
                    description=(
                        f"Agent risk score increased by {delta:.3f} "
                        f"({avg_first:.3f} → {avg_second:.3f}) "
                        f"over {len(events)} events. "
                        "Possible progressive boundary testing before major attack."
                    ),
                    evidence={
                        "avg_risk_first_half": round(avg_first, 4),
                        "avg_risk_second_half": round(avg_second, 4),
                        "risk_delta": round(delta, 4),
                        "escalation_threshold": self._config.escalation_delta,
                        "event_count": len(events),
                    },
                )
            ]
        return []

    def _check_cumulative_exposure(
        self,
        agent_id: str,
        events: list[WindowEvent],
    ) -> list[TemporalAlert]:
        """
        Detect CUMULATIVE_EXPOSURE: total financial exposure exceeds threshold.

        Trigger: sum of all amounts in window > cumulative_amount_threshold.
        This catches agents making many small trades that individually
        pass the $1M rule but collectively represent massive exposure.
        """
        total_amount = sum(e.amount for e in events if e.amount > 0)

        if total_amount > self._config.cumulative_amount_threshold:
            trade_count = sum(1 for e in events if e.amount > 0)
            return [
                TemporalAlert(
                    agent_id=agent_id,
                    threat=TemporalThreat.CUMULATIVE_EXPOSURE,
                    severity="CRITICAL",
                    description=(
                        f"Agent's cumulative financial exposure in window: "
                        f"${total_amount:,.0f} across {trade_count} trades "
                        f"(threshold: ${self._config.cumulative_amount_threshold:,.0f}). "
                        "Individual trades may be below block threshold but "
                        "cumulative exposure is unacceptable."
                    ),
                    evidence={
                        "total_amount": round(total_amount, 2),
                        "trade_count": trade_count,
                        "threshold": self._config.cumulative_amount_threshold,
                        "avg_per_trade": round(
                            total_amount / trade_count if trade_count > 0 else 0, 2
                        ),
                    },
                )
            ]
        return []

    def _check_repeated_block_evasion(
        self,
        agent_id: str,
        events: list[WindowEvent],
    ) -> list[TemporalAlert]:
        """
        Detect REPEATED_BLOCK_EVASION: agent retries after being blocked.

        Trigger: after a BLOCK event, the same action type appears
        more than evasion_threshold times.
        """
        blocked_types: set[str] = {e.action_type for e in events if e.was_blocked}
        if not blocked_types:
            return []

        alerts = []
        for blocked_type in blocked_types:
            # Count subsequent attempts of the same type
            retry_count = sum(
                1 for e in events if e.action_type == blocked_type and not e.was_blocked
            )
            if retry_count > self._config.evasion_threshold:
                alerts.append(
                    TemporalAlert(
                        agent_id=agent_id,
                        threat=TemporalThreat.REPEATED_BLOCK_EVASION,
                        severity="CRITICAL",
                        description=(
                            f"Agent retried blocked action '{blocked_type}' "
                            f"{retry_count} times with variations "
                            f"(threshold: {self._config.evasion_threshold}). "
                            "Possible systematic evasion of security controls."
                        ),
                        evidence={
                            "blocked_action_type": blocked_type,
                            "retry_count": retry_count,
                            "evasion_threshold": self._config.evasion_threshold,
                        },
                    )
                )
        return alerts
