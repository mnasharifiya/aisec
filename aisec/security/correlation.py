"""
AISec Multi-Agent Correlation Detector.

Detects coordinated attacks across multiple AI agents that individual
per-agent monitoring cannot catch.

Core threat:
    Multiple agents divide a dangerous objective into individually safe
    subtasks. Each agent remains below local thresholds, but together
    they produce a harmful system-level outcome.

Threat patterns detected:
    - COORDINATED_THRESHOLD_EVASION
    - SYNCHRONIZED_ACTION_BURST
    - CROSS_AGENT_ESCALATION
    - SHARED_TARGET_ATTACK

Security design:
    - Non-blocking by default:
        This detector does not directly override the base engine decision.
        It emits correlation alerts that can be consumed by SOC queues,
        audit logs, metrics, webhooks, or future enforcement policies.

    - Fail-safe for the base pipeline:
        Detector failures never crash AnalysisEngine. Errors are logged and
        an empty alert list is returned.

    - Privacy and log safety:
        Agent IDs, action types, and targets are sanitised and bounded before
        storage/logging to reduce log-injection and memory-abuse risk.

    - Thread-safe:
        Shared event store and deduplication cache are protected by one lock.

    - Bounded evidence:
        Alert evidence is capped to avoid huge SOC/webhook/audit payloads.

    - Research-ready:
        Alerts include severity, recommended response, correlation score,
        structured evidence, deterministic timestamp support, and stable
        fingerprints for repeatable evaluation.

Important limitation:
    This detector identifies correlation patterns. It does not prove intent,
    collusion, or communication between agents. Alerts should be interpreted
    as security signals requiring review or policy-based escalation.
"""

from __future__ import annotations

import hashlib
import re
import threading
import time
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from aisec.utils.logger import get_logger

log = get_logger("aisec.security.correlation")


# ── Enums ─────────────────────────────────────────────────────────────────────


class CorrelationThreat(str, Enum):
    """Types of multi-agent coordinated threats."""

    COORDINATED_THRESHOLD_EVASION = "coordinated_threshold_evasion"
    SYNCHRONIZED_ACTION_BURST = "synchronized_action_burst"
    CROSS_AGENT_ESCALATION = "cross_agent_escalation"
    SHARED_TARGET_ATTACK = "shared_target_attack"


class CorrelationSeverity(str, Enum):
    """Severity levels for correlation alerts."""

    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class CorrelationAction(str, Enum):
    """
    Recommended downstream response.

    The detector itself does not enforce this action in v1/v2. Consumers such
    as the AnalysisEngine, SOC queue, webhook dispatcher, or policy engine can
    decide how to use it.
    """

    WATCH = "WATCH"
    REVIEW = "REVIEW"
    BLOCK = "BLOCK"
    SAFE_STATE = "SAFE_STATE"


# ── Configuration ─────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class CorrelationConfig:
    """
    Configuration for multi-agent correlation detection.

    window_seconds:
        Sliding window for general correlation analysis.

    coordinated_amount_threshold:
        Total financial amount across all agents in the window that triggers
        COORDINATED_THRESHOLD_EVASION.

    sync_burst_threshold:
        Number of distinct agents submitting the same action type within
        sync_window_seconds that triggers SYNCHRONIZED_ACTION_BURST.

    sync_window_seconds:
        Time window for synchronized burst detection.

    escalation_window_seconds:
        After Agent A is blocked, how long to watch for Agent B attempting
        the same action.

    shared_target_agent_threshold:
        Number of distinct agents targeting the same non-generic target before
        SHARED_TARGET_ATTACK is raised.

    min_agents_for_correlation:
        Minimum number of distinct agents required before correlation analysis.

    max_events_tracked:
        Hard cap on event store size.

    max_agents_tracked:
        Hard cap on distinct agents retained in memory.

    alert_cooldown_seconds:
        Time during which duplicate alerts with the same fingerprint are
        suppressed.

    max_agents_in_evidence:
        Maximum number of agents included in alert evidence.
    """

    window_seconds: float = 300.0
    coordinated_amount_threshold: float = 5_000_000.0
    sync_burst_threshold: int = 3
    sync_window_seconds: float = 30.0
    escalation_window_seconds: float = 120.0
    shared_target_agent_threshold: int = 3
    min_agents_for_correlation: int = 2
    max_events_tracked: int = 100_000
    max_agents_tracked: int = 1_000
    alert_cooldown_seconds: float = 60.0
    max_agents_in_evidence: int = 20


DEFAULT_CORRELATION_CONFIG = CorrelationConfig()


# ── Data models ───────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class CrossAgentEvent:
    """A single event record in the cross-agent correlation store."""

    agent_id: str
    action_type: str
    risk_score: float
    was_blocked: bool
    amount: float
    target: str
    timestamp: float


@dataclass(frozen=True)
class CorrelationAlert:
    """Alert raised when a multi-agent coordinated threat is detected."""

    threat: CorrelationThreat
    severity: CorrelationSeverity
    recommended_action: CorrelationAction
    correlation_score: float
    agents: list[str]
    description: str
    evidence: dict[str, Any]
    fingerprint: str
    timestamp: float = field(default_factory=time.monotonic)

    def __str__(self) -> str:
        return (
            f"[{self.severity.value}] {self.threat.value} "
            f"score={self.correlation_score:.2f} "
            f"action={self.recommended_action.value} "
            f"agents={self.agents}: {self.description}"
        )


# ── Sanitisation helpers ──────────────────────────────────────────────────────

_LABEL_SAFE_RE = re.compile(r"[^A-Za-z0-9_.:-]")


def _sanitise_label(
    value: Any,
    *,
    max_len: int = 128,
    fallback: str = "unknown",
) -> str:
    """
    Sanitise identifiers before storing/logging them.

    Allows alphanumeric characters plus underscore, dot, colon, and hyphen.
    Removes control/log-injection characters and bounds length.
    """
    try:
        text = str(value)
    except Exception:
        text = fallback

    text = text.replace("\r", "").replace("\n", "").replace("\t", "")
    text = _LABEL_SAFE_RE.sub("", text)
    text = text[:max_len]

    return text if text else fallback


def _clamp_score(score: Any) -> float:
    """Return a risk/correlation score in [0.0, 1.0]."""
    try:
        value = float(score)
    except (TypeError, ValueError):
        return 0.0
    return max(0.0, min(1.0, value))


def _hash_fingerprint(parts: list[str]) -> str:
    """Create a short stable fingerprint for alert deduplication."""
    joined = "|".join(parts)
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()[:24]


# ── Correlation detector ──────────────────────────────────────────────────────


class MultiAgentCorrelationDetector:
    """
    Detects coordinated attacks across multiple AI agents.

    Maintains a shared sliding window of events across all agents and analyses
    cross-agent patterns after each new event.

    Integration:
        Called after AnalysisEngine.analyse() or equivalent.
        Alerts are logged and returned to the caller.
        This detector does not directly mutate the base decision.

    Example:
        detector = MultiAgentCorrelationDetector()

        alerts = detector.update(
            agent_id="bot_a",
            action_type="execute_trade",
            risk_score=0.45,
            was_blocked=False,
            amount=999_999,
            target="NYSE",
        )
    """

    def __init__(
        self,
        config: CorrelationConfig = DEFAULT_CORRELATION_CONFIG,
    ) -> None:
        self._config = config
        self._events: list[CrossAgentEvent] = []
        self._recent_alerts: dict[str, float] = {}
        self._last_timestamp: float | None = None
        self._lock = threading.RLock()

        log.info(
            "multi_agent_correlation_detector_initialized",
            window_seconds=config.window_seconds,
            coordinated_amount_threshold=config.coordinated_amount_threshold,
            sync_burst_threshold=config.sync_burst_threshold,
            max_events_tracked=config.max_events_tracked,
        )

    def update(
        self,
        agent_id: str,
        action_type: str,
        risk_score: float,
        was_blocked: bool,
        amount: float = 0.0,
        target: str = "",
        timestamp: float | None = None,
    ) -> list[CorrelationAlert]:
        """
        Record a new event and check for cross-agent threats.

        Args:
            agent_id:
                Agent that submitted the action.

            action_type:
                Action type attempted.

            risk_score:
                Risk score assigned by the engine.

            was_blocked:
                True if the base engine blocked this action.

            amount:
                Financial amount or exposure value. Use 0 if not applicable.

            target:
                Target resource or system.

            timestamp:
                Optional monotonic timestamp. Useful for deterministic tests.

        Returns:
            List of CorrelationAlert objects. Empty if no threats.
            Never raises.
        """
        try:
            return self._do_update(
                agent_id=agent_id,
                action_type=action_type,
                risk_score=risk_score,
                was_blocked=was_blocked,
                amount=amount,
                target=target,
                timestamp=timestamp,
            )
        except Exception as exc:
            log.error(
                "correlation_detector_error",
                exc_type=type(exc).__name__,
                detail=str(exc)[:200],
            )
            return []

    def active_agent_count(self) -> int:
        """Return number of distinct agents in the current window."""
        with self._lock:
            now = self._now_for_housekeeping()
            self._expire(now)
            return len({event.agent_id for event in self._events})

    def event_count(self) -> int:
        """Return total events in the current window."""
        with self._lock:
            now = self._now_for_housekeeping()
            self._expire(now)
            return len(self._events)

    def reset(self) -> None:
        """Clear all events and alert deduplication state."""
        with self._lock:
            self._events.clear()
            self._recent_alerts.clear()
            self._last_timestamp = None

        log.warning("correlation_detector_reset")

    # ── Private update path ───────────────────────────────────────────────────

    def _do_update(
        self,
        *,
        agent_id: str,
        action_type: str,
        risk_score: float,
        was_blocked: bool,
        amount: float,
        target: str,
        timestamp: float | None,
    ) -> list[CorrelationAlert]:
        now = timestamp if timestamp is not None else time.monotonic()
        self._last_timestamp = float(now)

        event = CrossAgentEvent(
            agent_id=_sanitise_label(
                agent_id,
                max_len=128,
                fallback="unknown_agent",
            ),
            action_type=_sanitise_label(
                action_type,
                max_len=128,
                fallback="unknown_action",
            ),
            risk_score=_clamp_score(risk_score),
            was_blocked=bool(was_blocked),
            amount=self._normalise_amount(amount),
            target=_sanitise_label(target, max_len=128, fallback=""),
            timestamp=float(now),
        )

        with self._lock:
            self._events.append(event)
            self._expire(now)
            self._enforce_bounds()

            active_agents = len({stored.agent_id for stored in self._events})
            if active_agents < self._config.min_agents_for_correlation:
                return []

            current_events = list(self._events)

        alerts: list[CorrelationAlert] = []
        alerts.extend(self._check_coordinated_amount(current_events, now))
        alerts.extend(
            self._check_synchronized_burst(
                current_events,
                event.action_type,
                now,
            )
        )
        alerts.extend(
            self._check_cross_agent_escalation(
                current_events,
                event,
                now,
            )
        )
        alerts.extend(self._check_shared_target(current_events, now))

        alerts = self._deduplicate_alerts(alerts, now)

        for alert in alerts:
            log.warning(
                "multi_agent_correlation_alert",
                threat=alert.threat.value,
                severity=alert.severity.value,
                recommended_action=alert.recommended_action.value,
                correlation_score=alert.correlation_score,
                agents=alert.agents,
                description=alert.description,
                fingerprint=alert.fingerprint,
            )

        return alerts

    def _now_for_housekeeping(self) -> float:
        """
        Return timestamp used for housekeeping operations.

        In production this normally tracks time.monotonic(). In deterministic
        tests, update() may receive artificial timestamps such as 1000.0, so
        count/expiry helpers must use the last observed event timestamp instead
        of real monotonic time.
        """
        return (
            self._last_timestamp
            if self._last_timestamp is not None
            else time.monotonic()
        )

    @staticmethod
    def _normalise_amount(amount: Any) -> float:
        try:
            return max(0.0, float(amount))
        except (TypeError, ValueError):
            return 0.0

    def _expire(self, now: float) -> None:
        """Remove expired events and old deduplication entries. Must hold lock."""
        cutoff = now - self._config.window_seconds
        self._events = [event for event in self._events if event.timestamp >= cutoff]

        alert_cutoff = now - self._config.alert_cooldown_seconds
        self._recent_alerts = {
            fingerprint: ts
            for fingerprint, ts in self._recent_alerts.items()
            if ts >= alert_cutoff
        }

    def _enforce_bounds(self) -> None:
        """Bound memory use. Must hold lock."""
        if len(self._events) > self._config.max_events_tracked:
            self._events = self._events[-self._config.max_events_tracked :]

        agents_seen: set[str] = set()
        bounded: list[CrossAgentEvent] = []

        for event in reversed(self._events):
            agents_seen.add(event.agent_id)
            if len(agents_seen) <= self._config.max_agents_tracked:
                bounded.append(event)

        self._events = list(reversed(bounded))

    # ── Detection checks ──────────────────────────────────────────────────────

    def _check_coordinated_amount(
        self,
        events: list[CrossAgentEvent],
        now: float,
    ) -> list[CorrelationAlert]:
        """
        Detect coordinated threshold evasion via amount splitting.

        Multiple agents each staying below per-agent thresholds may together
        exceed the coordinated system-level exposure threshold.
        """
        financial_events = [event for event in events if event.amount > 0]
        if not financial_events:
            return []

        agent_amounts: dict[str, float] = defaultdict(float)
        for event in financial_events:
            agent_amounts[event.agent_id] += event.amount

        if len(agent_amounts) < self._config.min_agents_for_correlation:
            return []

        total = sum(agent_amounts.values())
        threshold = self._config.coordinated_amount_threshold

        if total <= threshold:
            return []

        agents = sorted(agent_amounts.keys())
        top_agents = self._top_amount_agents(agent_amounts)

        score = min(1.0, total / max(threshold * 2.0, 1.0))

        # Deduplication must not include the changing exact total amount.
        # Otherwise the same coordinated attack produces a new alert whenever
        # the total changes slightly.
        fingerprint = _hash_fingerprint(
            [
                CorrelationThreat.COORDINATED_THRESHOLD_EVASION.value,
                ",".join(agents),
                str(round(threshold, 2)),
            ]
        )

        return [
            CorrelationAlert(
                threat=CorrelationThreat.COORDINATED_THRESHOLD_EVASION,
                severity=CorrelationSeverity.CRITICAL,
                recommended_action=CorrelationAction.REVIEW,
                correlation_score=round(score, 4),
                agents=self._bounded_agents(agents),
                description=(
                    f"Combined financial exposure across {len(agents)} agents "
                    f"is ${total:,.0f}, exceeding the configured threshold "
                    f"of ${threshold:,.0f}. Possible coordinated threshold "
                    f"evasion."
                ),
                evidence={
                    "total_amount": round(total, 2),
                    "threshold": threshold,
                    "agent_count": len(agents),
                    "top_agents": top_agents,
                    "evidence_bounded": len(agents) > len(top_agents),
                },
                fingerprint=fingerprint,
                timestamp=now,
            )
        ]

    def _check_synchronized_burst(
        self,
        events: list[CrossAgentEvent],
        action_type: str,
        now: float,
    ) -> list[CorrelationAlert]:
        """
        Detect multiple agents submitting the same action in a short window.
        """
        cutoff = now - self._config.sync_window_seconds
        recent = [
            event
            for event in events
            if event.action_type == action_type and event.timestamp >= cutoff
        ]

        unique_agents = sorted({event.agent_id for event in recent})
        if len(unique_agents) < self._config.sync_burst_threshold:
            return []

        score = min(
            1.0,
            len(unique_agents)
            / max(
                self._config.sync_burst_threshold * 2.0,
                1.0,
            ),
        )
        fingerprint = _hash_fingerprint(
            [
                CorrelationThreat.SYNCHRONIZED_ACTION_BURST.value,
                action_type,
                ",".join(unique_agents),
            ]
        )

        return [
            CorrelationAlert(
                threat=CorrelationThreat.SYNCHRONIZED_ACTION_BURST,
                severity=CorrelationSeverity.HIGH,
                recommended_action=CorrelationAction.REVIEW,
                correlation_score=round(score, 4),
                agents=self._bounded_agents(unique_agents),
                description=(
                    f"{len(unique_agents)} agents submitted action "
                    f"'{action_type}' within "
                    f"{self._config.sync_window_seconds:.0f}s. "
                    f"Possible synchronized action burst."
                ),
                evidence={
                    "action_type": action_type,
                    "agent_count": len(unique_agents),
                    "window_seconds": self._config.sync_window_seconds,
                    "threshold": self._config.sync_burst_threshold,
                    "evidence_bounded": (
                        len(unique_agents) > self._config.max_agents_in_evidence
                    ),
                },
                fingerprint=fingerprint,
                timestamp=now,
            )
        ]

    def _check_cross_agent_escalation(
        self,
        events: list[CrossAgentEvent],
        current_event: CrossAgentEvent,
        now: float,
    ) -> list[CorrelationAlert]:
        """
        Detect Agent B attempting an action recently blocked for Agent A.
        """
        if current_event.was_blocked:
            return []

        cutoff = now - self._config.escalation_window_seconds
        prior_blocks = [
            event
            for event in events
            if event.was_blocked
            and event.action_type == current_event.action_type
            and event.agent_id != current_event.agent_id
            and event.timestamp >= cutoff
        ]

        if not prior_blocks:
            return []

        blocking_agents = sorted({event.agent_id for event in prior_blocks})
        all_agents = sorted(set(blocking_agents + [current_event.agent_id]))

        max_prior_risk = max(
            (event.risk_score for event in prior_blocks),
            default=0.0,
        )
        score = max(
            0.70,
            min(
                1.0,
                (max_prior_risk + current_event.risk_score) / 2.0 + 0.20,
            ),
        )

        fingerprint = _hash_fingerprint(
            [
                CorrelationThreat.CROSS_AGENT_ESCALATION.value,
                current_event.action_type,
                current_event.agent_id,
                ",".join(blocking_agents),
            ]
        )

        return [
            CorrelationAlert(
                threat=CorrelationThreat.CROSS_AGENT_ESCALATION,
                severity=CorrelationSeverity.HIGH,
                recommended_action=CorrelationAction.REVIEW,
                correlation_score=round(score, 4),
                agents=self._bounded_agents(all_agents),
                description=(
                    f"Agent '{current_event.agent_id}' attempted action "
                    f"'{current_event.action_type}' within "
                    f"{self._config.escalation_window_seconds:.0f}s of "
                    f"the same action being blocked for {blocking_agents}. "
                    f"Possible cross-agent handoff."
                ),
                evidence={
                    "action_type": current_event.action_type,
                    "new_agent": current_event.agent_id,
                    "blocked_agents": blocking_agents[
                        : self._config.max_agents_in_evidence
                    ],
                    "blocked_agent_count": len(blocking_agents),
                    "window_seconds": self._config.escalation_window_seconds,
                    "current_risk_score": current_event.risk_score,
                    "max_prior_blocked_risk": max_prior_risk,
                },
                fingerprint=fingerprint,
                timestamp=now,
            )
        ]

    def _check_shared_target(
        self,
        events: list[CrossAgentEvent],
        now: float,
    ) -> list[CorrelationAlert]:
        """
        Detect multiple agents targeting the same sensitive resource.
        """
        generic_targets = {
            "",
            "MARKET",
            "NYSE",
            "NASDAQ",
            "CITYSYSTEM",
            "CITY_SYSTEM",
            "city_system",
            "unknown",
            "unknown_target",
        }

        target_agents: dict[str, set[str]] = defaultdict(set)
        target_risks: dict[str, list[float]] = defaultdict(list)

        for event in events:
            if event.target and event.target not in generic_targets:
                target_agents[event.target].add(event.agent_id)
                target_risks[event.target].append(event.risk_score)

        alerts: list[CorrelationAlert] = []

        for target, agents_set in target_agents.items():
            agents = sorted(agents_set)
            if len(agents) < self._config.shared_target_agent_threshold:
                continue

            avg_risk = sum(target_risks[target]) / max(
                len(target_risks[target]),
                1,
            )
            score = min(1.0, max(0.55, avg_risk + 0.20))

            fingerprint = _hash_fingerprint(
                [
                    CorrelationThreat.SHARED_TARGET_ATTACK.value,
                    target,
                    ",".join(agents),
                ]
            )

            alerts.append(
                CorrelationAlert(
                    threat=CorrelationThreat.SHARED_TARGET_ATTACK,
                    severity=CorrelationSeverity.HIGH,
                    recommended_action=CorrelationAction.REVIEW,
                    correlation_score=round(score, 4),
                    agents=self._bounded_agents(agents),
                    description=(
                        f"{len(agents)} agents targeted '{target}' within "
                        f"the correlation window. Possible coordinated "
                        f"targeting or distributed reconnaissance."
                    ),
                    evidence={
                        "target": target,
                        "agent_count": len(agents),
                        "agents_sample": self._bounded_agents(agents),
                        "average_risk_score": round(avg_risk, 4),
                        "threshold": self._config.shared_target_agent_threshold,
                        "evidence_bounded": (
                            len(agents) > self._config.max_agents_in_evidence
                        ),
                    },
                    fingerprint=fingerprint,
                    timestamp=now,
                )
            )

        return alerts

    # ── Alert handling ────────────────────────────────────────────────────────

    def _deduplicate_alerts(
        self,
        alerts: list[CorrelationAlert],
        now: float,
    ) -> list[CorrelationAlert]:
        """Suppress duplicate alerts inside alert_cooldown_seconds."""
        if not alerts:
            return []

        deduped: list[CorrelationAlert] = []

        with self._lock:
            cutoff = now - self._config.alert_cooldown_seconds
            self._recent_alerts = {
                fingerprint: ts
                for fingerprint, ts in self._recent_alerts.items()
                if ts >= cutoff
            }

            for alert in alerts:
                last_seen = self._recent_alerts.get(alert.fingerprint)
                if (
                    last_seen is not None
                    and now - last_seen < self._config.alert_cooldown_seconds
                ):
                    continue

                self._recent_alerts[alert.fingerprint] = now
                deduped.append(alert)

        return deduped

    def _bounded_agents(self, agents: list[str]) -> list[str]:
        return agents[: self._config.max_agents_in_evidence]

    def _top_amount_agents(self, agent_amounts: dict[str, float]) -> dict[str, float]:
        top_items = sorted(
            agent_amounts.items(),
            key=lambda item: item[1],
            reverse=True,
        )[: self._config.max_agents_in_evidence]

        return {agent: round(amount, 2) for agent, amount in top_items}
