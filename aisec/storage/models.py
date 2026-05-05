"""
AISec data models — the foundation of the entire system.
Every module reads and writes these structures.
Do not change field names without updating all consumers.
"""

from __future__ import annotations

import hashlib
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any


# ── Enumerations ──────────────────────────────────────────────────────────────

class Decision(str, Enum):
    """Final enforcement decision for an AI action."""
    ALLOW           = "ALLOW"
    BLOCK           = "BLOCK"
    PENDING_REVIEW  = "PENDING_REVIEW"
    ESCALATE        = "ESCALATE"


class Severity(str, Enum):
    """Alert and incident severity levels."""
    LOW      = "LOW"
    MEDIUM   = "MEDIUM"
    HIGH     = "HIGH"
    CRITICAL = "CRITICAL"


class AlertStatus(str, Enum):
    """Lifecycle state of a single alert."""
    NEW       = "NEW"
    QUEUED    = "QUEUED"
    ASSIGNED  = "ASSIGNED"
    IN_REVIEW = "IN_REVIEW"
    RESOLVED  = "RESOLVED"
    CLOSED    = "CLOSED"


class IncidentStatus(str, Enum):
    """Lifecycle state of an incident (one or more related alerts)."""
    NEW       = "NEW"
    OPEN      = "OPEN"
    ASSIGNED  = "ASSIGNED"
    ESCALATED = "ESCALATED"
    RESOLVED  = "RESOLVED"
    CLOSED    = "CLOSED"


class Scenario(str, Enum):
    """Built-in threat scenario identifiers."""
    TRADING_AI = "trading_ai"
    URBAN_AI   = "urban_ai"
    UNKNOWN    = "unknown"


# ── Helpers ───────────────────────────────────────────────────────────────────

def _now_utc() -> str:
    """Return the current UTC time as an ISO-8601 string."""
    return datetime.now(timezone.utc).isoformat()


def _new_id() -> str:
    """Generate a new random UUID string."""
    return str(uuid.uuid4())


# ── Core data models ──────────────────────────────────────────────────────────

@dataclass
class Event:
    """
    A single AI agent action captured by the interceptor.

    This is the raw input to the analysis pipeline.
    Every other model is derived from or linked to an Event.
    """
    action_type:   str
    agent_id:      str
    target:        str

    # Optional fields with sensible defaults
    target_type:    str                  = "unknown"
    severity_hint:  str | None           = None
    raw_payload:    dict[str, Any]       = field(default_factory=dict)
    metadata:       dict[str, Any]       = field(default_factory=dict)

    # Auto-generated — do not pass these manually
    event_id:   str = field(default_factory=_new_id)
    session_id: str = field(default_factory=_new_id)
    timestamp:  str = field(default_factory=_now_utc)
    scenario:   Scenario = Scenario.UNKNOWN

    def __post_init__(self) -> None:
        if not self.action_type:
            raise ValueError("action_type cannot be empty")
        if not self.agent_id:
            raise ValueError("agent_id cannot be empty")
        if not self.target:
            raise ValueError("target cannot be empty")


@dataclass
class FeatureVector:
    """
    Numerical encoding of an Event for risk scoring.

    Dimensions (in order):
        0  action_type_encoding   — mapped from action type string
        1  keyword_risk_score     — presence of dangerous keywords
        2  frequency_score        — burst rate of similar actions
        3  api_call_flag          — 1.0 if action calls external API
        4  file_access_flag       — 1.0 if action touches filesystem
        5  network_access_flag    — 1.0 if action uses network
        6  sensitive_path_flag    — 1.0 if target is a sensitive resource
        7  privilege_flag         — 1.0 if action requires elevated rights
    """
    event_id:   str
    vector:     list[float]
    dimensions: list[str] = field(default_factory=lambda: [
        "action_type_encoding",
        "keyword_risk_score",
        "frequency_score",
        "api_call_flag",
        "file_access_flag",
        "network_access_flag",
        "sensitive_path_flag",
        "privilege_flag",
    ])

    EXPECTED_DIMENSIONS: int = 8

    def __post_init__(self) -> None:
        if len(self.vector) != self.EXPECTED_DIMENSIONS:
            raise ValueError(
                f"FeatureVector must have exactly {self.EXPECTED_DIMENSIONS} "
                f"dimensions, got {len(self.vector)}"
            )
        if not all(0.0 <= v <= 1.0 for v in self.vector):
            raise ValueError("All feature values must be in range [0.0, 1.0]")


@dataclass
class AnalysisResult:
    """
    Output of the analysis engine for a single Event.

    Contains the risk score, which rules fired, similarity
    to baseline behavior, and the final enforcement decision.
    """
    event_id:           str
    risk_score:         float           # 0.0 (safe) to 1.0 (critical)
    decision:           Decision
    explanation:        str
    rule_hits:          list[str]       = field(default_factory=list)
    baseline_similarity: float          = 1.0   # 1.0 = normal, 0.0 = anomalous
    risk_delta:         float           = 0.0   # change vs previous event

    def __post_init__(self) -> None:
        if not 0.0 <= self.risk_score <= 1.0:
            raise ValueError(
                f"risk_score must be in [0.0, 1.0], got {self.risk_score}"
            )
        if not isinstance(self.decision, Decision):
            raise ValueError(f"decision must be a Decision enum, got {type(self.decision)}")


@dataclass
class Alert:
    """
    A human-reviewable notification raised for a suspicious event.

    Alerts are the primary unit of work in the SOC queue.
    """
    event_id:          str
    severity:          Severity
    reason:            str

    # Optional
    assigned_to:       str | None            = None
    delivery_channels: list[str]             = field(default_factory=lambda: ["cli"])

    # Auto-generated
    alert_id:   str         = field(default_factory=_new_id)
    status:     AlertStatus = AlertStatus.NEW
    created_at: str         = field(default_factory=_now_utc)


@dataclass
class Incident:
    """
    A grouped collection of related alerts representing a security event.

    Multiple alerts from the same agent or attack pattern
    can be linked to a single incident for unified tracking.
    """
    title:             str
    severity:          Severity
    related_event_ids: list[str] = field(default_factory=list)

    # Optional
    resolution: str | None = None

    # Auto-generated
    incident_id: str            = field(default_factory=_new_id)
    status:      IncidentStatus = IncidentStatus.NEW
    created_at:  str            = field(default_factory=_now_utc)
    updated_at:  str            = field(default_factory=_now_utc)

    def update(self, status: IncidentStatus, resolution: str | None = None) -> None:
        """Update incident status and optionally set resolution."""
        self.status     = status
        self.updated_at = _now_utc()
        if resolution:
            self.resolution = resolution


@dataclass
class AuditLogEntry:
    """
    A single tamper-evident entry in the hash-chained audit log.

    Each entry includes the SHA-256 hash of the previous entry,
    forming a chain. Any modification to any entry breaks the chain
    and is immediately detectable by verify_chain().
    """
    record_type: str            # "event", "decision", "alert", "analyst_action"
    record_id:   str
    payload:     dict[str, Any]
    prev_hash:   str            # SHA-256 of previous entry, "0" for genesis

    # Auto-generated
    log_id:       str = field(default_factory=_new_id)
    timestamp:    str = field(default_factory=_now_utc)
    current_hash: str = field(default="")

    def __post_init__(self) -> None:
        """Compute and store the hash of this entry on creation."""
        if not self.current_hash:
            self.current_hash = self._compute_hash()

    def _compute_hash(self) -> str:
        """
        Compute SHA-256 over the deterministic content of this entry.

        Note: current_hash is excluded from the digest to avoid
        circular dependency.
        """
        content = (
            f"{self.log_id}"
            f"{self.timestamp}"
            f"{self.record_type}"
            f"{self.record_id}"
            f"{self.prev_hash}"
            f"{sorted(self.payload.items())}"
        )
        return hashlib.sha256(content.encode()).hexdigest()

    def verify(self, expected_prev_hash: str) -> bool:
        """
        Verify this entry has not been tampered with.

        Recomputes the hash and checks:
          1. The recomputed hash matches current_hash (content integrity).
          2. prev_hash matches the expected value (chain continuity).
        """
        recomputed = self._compute_hash()
        return (
            recomputed == self.current_hash
            and self.prev_hash == expected_prev_hash
        )