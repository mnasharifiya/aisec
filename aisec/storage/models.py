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
        # Freeze payload copies to prevent post-construction mutation.
        # This ensures the audit log always reflects what was analysed.
        object.__setattr__(self, "raw_payload", dict(self.raw_payload))
        object.__setattr__(self, "metadata",    dict(self.metadata))


@dataclass
class FeatureVector:
    """Numerical encoding of an Event for risk scoring."""
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
    """Output of the analysis engine for a single Event."""
    event_id:           str
    risk_score:         float
    decision:           Decision
    explanation:        str
    rule_hits:          list[str]       = field(default_factory=list)
    baseline_similarity: float          = 1.0
    risk_delta:         float           = 0.0

    def __post_init__(self) -> None:
        if not 0.0 <= self.risk_score <= 1.0:
            raise ValueError(
                f"risk_score must be in [0.0, 1.0], got {self.risk_score}"
            )
        if not isinstance(self.decision, Decision):
            raise ValueError(f"decision must be a Decision enum, got {type(self.decision)}")


@dataclass
class Alert:
    """A human-reviewable notification raised for a suspicious event."""
    event_id:          str
    severity:          Severity
    reason:            str

    assigned_to:       str | None            = None
    delivery_channels: list[str]             = field(default_factory=lambda: ["cli"])

    alert_id:   str         = field(default_factory=_new_id)
    status:     AlertStatus = AlertStatus.NEW
    created_at: str         = field(default_factory=_now_utc)


@dataclass
class Incident:
    """A grouped collection of related alerts representing a security event."""
    title:             str
    severity:          Severity
    related_event_ids: list[str] = field(default_factory=list)

    resolution: str | None = None

    incident_id: str            = field(default_factory=_new_id)
    status:      IncidentStatus = IncidentStatus.NEW
    created_at:  str            = field(default_factory=_now_utc)
    updated_at:  str            = field(default_factory=_now_utc)

    def update(self, status: IncidentStatus, resolution: str | None = None) -> None:
        self.status     = status
        self.updated_at = _now_utc()
        if resolution:
            self.resolution = resolution


@dataclass
class AuditLogEntry:
    """A single tamper-evident entry in the hash-chained audit log."""
    record_type: str
    record_id:   str
    payload:     dict[str, Any]
    prev_hash:   str

    log_id:       str = field(default_factory=_new_id)
    timestamp:    str = field(default_factory=_now_utc)
    current_hash: str = field(default="")

    def __post_init__(self) -> None:
        if not self.current_hash:
            self.current_hash = self._compute_hash()

    def _compute_hash(self) -> str:
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
        recomputed = self._compute_hash()
        return (
            recomputed == self.current_hash
            and self.prev_hash == expected_prev_hash
        )
