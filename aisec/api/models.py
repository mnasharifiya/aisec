"""
AISec REST API request and response models.

Uses Pydantic for validation — every field is typed,
validated, and documented. Invalid requests are rejected
with clear error messages before they reach the engine.

Security:
    - All string fields have maximum length limits.
    - Numeric fields have range constraints.
    - Enum fields reject unknown values.
    - No model accepts arbitrary extra fields.
"""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, field_validator

# ── Request models ────────────────────────────────────────────────────────────


class ScenarioRequest(str, Enum):
    """Valid scenario values for API requests."""

    TRADING_AI = "trading_ai"
    URBAN_AI = "urban_ai"
    UNKNOWN = "unknown"


class AnalyseRequest(BaseModel):
    """
    Request body for POST /api/v1/analyse.

    Represents a single AI agent action to be evaluated.
    All fields are validated before reaching the engine.
    """

    model_config = {"extra": "forbid"}  # Reject unknown fields

    action_type: str = Field(
        ...,
        min_length=1,
        max_length=128,
        description="The action the AI agent is attempting.",
        examples=["execute_trade", "read_market_data"],
    )
    agent_id: str = Field(
        ...,
        min_length=3,
        max_length=64,
        description="Unique identifier of the AI agent.",
        examples=["trading_bot_v1", "urban_ctrl_prod"],
    )
    target: str = Field(
        ...,
        min_length=1,
        max_length=128,
        description="The resource or system the action targets.",
        examples=["NYSE", "power_grid_north"],
    )
    scenario: ScenarioRequest = Field(
        default=ScenarioRequest.UNKNOWN,
        description="Threat scenario — selects the rule set.",
    )
    payload: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional action parameters for risk scoring.",
        examples=[{"amount": 500000, "symbol": "AAPL"}],
    )

    @field_validator("action_type", "agent_id", "target")
    @classmethod
    def sanitise_string(cls, v: str) -> str:
        """Strip leading/trailing whitespace from string fields."""
        return v.strip()

    @field_validator("payload")
    @classmethod
    def limit_payload_size(cls, v: dict) -> dict:
        """Reject payloads with too many keys."""
        if len(v) > 50:
            raise ValueError(f"payload has {len(v)} keys — maximum is 50.")
        return v


class BatchAnalyseRequest(BaseModel):
    """Request body for POST /api/v1/analyse/batch."""

    model_config = {"extra": "forbid"}

    events: list[AnalyseRequest] = Field(
        ...,
        min_length=1,
        max_length=100,
        description="List of events to analyse. Maximum 100 per batch.",
    )


class ResolveRequest(BaseModel):
    """Request body for POST /api/v1/queue/resolve."""

    model_config = {"extra": "forbid"}

    event_id: str = Field(..., min_length=1, max_length=64)
    decision: str = Field(..., pattern="^(approve|block|escalate)$")
    analyst_id: str = Field(..., min_length=3, max_length=64)
    reason: str = Field(default="", max_length=512)


# ── Response models ───────────────────────────────────────────────────────────


class DecisionResponse(str, Enum):
    """Decision values in API responses."""

    ALLOW = "ALLOW"
    BLOCK = "BLOCK"
    ESCALATE = "ESCALATE"
    PENDING_REVIEW = "PENDING_REVIEW"


class AnalyseResponse(BaseModel):
    """Response body for POST /api/v1/analyse."""

    event_id: str
    agent_id: str
    action_type: str
    decision: DecisionResponse
    risk_score: float
    rule_hits: list[str]
    explanation: str
    log_entry_id: str
    blocked: bool
    requires_review: bool
    temporal_alerts: list[dict[str, Any]] = []


class BatchAnalyseResponse(BaseModel):
    """Response body for POST /api/v1/analyse/batch."""

    total: int
    blocked_count: int
    allowed_count: int
    review_count: int
    results: list[AnalyseResponse]


class HealthResponse(BaseModel):
    """Response body for GET /api/v1/health."""

    status: str  # "healthy" or "degraded"
    version: str
    audit_chain: str  # "intact" or "broken"
    audit_entries: int
    engine: str  # "ready"


class MetricsSummaryResponse(BaseModel):
    """Response body for GET /api/v1/metrics/summary."""

    total_events: int
    blocked_events: int
    allowed_events: int
    review_events: int
    block_rate: float
    avg_risk_score: float
    audit_chain_ok: bool


class AuditVerifyResponse(BaseModel):
    """Response body for GET /api/v1/audit/verify."""

    chain_intact: bool
    entry_count: int
    errors: list[str]


class QueueResponse(BaseModel):
    """Response body for GET /api/v1/queue."""

    pending_count: int
    events: list[dict[str, Any]]


class ErrorResponse(BaseModel):
    """Standard error response."""

    error: str
    detail: str
    code: int
