"""
AISec REST API route handlers.

Each route handler:
    1. Validates the request (Pydantic does this automatically)
    2. Calls the appropriate engine/service method
    3. Returns a typed response model
    4. Never exposes internal exceptions to callers

Security:
    - All inputs are validated by Pydantic before handlers run.
    - Engine errors are caught and returned as 500 responses.
    - No stack traces are returned to callers in production.
    - Rate limiting is applied at the API gateway level (v2).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.responses import JSONResponse

from aisec.api.models import (
    AnalyseRequest,
    AnalyseResponse,
    AuditVerifyResponse,
    BatchAnalyseRequest,
    BatchAnalyseResponse,
    DecisionResponse,
    ErrorResponse,
    HealthResponse,
    MetricsSummaryResponse,
    QueueResponse,
    ResolveRequest,
)
from aisec.core.engine import AnalysisEngine
from aisec.storage.models import Decision, Event, Scenario
from aisec.utils.logger import get_logger

log = get_logger("aisec.api.routes")

router = APIRouter()

# ── Engine dependency ─────────────────────────────────────────────────────────


def get_engine(request: Request) -> AnalysisEngine:
    """Dependency injection — returns the shared engine from app state."""
    return request.app.state.engine


# ── Scenario mapping ──────────────────────────────────────────────────────────

_SCENARIO_MAP = {
    "trading_ai": Scenario.TRADING_AI,
    "urban_ai": Scenario.URBAN_AI,
    "unknown": Scenario.UNKNOWN,
}


def _to_scenario(s: str) -> Scenario:
    return _SCENARIO_MAP.get(s, Scenario.UNKNOWN)


def _to_decision_response(d: Decision) -> DecisionResponse:
    return DecisionResponse(d.value)


def _result_to_response(result: Any) -> AnalyseResponse:
    """Convert an EngineResult to an AnalyseResponse."""
    temporal = []
    if hasattr(result, "temporal_alerts") and result.temporal_alerts:
        temporal = [
            {
                "threat": a.threat.name,
                "severity": a.severity,
                "description": a.description,
                "evidence": a.evidence,
            }
            for a in result.temporal_alerts
        ]

    return AnalyseResponse(
        event_id=result.event.event_id,
        agent_id=result.event.agent_id,
        action_type=result.event.action_type,
        decision=_to_decision_response(result.decision),
        risk_score=round(result.risk_score, 4),
        rule_hits=result.analysis.rule_hits,
        explanation=result.analysis.explanation,
        log_entry_id=result.log_entry_id,
        blocked=result.blocked,
        requires_review=result.requires_review,
        temporal_alerts=temporal,
    )


# ── Routes ────────────────────────────────────────────────────────────────────


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Liveness and readiness check",
    tags=["System"],
)
async def health(
    engine: AnalysisEngine = Depends(get_engine),
) -> HealthResponse:
    """
    Returns the health status of the AISec service.

    Used by load balancers, Kubernetes probes, and monitoring systems.
    A "degraded" status means the service is running but the audit
    chain has been broken — which is a critical security event.
    """
    try:
        ok, errors = engine.verify_audit_chain()
        audit_count = engine.audit_count()
        chain_status = "intact" if ok else f"BROKEN ({len(errors)} errors)"
        svc_status = "healthy" if ok else "degraded"
    except Exception as exc:
        log.error("health_check_failed", exc_type=type(exc).__name__)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="AISec engine health check failed",
        )

    return HealthResponse(
        status=svc_status,
        version="1.2.0",
        audit_chain=chain_status,
        audit_entries=audit_count,
        engine="ready",
    )


@router.post(
    "/analyse",
    response_model=AnalyseResponse,
    summary="Analyse a single AI agent action",
    tags=["Analysis"],
    status_code=status.HTTP_200_OK,
)
async def analyse(
    body: AnalyseRequest,
    engine: AnalysisEngine = Depends(get_engine),
) -> AnalyseResponse:
    """
    Analyse a single AI agent action and return an enforcement decision.

    The decision is one of:
    - **ALLOW**: Action is safe to proceed.
    - **BLOCK**: Action is denied immediately.
    - **ESCALATE**: Action requires senior analyst review.
    - **PENDING_REVIEW**: Action held pending human approval.

    Every call is logged to the tamper-evident audit trail.
    """
    try:
        event = Event(
            action_type=body.action_type,
            agent_id=body.agent_id,
            target=body.target,
            scenario=_to_scenario(body.scenario.value),
            raw_payload=body.payload,
        )
        result = engine.analyse(event)

        log.info(
            "api_analyse_complete",
            agent_id=body.agent_id,
            action_type=body.action_type,
            decision=result.decision.value,
            risk_score=result.risk_score,
        )

        return _result_to_response(result)

    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        )
    except Exception as exc:
        log.error(
            "api_analyse_error",
            exc_type=type(exc).__name__,
            detail=str(exc)[:200],
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Analysis engine error. Action blocked by default.",
        )


@router.post(
    "/analyse/batch",
    response_model=BatchAnalyseResponse,
    summary="Analyse multiple AI agent actions",
    tags=["Analysis"],
    status_code=status.HTTP_200_OK,
)
async def analyse_batch(
    body: BatchAnalyseRequest,
    engine: AnalysisEngine = Depends(get_engine),
) -> BatchAnalyseResponse:
    """
    Analyse up to 100 AI agent actions in a single request.

    Each event is analysed independently. A blocked event
    does not prevent analysis of subsequent events.
    All events are logged to the audit trail.
    """
    results = []
    blocked = 0
    allowed = 0
    review = 0

    for event_req in body.events:
        try:
            event = Event(
                action_type=event_req.action_type,
                agent_id=event_req.agent_id,
                target=event_req.target,
                scenario=_to_scenario(event_req.scenario.value),
                raw_payload=event_req.payload,
            )
            result = engine.analyse(event)
            response = _result_to_response(result)
            results.append(response)

            if result.blocked:
                blocked += 1
            elif result.requires_review:
                review += 1
            else:
                allowed += 1

        except Exception as exc:
            log.error(
                "api_batch_event_error",
                exc_type=type(exc).__name__,
                action_type=event_req.action_type,
            )
            # Include a synthetic BLOCK response for failed events
            results.append(
                AnalyseResponse(
                    event_id="error",
                    agent_id=event_req.agent_id,
                    action_type=event_req.action_type,
                    decision=DecisionResponse.BLOCK,
                    risk_score=1.0,
                    rule_hits=["API-ERROR"],
                    explanation="Analysis failed — blocked by default.",
                    log_entry_id="error",
                    blocked=True,
                    requires_review=False,
                )
            )
            blocked += 1

    log.info(
        "api_batch_complete",
        total=len(results),
        blocked=blocked,
        allowed=allowed,
        review=review,
    )

    return BatchAnalyseResponse(
        total=len(results),
        blocked_count=blocked,
        allowed_count=allowed,
        review_count=review,
        results=results,
    )


@router.get(
    "/audit/verify",
    response_model=AuditVerifyResponse,
    summary="Verify audit log hash chain integrity",
    tags=["Audit"],
)
async def audit_verify(
    engine: AnalysisEngine = Depends(get_engine),
) -> AuditVerifyResponse:
    """
    Verify the SHA-256 hash chain of the audit log.

    Returns intact=true if the chain is unbroken.
    Returns intact=false with error details if tampering is detected.
    A broken chain is a critical security event requiring immediate investigation.
    """
    try:
        ok, errors = engine.verify_audit_chain()
        count = engine.audit_count()
        return AuditVerifyResponse(
            chain_intact=ok,
            entry_count=count,
            errors=errors[:10],  # Return first 10 errors maximum
        )
    except Exception as exc:
        log.error("audit_verify_error", exc_type=type(exc).__name__)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Audit verification failed.",
        )


@router.get(
    "/metrics/summary",
    response_model=MetricsSummaryResponse,
    summary="Security metrics summary",
    tags=["Metrics"],
)
async def metrics_summary(
    engine: AnalysisEngine = Depends(get_engine),
) -> MetricsSummaryResponse:
    """
    Returns aggregated security metrics for dashboard display.

    For Prometheus-format metrics, use the /metrics endpoint
    (enabled when AISec is started with --metrics flag).
    """
    try:
        entries = engine._logger.get_all()
        analysis_entries = [e for e in entries if e.record_type == "analysis"]
        total = len(analysis_entries)
        blocked = sum(
            1
            for e in analysis_entries
            if e.payload.get("decision") in ("BLOCK", "ESCALATE")
        )
        review = sum(
            1 for e in analysis_entries if e.payload.get("decision") == "PENDING_REVIEW"
        )
        allowed = total - blocked - review

        scores = [e.payload.get("risk_score", 0.0) for e in analysis_entries]
        avg_risk = sum(scores) / len(scores) if scores else 0.0
        ok, _ = engine.verify_audit_chain()

        return MetricsSummaryResponse(
            total_events=total,
            blocked_events=blocked,
            allowed_events=allowed,
            review_events=review,
            block_rate=round(blocked / total, 4) if total > 0 else 0.0,
            avg_risk_score=round(avg_risk, 4),
            audit_chain_ok=ok,
        )
    except Exception as exc:
        log.error("metrics_error", exc_type=type(exc).__name__)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Metrics collection failed.",
        )
