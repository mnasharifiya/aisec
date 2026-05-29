"""
AISec analysis engine.

The analysis engine is the single entry point for the entire
analysis pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from aisec.core.decision import DecisionContext, DecisionEngine
from aisec.core.rules import RuleEngine
from aisec.core.scorer import RiskScorer
from aisec.core.vector import FeatureVectorBuilder
from aisec.storage.audit import AuditLogger, DEFAULT_LOG_PATH
from aisec.storage.models import AnalysisResult, Decision, Event

# Use TYPE_CHECKING for imports that cause circular dependencies
if TYPE_CHECKING:
    from aisec.core.temporal import (
        TemporalAlert,
        TemporalAnomalyDetector,
        TemporalConfig,
    )

# ── Engine result ─────────────────────────────────────────────────────────────


@dataclass
class EngineResult:
    """
    Complete output of the analysis engine for one Event.
    """

    event: Event
    analysis: AnalysisResult
    log_entry_id: str
    temporal_alerts: list[TemporalAlert] = field(default_factory=list)

    @property
    def blocked(self) -> bool:
        """True if the action must not proceed."""
        return self.analysis.decision in (
            Decision.BLOCK,
            Decision.ESCALATE,
            Decision.PENDING_REVIEW,
        )

    @property
    def requires_review(self) -> bool:
        """True if a human analyst must review before proceeding."""
        return self.analysis.decision == Decision.PENDING_REVIEW

    @property
    def decision(self) -> Decision:
        """Convenience accessor for the enforcement decision."""
        return self.analysis.decision

    @property
    def risk_score(self) -> float:
        """Convenience accessor for the risk score."""
        return self.analysis.risk_score


# ── Analysis engine ───────────────────────────────────────────────────────────


class AnalysisEngine:
    """Orchestrates the full AISec analysis pipeline."""

    def __init__(
        self,
        log_path: Path = DEFAULT_LOG_PATH,
        vector_builder: FeatureVectorBuilder | None = None,
        scorer: RiskScorer | None = None,
        rule_engine: RuleEngine | None = None,
        decision_engine: DecisionEngine | None = None,
        audit_logger: AuditLogger | None = None,
        temporal_config: TemporalConfig | None = None,
        enable_temporal: bool = True,
    ) -> None:
        self._builder = vector_builder or FeatureVectorBuilder()
        self._scorer = scorer or RiskScorer()
        self._rules = rule_engine or RuleEngine()
        self._decision = decision_engine or DecisionEngine()
        self._logger = audit_logger or AuditLogger(log_path)

        self._temporal = None
        if enable_temporal:
            # Deferred import to break circular dependency
            from aisec.core.temporal import TemporalAnomalyDetector, TemporalConfig

            self._temporal = TemporalAnomalyDetector(
                temporal_config or TemporalConfig()
            )

    def analyse(self, event: Event) -> EngineResult:
        """Run the full analysis pipeline for a single Event."""
        fv = self._builder.build(event)
        score = self._scorer.score(fv, event.scenario)
        rules = self._rules.evaluate(event)

        ctx = DecisionContext(event=event, rule_result=rules, score_result=score)
        analysis = self._decision.decide(ctx)

        log_entry = self._logger.log(
            record_type="analysis",
            record_id=event.event_id,
            payload={
                "agent_id": event.agent_id,
                "action_type": event.action_type,
                "target": event.target,
                "scenario": event.scenario.value,
                "risk_score": analysis.risk_score,
                "decision": analysis.decision.value,
                "rule_hits": analysis.rule_hits,
                "explanation": analysis.explanation,
            },
        )

        result = EngineResult(
            event=event,
            analysis=analysis,
            log_entry_id=log_entry.log_id,
        )

        if self._temporal is not None:
            temporal_alerts = self._temporal.update(result)
            for alert in temporal_alerts:
                self._logger.log(
                    record_type="temporal_alert",
                    record_id=event.event_id,
                    payload={
                        "agent_id": alert.agent_id,
                        "threat": alert.threat.name,
                        "severity": alert.severity,
                        "description": alert.description,
                        "evidence": alert.evidence,
                    },
                )
            result.temporal_alerts = temporal_alerts

        return result

    def verify_audit_chain(self) -> tuple[bool, list[str]]:
        """Verify the integrity of the audit log hash chain."""
        return self._logger.verify_chain()

    def audit_count(self) -> int:
        """Return the number of entries in the audit log."""
        return self._logger.count()
