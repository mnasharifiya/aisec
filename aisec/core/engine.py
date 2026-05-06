"""
AISec analysis engine.

The analysis engine is the single entry point for the entire
analysis pipeline. It accepts a raw Event and returns a complete
AnalysisResult by orchestrating:

    Event
      → FeatureVectorBuilder  (normalise raw data into numbers)
      → RiskScorer            (compute R(x) = sigmoid(w^T x + b))
      → RuleEngine            (apply hard policy rules)
      → DecisionEngine        (combine score + rules → decision)
      → AuditLogger           (write tamper-evident log entry)

External callers (CLI, SOC queue, interceptor) only need this module.
They never call the individual components directly.

Design principles:
    - Single responsibility: orchestrate, do not compute.
    - Every action is logged — no silent pass-throughs.
    - The engine is stateless between calls.
    - All dependencies are injected for testability.

Paper reference:
    Section 4 — Proposed Layered Control Framework.
    Section 6 — Prototype Implementation.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from aisec.core.decision import DecisionContext, DecisionEngine
from aisec.core.rules import RuleEngine
from aisec.core.scorer import RiskScorer
from aisec.core.vector import FeatureVectorBuilder
from aisec.storage.audit import AuditLogger, DEFAULT_LOG_PATH
from aisec.storage.models import AnalysisResult, Decision, Event


# ── Engine result ─────────────────────────────────────────────────────────────

@dataclass
class EngineResult:
    """
    Complete output of the analysis engine for one Event.

    Attributes:
        event:           The original event that was analysed.
        analysis:         Full AnalysisResult from the decision engine.
        log_entry_id:     ID of the audit log entry written for this result.
        blocked:          True if the action was blocked or escalated.
        requires_review:  True if a human analyst must review this action.
    """
    event: Event
    analysis: AnalysisResult
    log_entry_id: str

    @property
    def blocked(self) -> bool:
        """
        True if the action must not proceed without human intervention.

        BLOCK and ESCALATE are hard stops.
        PENDING_REVIEW is a soft stop — action is held until a human approves.
        All three prevent autonomous execution.
        """
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
    """
    Orchestrates the full AISec analysis pipeline for a single Event.

    Instantiate once and reuse — all components are stateless
    between calls except the audit logger which appends to disk.

    Usage:
        engine = AnalysisEngine()
        result = engine.analyse(event)

        if result.blocked:
            raise SecurityError("Action blocked by AISec")
        if result.requires_review:
            soc_queue.submit(result)
    """

    def __init__(
        self,
        log_path: Path = DEFAULT_LOG_PATH,
        vector_builder: FeatureVectorBuilder | None = None,
        scorer: RiskScorer | None = None,
        rule_engine: RuleEngine | None = None,
        decision_engine: DecisionEngine | None = None,
        audit_logger: AuditLogger | None = None,
    ) -> None:
        """
        Initialise the engine with its dependencies.

        All dependencies have sensible defaults and are only exposed
        as parameters to allow injection during testing.

        Args:
            log_path:         Where to write the audit log.
            vector_builder:   Converts Events to FeatureVectors.
            scorer:           Computes R(x) = sigmoid(w^T x + b).
            rule_engine:      Applies hard policy rules.
            decision_engine:  Combines score + rules into a decision.
            audit_logger:     Writes tamper-evident log entries.
        """
        self._builder = vector_builder or FeatureVectorBuilder()
        self._scorer = scorer or RiskScorer()
        self._rules = rule_engine or RuleEngine()
        self._decision = decision_engine or DecisionEngine()
        self._logger = audit_logger or AuditLogger(log_path)

    def analyse(self, event: Event) -> EngineResult:
        """
        Run the full analysis pipeline for a single Event.

        Steps:
            1. Build feature vector from the event.
            2. Score the feature vector.
            3. Evaluate hard rules.
            4. Combine into a decision.
            5. Write to audit log.

        Args:
            event: A normalised Event from the interceptor.

        Returns:
            EngineResult containing the decision and audit log entry ID.
        """
        # Step 1 — feature vector
        fv = self._builder.build(event)

        # Step 2 — risk score
        score = self._scorer.score(fv, event.scenario)

        # Step 3 — rule evaluation
        rules = self._rules.evaluate(event)

        # Step 4 — decision
        ctx = DecisionContext(event=event, rule_result=rules, score_result=score)
        analysis = self._decision.decide(ctx)

        # Step 5 — audit log
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

        return EngineResult(
            event=event,
            analysis=analysis,
            log_entry_id=log_entry.log_id,
        )

    def verify_audit_chain(self) -> tuple[bool, list[str]]:
        """
        Verify the integrity of the audit log hash chain.

        Returns:
            (True, [])            — chain is intact.
            (False, [error, ...]) — chain broken, errors describe where.
        """
        return self._logger.verify_chain()

    def audit_count(self) -> int:
        """Return the number of entries in the audit log."""
        return self._logger.count()