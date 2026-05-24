"""
AISec decision engine.

Combines rule engine results and risk scorer output into a
single enforceable decision for each AI agent action.

Decision logic (in priority order):
    1. If any rule fired with BLOCK   → BLOCK immediately.
    2. If any rule fired with ESCALATE → ESCALATE immediately.
    3. If risk_score >= 0.80          → BLOCK.
    4. If risk_score >= 0.60          → PENDING_REVIEW.
    5. If any rule fired with REVIEW  → PENDING_REVIEW.
    6. If risk_score >= 0.30          → LOG and WATCH (ALLOW with flag).
    7. Otherwise                      → ALLOW.

Design principles:
    - Rules always override the scorer for BLOCK and ESCALATE.
    - The scorer provides a safety net for actions rules do not cover.
    - Every decision includes a human-readable explanation.
    - All decisions are written to the audit log — no silent actions.

Paper reference:
    Section 5 — Formalization and Enforceable Control Mechanisms.
    Section 9 — Decision Policy.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from aisec.core.rules import RuleEngineResult
from aisec.core.scorer import ScoreResult
from aisec.storage.models import AnalysisResult, Decision, Event

# ── Thresholds ────────────────────────────────────────────────────────────────
#
# These match Section 9.1 of the architecture document exactly.
# Change only with documented justification and a test update.

THRESHOLD_BLOCK: float = 0.80  # risk >= 0.80 → immediate block
THRESHOLD_REVIEW: float = 0.60  # risk >= 0.60 → human review required
THRESHOLD_WATCH: float = 0.30  # risk >= 0.30 → log and watch


# ── Decision engine ───────────────────────────────────────────────────────────


@dataclass
class DecisionContext:
    """
    All inputs needed by the decision engine for one event.

    Attributes:
        event:        The original AI action being evaluated.
        rule_result:  Output from the rule engine.
        score_result: Output from the risk scorer.
    """

    event: Event
    rule_result: RuleEngineResult
    score_result: ScoreResult


class DecisionEngine:
    """
    Produces a single AnalysisResult from rule hits and risk score.

    The engine is stateless — it reads inputs and returns a result.
    All state (audit logging, SOC queue) is handled by the caller.

    Usage:
        engine  = DecisionEngine()
        result  = engine.decide(context)
        # result.decision is ALLOW / BLOCK / PENDING_REVIEW / ESCALATE
        # result.explanation tells the analyst exactly why
    """

    def decide(self, ctx: DecisionContext) -> AnalysisResult:
        """
        Evaluate context and return an enforceable AnalysisResult.

        Args:
            ctx: DecisionContext containing event, rule hits, and score.

        Returns:
            AnalysisResult with decision, risk score, and explanation.
        """
        decision, explanation = self._apply_logic(ctx)

        return AnalysisResult(
            event_id=ctx.event.event_id,
            risk_score=ctx.score_result.risk_score,
            decision=decision,
            explanation=explanation,
            rule_hits=ctx.rule_result.rule_ids,
            baseline_similarity=1.0,  # temporal module wires this in v2
            risk_delta=0.0,  # temporal module wires this in v2
        )

    # ── Decision logic ────────────────────────────────────────────────────────

    def _apply_logic(self, ctx: DecisionContext) -> tuple[Decision, str]:
        """
        Apply the priority-ordered decision logic.

        Returns:
            (Decision, explanation string)
        """
        score = ctx.score_result.risk_score
        rules = ctx.rule_result

        # ── Priority 1: Rule-driven BLOCK ────────────────────────────────────
        if rules.final_decision == Decision.BLOCK:
            reason = self._first_reason(rules, Decision.BLOCK)
            return (
                Decision.BLOCK,
                f"[RULE BLOCK] {reason} "
                f"| risk={score:.3f} "
                f"| rules={rules.rule_ids}",
            )

        # ── Priority 2: Rule-driven ESCALATE ─────────────────────────────────
        if rules.final_decision == Decision.ESCALATE:
            reason = self._first_reason(rules, Decision.ESCALATE)
            return (
                Decision.ESCALATE,
                f"[RULE ESCALATE] {reason} "
                f"| risk={score:.3f} "
                f"| rules={rules.rule_ids}",
            )

        # ── Priority 3: Score-driven BLOCK ────────────────────────────────────
        if score >= THRESHOLD_BLOCK:
            return (
                Decision.BLOCK,
                f"[SCORE BLOCK] risk_score={score:.3f} >= {THRESHOLD_BLOCK} "
                f"| {ctx.score_result.explanation}",
            )

        # ── Priority 4: Score-driven REVIEW ──────────────────────────────────
        if score >= THRESHOLD_REVIEW:
            return (
                Decision.PENDING_REVIEW,
                f"[SCORE REVIEW] risk_score={score:.3f} >= {THRESHOLD_REVIEW} "
                f"| {ctx.score_result.explanation}",
            )

        # ── Priority 5: Rule-driven REVIEW ───────────────────────────────────
        if rules.final_decision == Decision.PENDING_REVIEW:
            reason = self._first_reason(rules, Decision.PENDING_REVIEW)
            return (
                Decision.PENDING_REVIEW,
                f"[RULE REVIEW] {reason} "
                f"| risk={score:.3f} "
                f"| rules={rules.rule_ids}",
            )

        # ── Priority 6: Score-driven WATCH ───────────────────────────────────
        if score >= THRESHOLD_WATCH:
            return (
                Decision.ALLOW,
                f"[WATCH] risk_score={score:.3f} >= {THRESHOLD_WATCH} — "
                f"allowed but logged for monitoring "
                f"| {ctx.score_result.explanation}",
            )

        # ── Priority 7: ALLOW ─────────────────────────────────────────────────
        return (
            Decision.ALLOW,
            f"[ALLOW] risk_score={score:.3f} — "
            f"below all thresholds, no rules fired "
            f"| {ctx.score_result.explanation}",
        )

    @staticmethod
    def _first_reason(rules: RuleEngineResult, decision: Decision) -> str:
        """
        Return the reason string from the first rule hit matching decision.

        Falls back to a generic message if no matching hit is found.
        """
        for hit in rules.hits:
            if hit.decision == decision:
                return hit.reason
        return f"Rule decision: {decision.value}"
