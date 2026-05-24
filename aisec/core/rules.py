"""
AISec rule engine.

Evaluates AI agent actions against a set of hard policy rules.
Rules fire before the risk scorer — a rule hit can immediately
block or escalate an action regardless of its numerical risk score.

Design principles:
  - Rules are pure functions: (Event) -> RuleResult
  - Rules are grouped by scenario for clarity and extensibility
  - Adding a new rule = adding one function + registering it
  - Rule hits are logged by ID so analysts can trace decisions
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

from aisec.storage.models import Decision, Event, Scenario

# ── Rule result ───────────────────────────────────────────────────────────────


@dataclass
class RuleResult:
    """
    Outcome of evaluating a single rule against an Event.

    Attributes:
        fired:      True if the rule matched the event.
        rule_id:    Unique identifier of this rule.
        decision:   Enforcement decision if fired, else None.
        reason:     Human-readable explanation for analysts.
    """

    fired: bool
    rule_id: str
    decision: Decision | None = None
    reason: str = ""


# ── Type alias ────────────────────────────────────────────────────────────────

RuleFunction = Callable[[Event], RuleResult]


# ── Shared rule helpers ───────────────────────────────────────────────────────


def _no_match(rule_id: str) -> RuleResult:
    """Return a standard non-firing result."""
    return RuleResult(fired=False, rule_id=rule_id)


def _block(rule_id: str, reason: str) -> RuleResult:
    """Return a firing result that immediately blocks the action."""
    return RuleResult(
        fired=True,
        rule_id=rule_id,
        decision=Decision.BLOCK,
        reason=reason,
    )


def _escalate(rule_id: str, reason: str) -> RuleResult:
    """Return a firing result that escalates to a senior analyst."""
    return RuleResult(
        fired=True,
        rule_id=rule_id,
        decision=Decision.ESCALATE,
        reason=reason,
    )


def _review(rule_id: str, reason: str) -> RuleResult:
    """Return a firing result that routes to the SOC review queue."""
    return RuleResult(
        fired=True,
        rule_id=rule_id,
        decision=Decision.PENDING_REVIEW,
        reason=reason,
    )


# ── Trading AI rules (Scenario A) ─────────────────────────────────────────────


def rule_trading_large_trade(event: Event) -> RuleResult:
    """
    Block any trade where the amount exceeds the high-risk threshold.

    Threshold: $1,000,000 USD (configurable in future via policies.yaml).
    Rationale: A single trade of this size can destabilise markets.
    Paper reference: Scenario A — Autonomous Trading Agent.
    """
    rule_id = "TRADING-001"

    if event.action_type not in ("execute_trade", "execute_large_trade"):
        return _no_match(rule_id)

    amount = event.raw_payload.get("amount", 0)
    try:
        amount = float(amount)
    except (TypeError, ValueError):
        return _no_match(rule_id)

    if amount >= 1_000_000:
        return _block(
            rule_id,
            f"Trade amount ${amount:,.0f} exceeds $1,000,000 threshold. "
            "Immediate block — human approval required before any trade of this size.",
        )
    return _no_match(rule_id)


def rule_trading_news_manipulation(event: Event) -> RuleResult:
    """
    Block any attempt to manipulate news feeds or market data.

    Rationale: AI-generated disinformation to move markets is illegal
    and can cause systemic economic harm.
    Paper reference: Scenario A threat — news feed manipulation.
    """
    rule_id = "TRADING-002"

    blocked_actions = {
        "manipulate_news_feed",
        "inject_market_data",
        "publish_fake_news",
        "alter_market_feed",
    }

    if event.action_type in blocked_actions:
        return _block(
            rule_id,
            f"Action '{event.action_type}' is classified as market manipulation. "
            "This action is unconditionally blocked regardless of amount.",
        )
    return _no_match(rule_id)


def rule_trading_risk_limit_override(event: Event) -> RuleResult:
    """
    Escalate any attempt to modify or override risk management settings.

    Rationale: Self-modification of risk limits is a key early indicator
    of an AI agent pursuing unsafe strategies.
    """
    rule_id = "TRADING-003"

    if event.action_type in ("override_risk_limit", "modify_risk_parameters"):
        return _escalate(
            rule_id,
            f"Agent '{event.agent_id}' attempted to override risk limits. "
            "Escalated immediately — this is a red-flag behaviour.",
        )
    return _no_match(rule_id)


def rule_trading_after_hours(event: Event) -> RuleResult:
    """
    Flag trades executed outside normal market hours for review.

    Rationale: After-hours trading by autonomous agents without
    oversight introduces unacceptable market risk.
    """
    rule_id = "TRADING-004"

    if event.action_type in ("execute_trade", "execute_large_trade"):
        after_hours = event.raw_payload.get("after_hours", False)
        if after_hours:
            return _review(
                rule_id,
                f"Trade flagged for human review — executed outside "
                "normal market hours without prior approval.",
            )
    return _no_match(rule_id)


# ── Urban AI rules (Scenario B) ───────────────────────────────────────────────


def rule_urban_curfew(event: Event) -> RuleResult:
    """
    Block any attempt to impose city-wide movement restrictions.

    Rationale: Restricting citizen movement is a civil rights matter
    that must never be decided autonomously by an AI system.
    Paper reference: Scenario B — Smart City Infrastructure AI.
    """
    rule_id = "URBAN-001"

    if event.action_type in ("set_curfew", "restrict_movement", "lockdown_zone"):
        zone = event.raw_payload.get("zone", "unknown")
        hours = event.raw_payload.get("duration_hours", 0)
        return _block(
            rule_id,
            f"Curfew attempt blocked — zone={zone}, duration={hours}h. "
            "Civil movement restrictions require government authorisation. "
            "This action is unconditionally blocked.",
        )
    return _no_match(rule_id)


def rule_urban_power_grid(event: Event) -> RuleResult:
    """
    Escalate any action that would cut power to a populated zone.

    Rationale: Power outages in populated areas endanger lives —
    hospitals, emergency services, heating in winter.
    """
    rule_id = "URBAN-002"

    if event.action_type in ("cut_power", "shutdown_power_grid", "disable_zone_power"):
        zone = event.raw_payload.get("zone", "unknown")
        return _escalate(
            rule_id,
            f"Power cutoff to zone '{zone}' escalated. "
            "This action risks lives — requires senior human authorisation.",
        )
    return _no_match(rule_id)


def rule_urban_emergency_services(event: Event) -> RuleResult:
    """
    Block any action that interferes with emergency services.

    Rationale: Police, fire, and ambulance systems must never be
    disrupted by an autonomous AI under any circumstances.
    """
    rule_id = "URBAN-003"

    protected_targets = {
        "emergency_dispatch",
        "police_network",
        "fire_service",
        "ambulance_routing",
        "hospital_power",
    }

    if event.target in protected_targets:
        return _block(
            rule_id,
            f"Action targeting '{event.target}' is unconditionally blocked. "
            "Emergency service systems are protected infrastructure.",
        )
    return _no_match(rule_id)


def rule_urban_traffic_override(event: Event) -> RuleResult:
    """
    Route large-scale traffic overrides to human review.

    Minor traffic adjustments are permitted. City-wide overrides
    affecting more than 50 intersections require analyst approval.
    """
    rule_id = "URBAN-004"

    if event.action_type in ("override_traffic_system", "mass_traffic_redirect"):
        affected = event.raw_payload.get("affected_intersections", 0)
        try:
            affected = int(affected)
        except (TypeError, ValueError):
            affected = 0

        if affected > 50:
            return _review(
                rule_id,
                f"Traffic override affecting {affected} intersections "
                "requires human review before execution.",
            )
    return _no_match(rule_id)


# ── Rule registry ─────────────────────────────────────────────────────────────

# Rules are grouped by scenario.
# To add a new rule: write the function above, add it to the correct list here.

TRADING_AI_RULES: list[RuleFunction] = [
    rule_trading_large_trade,
    rule_trading_news_manipulation,
    rule_trading_risk_limit_override,
    rule_trading_after_hours,
]

URBAN_AI_RULES: list[RuleFunction] = [
    rule_urban_curfew,
    rule_urban_power_grid,
    rule_urban_emergency_services,
    rule_urban_traffic_override,
]

SCENARIO_RULES: dict[Scenario, list[RuleFunction]] = {
    Scenario.TRADING_AI: TRADING_AI_RULES,
    Scenario.URBAN_AI: URBAN_AI_RULES,
}


# ── Rule engine ───────────────────────────────────────────────────────────────


@dataclass
class RuleEngineResult:
    """
    Aggregated output after evaluating all rules for an event.

    Attributes:
        hits:            All rules that fired.
        final_decision:  Highest-priority decision from all hits.
                         BLOCK > ESCALATE > PENDING_REVIEW > None.
        rule_ids:        IDs of all fired rules for audit logging.
    """

    hits: list[RuleResult] = field(default_factory=list)
    final_decision: Decision | None = None
    rule_ids: list[str] = field(default_factory=list)

    @property
    def any_fired(self) -> bool:
        """True if at least one rule fired."""
        return len(self.hits) > 0


# Decision priority — higher index = higher priority
_DECISION_PRIORITY: dict[Decision, int] = {
    Decision.ALLOW: 0,
    Decision.PENDING_REVIEW: 1,
    Decision.ESCALATE: 2,
    Decision.BLOCK: 3,
}


class RuleEngine:
    """
    Evaluates an Event against all rules registered for its scenario.

    Usage:
        engine = RuleEngine()
        result = engine.evaluate(event)
        if result.any_fired:
            print(result.final_decision)
            print(result.rule_ids)
    """

    def evaluate(self, event: Event) -> RuleEngineResult:
        """
        Run all rules for the event's scenario and return aggregated results.

        Rules are evaluated in registration order.
        The final decision is the highest-priority decision across all hits.

        Args:
            event: The intercepted AI action to evaluate.

        Returns:
            RuleEngineResult with all hits and the final enforcement decision.
        """
        rules = SCENARIO_RULES.get(event.scenario, [])
        result = RuleEngineResult()

        for rule_fn in rules:
            rule_result = rule_fn(event)
            if rule_result.fired:
                result.hits.append(rule_result)
                result.rule_ids.append(rule_result.rule_id)
                result.final_decision = self._highest_priority(
                    result.final_decision,
                    rule_result.decision,
                )

        return result

    @staticmethod
    def _highest_priority(
        current: Decision | None,
        incoming: Decision | None,
    ) -> Decision | None:
        """Return whichever decision has higher enforcement priority."""
        if current is None:
            return incoming
        if incoming is None:
            return current
        if _DECISION_PRIORITY[incoming] > _DECISION_PRIORITY[current]:
            return incoming
        return current
