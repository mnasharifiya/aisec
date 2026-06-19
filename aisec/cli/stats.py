"""
AISec statistics dashboard CLI command.

Displays a professional security statistics dashboard from the AISec
tamper-evident audit log.

Run:
    aisec stats
    aisec stats --log-path ./audit.jsonl
    aisec stats --json
    aisec stats --top 10
"""

from __future__ import annotations

import json
import statistics as stats
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import click

from aisec.storage.audit import DEFAULT_LOG_PATH

DECISION_BLOCKING = {"BLOCK", "ESCALATE"}
DECISION_REVIEW = {"PENDING_REVIEW", "REVIEW", "HUMAN_REVIEW"}
DECISION_ALLOW = {"ALLOW", "APPROVE", "APPROVED"}


def _payload(entry: Any) -> dict[str, Any]:
    """Safely return an audit entry payload."""
    value = getattr(entry, "payload", {}) or {}
    return value if isinstance(value, dict) else {}


def _record_type(entry: Any) -> str:
    """Safely return an audit entry record type."""
    return str(getattr(entry, "record_type", "") or "")


def _safe_float(value: Any, default: float = 0.0) -> float:
    """Convert a value to float without crashing the dashboard."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _decision(entry: Any) -> str:
    """Normalize decision labels."""
    return str(_payload(entry).get("decision", "UNKNOWN") or "UNKNOWN").upper()


def _percent(part: int, total: int) -> str:
    """Format percentage safely."""
    if total <= 0:
        return "0.0%"
    return f"{part / total * 100:.1f}%"


def _counter_from_payload(
    entries: Iterable[Any],
    key: str,
    *,
    fallback: str = "unknown",
) -> Counter[str]:
    """Build a Counter from a payload key."""
    counter: Counter[str] = Counter()

    for entry in entries:
        value = _payload(entry).get(key, fallback)
        value = str(value or fallback)
        counter[value] += 1

    return counter


def _rule_counter(entries: Iterable[Any]) -> Counter[str]:
    """Count rule hits across analysis entries."""
    counter: Counter[str] = Counter()

    for entry in entries:
        rule_hits = _payload(entry).get("rule_hits", [])

        if isinstance(rule_hits, str):
            counter[rule_hits] += 1
            continue

        if isinstance(rule_hits, list):
            for rule in rule_hits:
                counter[str(rule)] += 1

    return counter


def _risk_scores(entries: Iterable[Any]) -> list[float]:
    """Extract valid risk scores."""
    scores: list[float] = []

    for entry in entries:
        raw_score = _payload(entry).get("risk_score", None)
        if raw_score is None:
            continue

        score = _safe_float(raw_score, default=-1.0)
        if 0.0 <= score <= 1.0:
            scores.append(score)

    return scores


def _risk_bucket_counts(scores: list[float]) -> dict[str, int]:
    """Group risk scores into useful demo buckets."""
    buckets = {
        "low_0_0_to_0_3": 0,
        "medium_0_3_to_0_7": 0,
        "high_0_7_to_0_9": 0,
        "critical_0_9_to_1_0": 0,
    }

    for score in scores:
        if score < 0.3:
            buckets["low_0_0_to_0_3"] += 1
        elif score < 0.7:
            buckets["medium_0_3_to_0_7"] += 1
        elif score < 0.9:
            buckets["high_0_7_to_0_9"] += 1
        else:
            buckets["critical_0_9_to_1_0"] += 1

    return buckets


def _top_items(counter: Counter[str], limit: int) -> list[tuple[str, int]]:
    """Return top counter entries."""
    return counter.most_common(max(1, limit))


def _print_counter(title: str, counter: Counter[str], limit: int) -> None:
    """Print a formatted top-N counter section."""
    if not counter:
        return

    click.echo(f"\n  {'─' * 56}")
    click.echo(f"  {title}")
    click.echo(f"  {'─' * 56}")

    for name, count in _top_items(counter, limit):
        click.echo(f"  {name:<35} {count:>8,}")


def _json_dashboard(
    *,
    total_entries: int,
    total_analysis: int,
    blocked: int,
    reviewed: int,
    allowed: int,
    unknown: int,
    scores: list[float],
    temporal_count: int,
    correlation_count: int,
    safe_state_count: int,
    analyst_count: int,
    active_safe_states: int,
    audit_ok: bool,
    audit_errors: list[str],
    top_rules: Counter[str],
    top_agents: Counter[str],
    top_actions: Counter[str],
) -> dict[str, Any]:
    """Build JSON dashboard payload."""
    risk_stats: dict[str, Any] = {
        "count": len(scores),
        "buckets": _risk_bucket_counts(scores),
    }

    if scores:
        risk_stats.update(
            {
                "mean": stats.mean(scores),
                "median": stats.median(scores),
                "min": min(scores),
                "max": max(scores),
                "stdev": stats.stdev(scores) if len(scores) > 1 else None,
            }
        )

    return {
        "audit": {
            "total_entries": total_entries,
            "chain_intact": audit_ok,
            "errors": audit_errors,
        },
        "analysis": {
            "total_events": total_analysis,
            "blocked_or_escalated": blocked,
            "pending_review": reviewed,
            "allowed": allowed,
            "unknown": unknown,
            "block_rate_percent": (
                (blocked / total_analysis * 100) if total_analysis else 0.0
            ),
        },
        "risk_scores": risk_stats,
        "alerts": {
            "temporal_alerts": temporal_count,
            "correlation_alerts": correlation_count,
            "safe_state_activations": safe_state_count,
            "analyst_decisions": analyst_count,
            "active_safe_states": active_safe_states,
        },
        "top_rules": dict(top_rules),
        "top_agents": dict(top_agents),
        "top_actions": dict(top_actions),
    }


@click.command("stats")
@click.option(
    "--log-path",
    type=click.Path(path_type=Path),
    default=DEFAULT_LOG_PATH,
    show_default=True,
    help="Path to AISec audit log.",
)
@click.option(
    "--top",
    "top_n",
    type=int,
    default=5,
    show_default=True,
    help="Number of top rules/agents/actions to show.",
)
@click.option(
    "--json",
    "json_mode",
    is_flag=True,
    help="Print machine-readable JSON output.",
)
@click.option(
    "--fail-on-broken-chain",
    is_flag=True,
    help="Exit with non-zero code if audit chain verification fails.",
)
def stats_command(
    log_path: Path,
    top_n: int,
    json_mode: bool,
    fail_on_broken_chain: bool,
) -> None:
    """
    Display security statistics dashboard.

    Shows decision distribution, risk score analysis, top rules,
    top agents, temporal alerts, correlation alerts, safe-state
    activity, analyst decisions, and audit-chain status.
    """
    from aisec.core.engine import AnalysisEngine

    engine = AnalysisEngine(log_path=log_path)

    try:
        entries = engine._logger.get_all()
    except Exception as exc:
        raise click.ClickException(f"Failed to read audit log: {exc}") from exc

    analysis = [entry for entry in entries if _record_type(entry) == "analysis"]

    temporal = [
        entry
        for entry in entries
        if _record_type(entry) in {"temporal_alert", "temporal_anomaly"}
    ]

    correlation = [
        entry
        for entry in entries
        if _record_type(entry) in {"correlation_alert", "multi_agent_correlation"}
    ]

    safe_state_entries = [
        entry
        for entry in entries
        if _record_type(entry) in {"safe_state_entry", "safe_state_enter"}
    ]

    analyst_decisions = [
        entry for entry in entries if _record_type(entry) == "analyst_decision"
    ]

    total = len(analysis)

    decisions = [_decision(entry) for entry in analysis]
    blocked = sum(1 for decision in decisions if decision in DECISION_BLOCKING)
    reviewed = sum(1 for decision in decisions if decision in DECISION_REVIEW)
    allowed = sum(1 for decision in decisions if decision in DECISION_ALLOW)
    unknown = total - blocked - reviewed - allowed

    scores = _risk_scores(analysis)

    rule_counts = _rule_counter(analysis)
    agent_counts = _counter_from_payload(analysis, "agent_id")
    action_counts = _counter_from_payload(analysis, "action_type")

    ok, errors = engine.verify_audit_chain()
    error_texts = [str(error) for error in errors]

    active_safe_states = engine.safe_state.active_count()
    total_audit_entries = engine.audit_count()

    if json_mode:
        payload = _json_dashboard(
            total_entries=total_audit_entries,
            total_analysis=total,
            blocked=blocked,
            reviewed=reviewed,
            allowed=allowed,
            unknown=unknown,
            scores=scores,
            temporal_count=len(temporal),
            correlation_count=len(correlation),
            safe_state_count=len(safe_state_entries),
            analyst_count=len(analyst_decisions),
            active_safe_states=active_safe_states,
            audit_ok=ok,
            audit_errors=error_texts,
            top_rules=rule_counts,
            top_agents=agent_counts,
            top_actions=action_counts,
        )
        click.echo(json.dumps(payload, indent=2, sort_keys=True))
        if fail_on_broken_chain and not ok:
            raise click.ClickException("Audit chain is broken.")
        return

    click.echo("\n" + "=" * 60)
    click.echo("  AISec Security Statistics Dashboard")
    click.echo("=" * 60)

    click.echo(f"\n  Audit log: {log_path}")

    if total == 0:
        click.echo("\n  No events analysed yet.")
        click.echo("  Run: aisec monitor --scenario trading_ai\n")

        ok, errors = engine.verify_audit_chain()
        chain_str = (
            click.style("INTACT", fg="green")
            if ok
            else click.style(f"BROKEN ({len(errors)} errors)", fg="red")
        )

        click.echo(f"  Audit entries:     {total_audit_entries:>8,}")
        click.echo(f"  Chain integrity:   {chain_str}")
        click.echo("\n" + "=" * 60 + "\n")

        if fail_on_broken_chain and not ok:
            raise click.ClickException("Audit chain is broken.")

        return

    click.echo(f"\n  {'─' * 56}")
    click.echo("  Event Analysis Summary")
    click.echo(f"  {'─' * 56}")
    click.echo(f"  Total events analysed:     {total:>8,}")
    click.echo(
        f"  Blocked / Escalated:       {blocked:>8,}  ({_percent(blocked, total)})"
    )
    click.echo(
        f"  Pending review:            {reviewed:>8,}  ({_percent(reviewed, total)})"
    )
    click.echo(
        f"  Allowed:                   {allowed:>8,}  ({_percent(allowed, total)})"
    )
    click.echo(
        f"  Unknown decisions:         {unknown:>8,}  ({_percent(unknown, total)})"
    )

    if scores:
        click.echo(f"\n  {'─' * 56}")
        click.echo("  Risk Score Distribution")
        click.echo(f"  {'─' * 56}")
        click.echo(f"  Count:  {len(scores):>8,}")
        click.echo(f"  Mean:   {stats.mean(scores):.4f}")
        click.echo(f"  Median: {stats.median(scores):.4f}")
        if len(scores) > 1:
            click.echo(f"  Stdev:  {stats.stdev(scores):.4f}")
        else:
            click.echo("  Stdev:  N/A")
        click.echo(f"  Min:    {min(scores):.4f}")
        click.echo(f"  Max:    {max(scores):.4f}")

        buckets = _risk_bucket_counts(scores)
        click.echo("\n  Risk buckets:")
        click.echo(f"    Low       0.0–0.3:   {buckets['low_0_0_to_0_3']:>8,}")
        click.echo(f"    Medium    0.3–0.7:   {buckets['medium_0_3_to_0_7']:>8,}")
        click.echo(f"    High      0.7–0.9:   {buckets['high_0_7_to_0_9']:>8,}")
        click.echo(f"    Critical  0.9–1.0:   {buckets['critical_0_9_to_1_0']:>8,}")

    _print_counter("Top Rules Fired", rule_counts, top_n)
    _print_counter("Top Agents", agent_counts, top_n)
    _print_counter("Top Action Types", action_counts, top_n)

    click.echo(f"\n  {'─' * 56}")
    click.echo("  Security Alerts")
    click.echo(f"  {'─' * 56}")
    click.echo(f"  Temporal alerts:           {len(temporal):>8,}")
    click.echo(f"  Correlation alerts:        {len(correlation):>8,}")
    click.echo(f"  Safe state activations:    {len(safe_state_entries):>8,}")
    click.echo(f"  Analyst decisions:         {len(analyst_decisions):>8,}")
    click.echo(f"  Active safe states:        {active_safe_states:>8,}")

    chain_str = (
        click.style("INTACT", fg="green")
        if ok
        else click.style(f"BROKEN ({len(errors)} errors)", fg="red")
    )

    click.echo(f"\n  {'─' * 56}")
    click.echo("  Audit Chain Status")
    click.echo(f"  {'─' * 56}")
    click.echo(f"  Total audit entries:       {total_audit_entries:>8,}")
    click.echo(f"  Chain integrity:           {chain_str}")

    if not ok:
        click.echo("\n  Audit chain errors:")
        for error in error_texts[:5]:
            click.echo(f"  - {error}")
        if len(error_texts) > 5:
            click.echo(f"  ... and {len(error_texts) - 5} more error(s).")

    click.echo("\n" + "=" * 60 + "\n")

    if fail_on_broken_chain and not ok:
        raise click.ClickException("Audit chain is broken.")
