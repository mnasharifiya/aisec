"""
AISec live monitor CLI command.
"""

from __future__ import annotations

import json
import random
import time
from pathlib import Path
from typing import Any

import click

from aisec.storage.audit import DEFAULT_LOG_PATH


def _decision_name(decision: Any) -> str:
    if hasattr(decision, "name"):
        return str(decision.name)
    return str(decision)


def _decision_label_color(decision: Any) -> tuple[str, str]:
    name = _decision_name(decision).upper()

    if name == "ALLOW":
        return "ALLOW", "green"
    if name == "BLOCK":
        return "BLOCK", "red"
    if name == "ESCALATE":
        return "ESCALATE", "red"
    if name == "PENDING_REVIEW":
        return "REVIEW", "yellow"

    return name, "white"


def _risk_color(score: float) -> str:
    if score >= 0.90:
        return "red"
    if score >= 0.70:
        return "yellow"
    if score >= 0.30:
        return "cyan"
    return "green"


def _safe_alert_name(alert: Any) -> str:
    threat = getattr(alert, "threat", None)
    if hasattr(threat, "name"):
        return str(threat.name)
    return str(threat or "UNKNOWN")


def _safe_alert_severity(alert: Any) -> str:
    severity = getattr(alert, "severity", "unknown")
    if hasattr(severity, "name"):
        return str(severity.name)
    return str(severity)


@click.command("monitor")
@click.option(
    "--scenario",
    default="trading_ai",
    type=click.Choice(["trading_ai", "urban_ai", "both"]),
    show_default=True,
    help="Scenario to simulate.",
)
@click.option(
    "--steps",
    type=click.IntRange(1, 100_000),
    default=20,
    show_default=True,
    help="Number of events to simulate.",
)
@click.option(
    "--delay",
    type=click.FloatRange(0.0, 60.0),
    default=0.1,
    show_default=True,
    help="Delay between events in seconds.",
)
@click.option(
    "--seed",
    type=int,
    default=None,
    help="Random seed for reproducible simulations.",
)
@click.option(
    "--json",
    "json_mode",
    is_flag=True,
    help="Print JSON lines instead of table output.",
)
@click.option(
    "--log-path",
    type=click.Path(path_type=Path),
    default=DEFAULT_LOG_PATH,
    show_default=True,
    help="Path to AISec audit log.",
)
def monitor_command(
    scenario: str,
    steps: int,
    delay: float,
    seed: int | None,
    json_mode: bool,
    log_path: Path,
) -> None:
    """
    Live stream of AISec agent activity.
    """
    from aisec.core.engine import AnalysisEngine
    from aisec.storage.models import Event, Scenario

    if seed is not None:
        random.seed(seed)

    engine = AnalysisEngine(log_path=log_path)

    trading_actions = [
        ("read_market_data", Scenario.TRADING_AI, "MARKET", {}),
        ("read_market_data", Scenario.TRADING_AI, "MARKET", {}),
        ("read_market_data", Scenario.TRADING_AI, "MARKET", {}),
        ("execute_trade", Scenario.TRADING_AI, "MARKET", {"amount": 50_000}),
        ("execute_trade", Scenario.TRADING_AI, "MARKET", {"amount": 500_000}),
        ("execute_large_trade", Scenario.TRADING_AI, "MARKET", {"amount": 2_400_000}),
        ("manipulate_news_feed", Scenario.TRADING_AI, "MARKET", {}),
        ("override_risk_limit", Scenario.TRADING_AI, "MARKET", {}),
        (
            "execute_trade",
            Scenario.TRADING_AI,
            "MARKET",
            {"amount": 800, "after_hours": True},
        ),
    ]

    urban_actions = [
        ("read_sensor", Scenario.URBAN_AI, "city_system", {}),
        ("read_sensor", Scenario.URBAN_AI, "city_system", {}),
        ("adjust_traffic_flow", Scenario.URBAN_AI, "city_system", {}),
        (
            "mass_traffic_redirect",
            Scenario.URBAN_AI,
            "city_system",
            {"affected_intersections": 10},
        ),
        (
            "mass_traffic_redirect",
            Scenario.URBAN_AI,
            "city_system",
            {"affected_intersections": 120},
        ),
        (
            "set_curfew",
            Scenario.URBAN_AI,
            "city_system",
            {"zone": "ALL", "duration_hours": 48},
        ),
        (
            "shutdown_power_grid",
            Scenario.URBAN_AI,
            "city_system",
            {"zone": "North"},
        ),
        (
            "adjust_routing",
            Scenario.URBAN_AI,
            "city_system",
            {"target": "ambulance_routing"},
        ),
    ]

    if scenario == "trading_ai":
        pool = trading_actions
    elif scenario == "urban_ai":
        pool = urban_actions
    else:
        pool = trading_actions + urban_actions

    blocked = 0
    reviewed = 0
    escalated = 0
    temporal_count = 0
    correlation_count = 0
    safe_state_blocks = 0
    start = time.perf_counter()

    if not json_mode:
        click.echo(f"\n  AISec Live Monitor -- {scenario} -- {steps} events")
        click.echo(f"  Log path: {log_path}")
        if seed is not None:
            click.echo(f"  Seed: {seed}")
        click.echo(f"  {'-' * 72}")
        click.echo(f"  {'#':>4}  {'Action':<30} {'Risk':>6}  {'Decision':<14} Agent")
        click.echo(f"  {'-' * 72}")

    for index in range(1, steps + 1):
        action_type, scen, target, payload = random.choice(pool)
        agent_id = f"agent_{((index - 1) % 3) + 1:02d}"

        event = Event(
            action_type=action_type,
            agent_id=agent_id,
            target=target,
            scenario=scen,
            raw_payload=dict(payload),
        )

        before = time.perf_counter()
        result = engine.analyse(event)
        latency_ms = (time.perf_counter() - before) * 1000

        risk_score = float(getattr(result, "risk_score", 0.0))
        decision = getattr(result, "decision", "UNKNOWN")
        decision_name = _decision_name(decision).upper()
        label, color = _decision_label_color(decision)

        result_blocked = bool(getattr(result, "blocked", False))
        temporal_alerts = getattr(result, "temporal_alerts", []) or []
        correlation_alerts = getattr(result, "correlation_alerts", []) or []
        safe_state_block = bool(getattr(result, "safe_state_block", False))

        if result_blocked:
            blocked += 1
        if decision_name == "PENDING_REVIEW":
            reviewed += 1
        if decision_name == "ESCALATE":
            escalated += 1
        if safe_state_block:
            safe_state_blocks += 1

        temporal_count += len(temporal_alerts)
        correlation_count += len(correlation_alerts)

        if json_mode:
            click.echo(
                json.dumps(
                    {
                        "index": index,
                        "scenario": scenario,
                        "agent_id": agent_id,
                        "action_type": action_type,
                        "target": target,
                        "risk_score": risk_score,
                        "decision": decision_name,
                        "blocked": result_blocked,
                        "safe_state_block": safe_state_block,
                        "temporal_alert_count": len(temporal_alerts),
                        "correlation_alert_count": len(correlation_alerts),
                        "latency_ms": latency_ms,
                    },
                    sort_keys=True,
                )
            )
        else:
            prefix = "X" if result_blocked else "+"
            risk_text = click.style(f"{risk_score:>6.3f}", fg=_risk_color(risk_score))

            click.echo(
                f"  {index:>4}  {prefix} {action_type:<30} "
                f"{risk_text}  "
                + click.style(f"{label:<14}", fg=color)
                + f" {agent_id}  {latency_ms:.1f}ms"
            )

            for alert in temporal_alerts:
                click.echo(
                    click.style(
                        f"        ! TEMPORAL: {_safe_alert_name(alert)} "
                        f"[{_safe_alert_severity(alert)}]",
                        fg="yellow",
                    )
                )

            for alert in correlation_alerts:
                click.echo(
                    click.style(
                        f"        ! CORRELATION: {_safe_alert_name(alert)} "
                        f"[{_safe_alert_severity(alert)}]",
                        fg="magenta",
                    )
                )

            if safe_state_block:
                click.echo(
                    click.style(
                        "        SAFE STATE BLOCK -- agent restricted",
                        fg="red",
                    )
                )

        if delay > 0:
            time.sleep(delay)

    duration = max(time.perf_counter() - start, 0.0001)
    throughput = steps / duration

    summary = {
        "type": "summary",
        "events": steps,
        "blocked": blocked,
        "pending_review": reviewed,
        "escalated": escalated,
        "safe_state_blocks": safe_state_blocks,
        "temporal_alerts": temporal_count,
        "correlation_alerts": correlation_count,
        "block_rate_percent": blocked / steps * 100,
        "events_per_second": throughput,
    }

    if json_mode:
        click.echo(json.dumps(summary, sort_keys=True))
        return

    click.echo(f"  {'-' * 72}")
    click.echo(
        f"  Complete: {steps} events, {blocked} blocked "
        f"({blocked / steps * 100:.1f}% block rate)"
    )
    click.echo(
        f"  Review: {reviewed} pending, {escalated} escalated, "
        f"{safe_state_blocks} safe-state blocks"
    )
    click.echo(f"  Alerts: {temporal_count} temporal, {correlation_count} correlation")
    click.echo(f"  Throughput: {throughput:.2f} events/sec\n")
