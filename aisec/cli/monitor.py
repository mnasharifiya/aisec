"""
AISec live monitor CLI command.

Streams AI agent actions in real time to the terminal,
colour-coded by enforcement decision. Designed to feel
like a professional security operations display.

Security considerations:
    - Monitor is read-only — it cannot modify decisions.
    - Display data is sanitised before rendering.
    - Long strings are truncated to prevent terminal overflow.
    - The monitor cannot be used to approve or block actions.
      That is exclusively the SOC console's responsibility.

Usage:
    aisec monitor --scenario trading_ai
    aisec monitor --scenario urban_ai
    aisec monitor --scenario both
    aisec monitor --steps 50 --scenario trading_ai
"""

from __future__ import annotations

import time
from pathlib import Path

import click
from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from aisec.agents.trading_agent import (
    DANGEROUS_ACTIONS as TRADING_DANGEROUS,
    SAFE_ACTIONS as TRADING_SAFE,
    TradingAgent,
)
from aisec.agents.urban_agent import (
    DANGEROUS_ACTIONS as URBAN_DANGEROUS,
    SAFE_ACTIONS as URBAN_SAFE,
    UrbanAgent,
)
from aisec.core.engine import AnalysisEngine, EngineResult
from aisec.storage.models import Decision

console = Console()

# ── Display constants ─────────────────────────────────────────────────────────

DECISION_STYLE: dict[Decision, tuple[str, str]] = {
    Decision.ALLOW:          ("ALLOW",   "bold green"),
    Decision.BLOCK:          ("BLOCK",   "bold red"),
    Decision.ESCALATE:       ("ESCALATE","bold magenta"),
    Decision.PENDING_REVIEW: ("REVIEW",  "bold yellow"),
}

MAX_ACTION_LEN:      int = 28
MAX_TARGET_LEN:      int = 22
MAX_EXPLANATION_LEN: int = 55
PAUSE_BETWEEN_STEPS: float = 0.35   # seconds — realistic streaming feel


# ── Helpers ───────────────────────────────────────────────────────────────────

def _truncate(s: str, max_len: int) -> str:
    """Truncate a string to max_len characters with ellipsis."""
    s = str(s).strip()
    if len(s) <= max_len:
        return s
    return s[: max_len - 1] + "…"


def _decision_badge(decision: Decision) -> Text:
    """Return a styled Rich Text badge for the given decision."""
    label, style = DECISION_STYLE.get(
        decision,
        (decision.value, "white"),
    )
    # Pad to fixed width so columns stay aligned
    padded = label.ljust(8)
    return Text(padded, style=style)


def _risk_bar(score: float) -> str:
    """
    Convert a risk score in [0.0, 1.0] to a visual bar.

    Example: 0.75 → '███████░░░ 0.75'
    """
    filled = int(score * 10)
    empty  = 10 - filled
    bar    = "█" * filled + "░" * empty
    return f"{bar} {score:.2f}"


def _risk_style(score: float) -> str:
    """Return a Rich style string based on the risk score."""
    if score >= 0.80:
        return "bold red"
    if score >= 0.60:
        return "bold yellow"
    if score >= 0.30:
        return "yellow"
    return "green"


# ── Table builder ─────────────────────────────────────────────────────────────

def _build_table(results: list[EngineResult], scenario_label: str) -> Table:
    """
    Build a Rich Table from a list of EngineResults.

    The table is rebuilt on every update so Rich can
    animate it smoothly inside the Live context.
    """
    table = Table(
        title=f"[bold cyan]AISec Live Monitor — {scenario_label}[/bold cyan]",
        show_header=True,
        header_style="bold white on dark_blue",
        border_style="dim",
        expand=True,
        show_lines=True,
    )

    table.add_column("Time",       style="dim",         width=10, no_wrap=True)
    table.add_column("Decision",   width=10,             no_wrap=True)
    table.add_column("Agent",      style="cyan",         width=16, no_wrap=True)
    table.add_column("Action",     style="white",        width=30, no_wrap=True)
    table.add_column("Target",     style="dim white",    width=24, no_wrap=True)
    table.add_column("Risk",       width=18,             no_wrap=True)
    table.add_column("Explanation",style="dim",          min_width=30)

    for result in results:
        ts         = result.event.timestamp[11:19]   # HH:MM:SS
        decision   = _decision_badge(result.analysis.decision)
        agent      = _truncate(result.event.agent_id, 16)
        action     = _truncate(result.event.action_type, MAX_ACTION_LEN)
        target     = _truncate(result.event.target, MAX_TARGET_LEN)
        risk       = _risk_bar(result.analysis.risk_score)
        risk_style = _risk_style(result.analysis.risk_score)
        explanation = _truncate(result.analysis.explanation, MAX_EXPLANATION_LEN)

        table.add_row(
            ts,
            decision,
            agent,
            action,
            target,
            Text(risk, style=risk_style),
            explanation,
        )

    return table


def _build_summary_panel(results: list[EngineResult]) -> Panel:
    """Build a summary statistics panel shown below the event table."""
    total    = len(results)
    blocked  = sum(1 for r in results if r.analysis.decision == Decision.BLOCK)
    escalate = sum(1 for r in results if r.analysis.decision == Decision.ESCALATE)
    review   = sum(1 for r in results if r.analysis.decision == Decision.PENDING_REVIEW)
    allowed  = sum(1 for r in results if r.analysis.decision == Decision.ALLOW)

    avg_risk = (
        sum(r.analysis.risk_score for r in results) / total
        if total > 0 else 0.0
    )

    summary = (
        f"[bold white]Events:[/] {total}   "
        f"[bold red]Blocked:[/] {blocked}   "
        f"[bold magenta]Escalated:[/] {escalate}   "
        f"[bold yellow]Review:[/] {review}   "
        f"[bold green]Allowed:[/] {allowed}   "
        f"[cyan]Avg Risk:[/] {avg_risk:.3f}"
    )

    return Panel(summary, title="[bold]Session Summary[/bold]", border_style="dim")


# ── Click command ─────────────────────────────────────────────────────────────

@click.command("monitor")
@click.option(
    "--scenario",
    type=click.Choice(["trading_ai", "urban_ai", "both"], case_sensitive=False),
    default="both",
    show_default=True,
    help="Which AI scenario to simulate.",
)
@click.option(
    "--steps",
    type=click.IntRange(min=1, max=500),
    default=30,
    show_default=True,
    help="Number of agent actions to simulate.",
)
@click.option(
    "--speed",
    type=click.FloatRange(min=0.0, max=5.0),
    default=PAUSE_BETWEEN_STEPS,
    show_default=True,
    help="Seconds between events (0 = as fast as possible).",
)
def monitor_command(scenario: str, steps: int, speed: float) -> None:
    """
    Stream AI agent actions live in the terminal.

    Simulates the selected scenario and displays every action
    as it is evaluated by the AISec analysis engine.

    \\b
    Examples:
        aisec monitor
        aisec monitor --scenario trading_ai --steps 20
        aisec monitor --scenario both --speed 0.5
    """
    log_path = Path(".aisec") / "monitor_session.jsonl"
    engine   = AnalysisEngine(log_path=log_path)

    scenario_label = {
        "trading_ai": "Scenario A — Autonomous Trading AI",
        "urban_ai":   "Scenario B — Smart City Urban AI",
        "both":       "Scenario A + B — Trading AI & Urban AI",
    }[scenario]

    console.print()
    console.print(
        Text(f"  Starting live monitor — {scenario_label}", style="bold cyan")
    )
    console.print(
        Text(f"  Simulating {steps} actions | speed={speed}s | Ctrl+C to stop",
             style="dim")
    )
    console.print()

    results: list[EngineResult] = []

    # Build agent list based on scenario selection
    agents = []
    if scenario in ("trading_ai", "both"):
        agents.append(("trading", TradingAgent(engine)))
    if scenario in ("urban_ai", "both"):
        agents.append(("urban", UrbanAgent(engine)))

    # Pre-build action sequences
    action_sequence = _build_action_sequence(scenario, steps)

    try:
        with Live(
            _build_table(results, scenario_label),
            console=console,
            refresh_per_second=4,
            vertical_overflow="visible",
        ) as live:

            for agent_name, action in action_sequence:
                # Find the right agent
                agent_obj = next(
                    (a for name, a in agents if name == agent_name), None
                )
                if agent_obj is None:
                    continue

                # Execute through agent (fully intercepted by engine)
                result = agent_obj.attempt_action(action)
                results.append(result)

                # Update the live display
                from rich.console import Group
                live.update(
                    Group(
                        _build_table(results, scenario_label),
                        _build_summary_panel(results),
                    )
                )

                if speed > 0:
                    time.sleep(speed)

    except KeyboardInterrupt:
        console.print()
        console.print(Text("  Monitor stopped by user.", style="yellow"))

    # Final summary
    console.print()
    _print_final_summary(results, engine)


def _build_action_sequence(
    scenario: str, steps: int
) -> list[tuple[str, object]]:
    """
    Build a mixed action sequence for the given scenario.

    Returns a list of (agent_name, action) tuples.
    For 'both', actions alternate between trading and urban.
    """
    import random   # Non-cryptographic — simulation only, not security-sensitive

    sequence = []

    trading_pool = TRADING_SAFE * 2 + TRADING_DANGEROUS
    urban_pool   = URBAN_SAFE * 3 + URBAN_DANGEROUS

    for i in range(steps):
        if scenario == "trading_ai":
            sequence.append(("trading", random.choice(trading_pool)))
        elif scenario == "urban_ai":
            sequence.append(("urban", random.choice(urban_pool)))
        else:
            # Alternate between trading and urban
            if i % 2 == 0:
                sequence.append(("trading", random.choice(trading_pool)))
            else:
                sequence.append(("urban", random.choice(urban_pool)))

    return sequence


def _print_final_summary(
    results: list[EngineResult], engine: AnalysisEngine
) -> None:
    """Print a final security summary after the simulation ends."""
    total    = len(results)
    blocked  = sum(1 for r in results if r.analysis.decision == Decision.BLOCK)
    escalate = sum(1 for r in results if r.analysis.decision == Decision.ESCALATE)
    review   = sum(1 for r in results if r.analysis.decision == Decision.PENDING_REVIEW)
    allowed  = sum(1 for r in results if r.analysis.decision == Decision.ALLOW)

    ok, errors = engine.verify_audit_chain()
    chain_status = (
        Text("✔ INTACT", style="bold green")
        if ok
        else Text(f"✘ BROKEN ({len(errors)} errors)", style="bold red")
    )

    console.print(Panel(
        f"[bold white]Total events:[/]    {total}\n"
        f"[bold red]Blocked:[/]          {blocked}\n"
        f"[bold magenta]Escalated:[/]       {escalate}\n"
        f"[bold yellow]Under review:[/]    {review}\n"
        f"[bold green]Allowed:[/]          {allowed}\n"
        f"[cyan]Audit chain:[/]      ",
        title="[bold cyan]AISec Session Complete[/bold cyan]",
        border_style="cyan",
    ))
    console.print(f"  Audit chain integrity: ", end="")
    console.print(chain_status)
    console.print()
