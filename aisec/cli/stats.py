"""
AISec statistics dashboard CLI command.

Displays a comprehensive security statistics overview
from the audit log — decision distribution, risk trends,
top agents, and audit chain integrity status.

Security considerations:
    - Dashboard is strictly read-only.
    - All data is read from the tamper-evident audit log.
    - No statistics can be modified through this interface.
    - If the audit chain is broken, a critical alert is shown.

Usage:
    aisec stats
    aisec stats --log .aisec/audit.jsonl
    aisec stats --tail 50
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import click
from rich.columns import Columns
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from aisec.storage.audit import AuditLogger, DEFAULT_LOG_PATH
from aisec.storage.models import Decision

console = Console()


# ── Display helpers ───────────────────────────────────────────────────────────


def _bar(value: int, total: int, width: int = 20) -> str:
    """
    Render a proportional bar chart segment.

    Example: _bar(7, 10, 20) → '██████████████░░░░░░'
    """
    if total == 0:
        return "░" * width
    filled = int((value / total) * width)
    empty = width - filled
    return "█" * filled + "░" * empty


def _risk_level(score: float) -> tuple[str, str]:
    """Return (label, style) for a risk score."""
    if score >= 0.80:
        return "CRITICAL", "bold red"
    if score >= 0.60:
        return "HIGH", "bold yellow"
    if score >= 0.30:
        return "MEDIUM", "yellow"
    return "LOW", "green"


# ── Statistics computation ────────────────────────────────────────────────────


def _compute_stats(entries: list) -> dict:
    """
    Compute all statistics from a list of AuditLogEntry objects.

    Returns a dictionary with all metrics needed by the display
    functions. Separates analysis entries from analyst decisions.
    """
    analysis_entries = [e for e in entries if e.record_type == "analysis"]
    analyst_entries = [e for e in entries if e.record_type == "analyst_decision"]

    total = len(analysis_entries)

    if total == 0:
        return {"total": 0}

    # Decision distribution
    decision_counts: Counter[str] = Counter()
    for entry in analysis_entries:
        decision_counts[entry.payload.get("decision", "UNKNOWN")] += 1

    # Risk score statistics
    scores = [
        entry.payload.get("risk_score", 0.0)
        for entry in analysis_entries
        if isinstance(entry.payload.get("risk_score"), (int, float))
    ]
    avg_risk = sum(scores) / len(scores) if scores else 0.0
    max_risk = max(scores) if scores else 0.0
    min_risk = min(scores) if scores else 0.0

    # Risk distribution buckets
    risk_buckets = {
        "critical (≥0.80)": sum(1 for s in scores if s >= 0.80),
        "high (0.60-0.80)": sum(1 for s in scores if 0.60 <= s < 0.80),
        "medium (0.30-0.60)": sum(1 for s in scores if 0.30 <= s < 0.60),
        "low (<0.30)": sum(1 for s in scores if s < 0.30),
    }

    # Agent activity
    agent_counts: Counter[str] = Counter()
    for entry in analysis_entries:
        agent = entry.payload.get("agent_id", "unknown")
        agent_counts[agent] += 1

    # Top blocked actions
    blocked_actions: Counter[str] = Counter()
    for entry in analysis_entries:
        if entry.payload.get("decision") in ("BLOCK", "ESCALATE"):
            action = entry.payload.get("action_type", "unknown")
            blocked_actions[action] += 1

    # Rule hits
    rule_hits: Counter[str] = Counter()
    for entry in analysis_entries:
        for rule_id in entry.payload.get("rule_hits", []):
            rule_hits[rule_id] += 1

    # Analyst activity
    analyst_counts: Counter[str] = Counter()
    for entry in analyst_entries:
        decision = entry.payload.get("analyst_decision", "unknown")
        analyst_counts[decision] += 1

    # Scenario breakdown
    scenario_counts: Counter[str] = Counter()
    for entry in analysis_entries:
        scenario = entry.payload.get("scenario", "unknown")
        scenario_counts[scenario] += 1

    return {
        "total": total,
        "decision_counts": decision_counts,
        "avg_risk": avg_risk,
        "max_risk": max_risk,
        "min_risk": min_risk,
        "risk_buckets": risk_buckets,
        "agent_counts": agent_counts,
        "blocked_actions": blocked_actions,
        "rule_hits": rule_hits,
        "analyst_counts": analyst_counts,
        "scenario_counts": scenario_counts,
        "analyst_total": len(analyst_entries),
    }


# ── Panels ────────────────────────────────────────────────────────────────────


def _decision_panel(stats: dict) -> Panel:
    """Render the decision distribution panel."""
    total = stats["total"]
    counts = stats["decision_counts"]

    lines = []
    order = [
        (Decision.BLOCK.value, "bold red"),
        (Decision.ESCALATE.value, "bold magenta"),
        (Decision.PENDING_REVIEW.value, "bold yellow"),
        (Decision.ALLOW.value, "bold green"),
    ]

    for decision_val, style in order:
        count = counts.get(decision_val, 0)
        pct = (count / total * 100) if total > 0 else 0
        bar = _bar(count, total, width=24)
        lines.append(
            f"[{style}]{decision_val:<16}[/{style}] "
            f"[cyan]{bar}[/cyan] "
            f"[white]{count:>4}[/white] "
            f"[dim]({pct:5.1f}%)[/dim]"
        )

    content = "\n".join(lines)
    content += f"\n\n[bold white]Total events:[/bold white] {total}"

    return Panel(
        content,
        title="[bold]Decision Distribution[/bold]",
        border_style="cyan",
    )


def _risk_panel(stats: dict) -> Panel:
    """Render the risk score statistics panel."""
    avg = stats["avg_risk"]
    high = stats["max_risk"]
    low = stats["min_risk"]
    buckets = stats["risk_buckets"]
    total = stats["total"]

    avg_label, avg_style = _risk_level(avg)
    max_label, max_style = _risk_level(high)

    lines = [
        f"[bold white]Average risk:[/]  [{avg_style}]{avg:.4f} ({avg_label})[/{avg_style}]",
        f"[bold white]Highest risk:[/]  [{max_style}]{high:.4f} ({max_label})[/{max_style}]",
        f"[bold white]Lowest risk:[/]   [green]{low:.4f}[/green]",
        "",
        "[bold white]Risk distribution:[/]",
    ]

    bucket_styles = {
        "critical (≥0.80)": "bold red",
        "high (0.60-0.80)": "bold yellow",
        "medium (0.30-0.60)": "yellow",
        "low (<0.30)": "green",
    }

    for label, count in buckets.items():
        style = bucket_styles.get(label, "white")
        bar = _bar(count, total, width=16)
        lines.append(
            f"  [{style}]{label:<22}[/{style}] "
            f"[cyan]{bar}[/cyan] [white]{count}[/white]"
        )

    return Panel(
        "\n".join(lines),
        title="[bold]Risk Score Analysis[/bold]",
        border_style="yellow",
    )


def _agent_panel(stats: dict) -> Panel:
    """Render the agent activity panel."""
    total = stats["total"]
    agents = stats["agent_counts"].most_common(10)

    if not agents:
        return Panel("No agent data.", title="[bold]Agent Activity[/bold]")

    lines = []
    for agent, count in agents:
        bar = _bar(count, total, width=20)
        pct = (count / total * 100) if total > 0 else 0
        lines.append(
            f"[cyan]{agent:<22}[/cyan] "
            f"[dim]{bar}[/dim] "
            f"[white]{count:>4}[/white] [dim]({pct:.1f}%)[/dim]"
        )

    return Panel(
        "\n".join(lines),
        title="[bold]Agent Activity[/bold]",
        border_style="cyan",
    )


def _rules_panel(stats: dict) -> Panel:
    """Render the top fired rules panel."""
    rules = stats["rule_hits"].most_common(10)

    if not rules:
        return Panel(
            "[dim]No rules fired in this session.[/dim]",
            title="[bold]Top Fired Rules[/bold]",
            border_style="magenta",
        )

    total_hits = sum(stats["rule_hits"].values())
    lines = []
    for rule_id, count in rules:
        bar = _bar(count, total_hits, width=16)
        lines.append(
            f"[magenta]{rule_id:<14}[/magenta] "
            f"[dim]{bar}[/dim] "
            f"[white]{count}[/white] hits"
        )

    return Panel(
        "\n".join(lines),
        title="[bold]Top Fired Rules[/bold]",
        border_style="magenta",
    )


def _blocked_actions_panel(stats: dict) -> Panel:
    """Render the most blocked action types panel."""
    actions = stats["blocked_actions"].most_common(8)

    if not actions:
        return Panel(
            "[green]No blocked actions recorded.[/green]",
            title="[bold]Most Blocked Actions[/bold]",
            border_style="red",
        )

    total = sum(stats["blocked_actions"].values())
    lines = []
    for action, count in actions:
        bar = _bar(count, total, width=16)
        lines.append(
            f"[red]{action:<30}[/red] " f"[dim]{bar}[/dim] " f"[white]{count}[/white]"
        )

    return Panel(
        "\n".join(lines),
        title="[bold]Most Blocked Actions[/bold]",
        border_style="red",
    )


def _scenario_panel(stats: dict) -> Panel:
    """Render the scenario breakdown panel."""
    total = stats["total"]
    scenarios = stats["scenario_counts"]
    analyst = stats["analyst_counts"]
    analyst_t = stats["analyst_total"]

    scenario_styles = {
        "trading_ai": "bold cyan",
        "urban_ai": "bold green",
        "unknown": "dim",
    }

    lines = ["[bold white]Events by scenario:[/bold white]"]
    for scenario, count in scenarios.most_common():
        style = scenario_styles.get(scenario, "white")
        bar = _bar(count, total, width=16)
        lines.append(
            f"  [{style}]{scenario:<16}[/{style}] "
            f"[dim]{bar}[/dim] [white]{count}[/white]"
        )

    if analyst_t > 0:
        lines.append("")
        lines.append(f"[bold white]Analyst decisions:[/bold white] {analyst_t} total")
        for decision, count in analyst.most_common():
            lines.append(f"  [yellow]{decision:<12}[/yellow] {count}")

    return Panel(
        "\n".join(lines),
        title="[bold]Scenario & Analyst Breakdown[/bold]",
        border_style="green",
    )


def _integrity_panel(ok: bool, errors: list[str], count: int) -> Panel:
    """Render the audit chain integrity status panel."""
    if ok:
        content = (
            f"[bold green]✔ CHAIN INTACT[/bold green]\n"
            f"[white]{count} entries verified — no tampering detected.[/white]"
        )
        border = "green"
    else:
        error_lines = "\n".join(f"  [red]• {e}[/red]" for e in errors[:5])
        content = (
            f"[bold red]✘ CHAIN BROKEN — {len(errors)} error(s)[/bold red]\n"
            f"{error_lines}\n"
            f"[bold red]Audit log may have been tampered with.[/bold red]"
        )
        border = "red"

    return Panel(
        content,
        title="[bold]Audit Chain Integrity[/bold]",
        border_style=border,
    )


# ── Click command ─────────────────────────────────────────────────────────────


@click.command("stats")
@click.option(
    "--log",
    type=click.Path(exists=False, path_type=Path),
    default=DEFAULT_LOG_PATH,
    show_default=True,
    help="Path to the audit log file.",
)
@click.option(
    "--tail",
    type=click.IntRange(min=1, max=100_000),
    default=None,
    help="Analyse only the last N log entries.",
)
def stats_command(log: Path, tail: int | None) -> None:
    """
    Display a security statistics dashboard from the audit log.

    Reads the AISec audit log and renders a comprehensive overview
    including decision distribution, risk analysis, agent activity,
    rule hits, and audit chain integrity verification.

    \\b
    Examples:
        aisec stats
        aisec stats --tail 100
        aisec stats --log .aisec/soc_session.jsonl
    """
    console.print()

    # Load the audit log
    if not log.exists():
        console.print(
            Panel(
                f"[yellow]Audit log not found at: {log}[/yellow]\n\n"
                "[dim]Run 'aisec monitor' or 'aisec soc' first to generate data.[/dim]",
                title="[bold]AISec Statistics[/bold]",
                border_style="yellow",
            )
        )
        return

    logger = AuditLogger(log_path=log)
    entries = logger.get_all()

    if tail is not None:
        entries = entries[-tail:]

    ok, errors = logger.verify_chain()
    count = len(entries)

    if count == 0:
        console.print(
            Panel(
                "[yellow]Audit log is empty — no data to display.[/yellow]",
                title="[bold]AISec Statistics[/bold]",
                border_style="yellow",
            )
        )
        return

    stats = _compute_stats(entries)

    # ── Render dashboard ──────────────────────────────────────────────────────

    console.print(
        Text(
            f"  AISec Security Dashboard — {count} audit entries",
            style="bold cyan",
        )
    )
    console.print(Text(f"  Log: {log}", style="dim"))
    console.print()

    if stats["total"] == 0:
        console.print(
            Panel(
                "[dim]No analysis events found in this log.[/dim]",
                border_style="dim",
            )
        )
        return

    # Row 1: Decision distribution + Risk analysis
    console.print(
        Columns(
            [
                _decision_panel(stats),
                _risk_panel(stats),
            ]
        )
    )

    # Row 2: Agent activity + Scenario breakdown
    console.print(
        Columns(
            [
                _agent_panel(stats),
                _scenario_panel(stats),
            ]
        )
    )

    # Row 3: Blocked actions + Top rules
    console.print(
        Columns(
            [
                _blocked_actions_panel(stats),
                _rules_panel(stats),
            ]
        )
    )

    # Row 4: Audit chain integrity — full width
    console.print(_integrity_panel(ok, errors, count))
    console.print()
