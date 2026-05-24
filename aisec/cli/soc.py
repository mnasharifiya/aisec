"""
AISec SOC console — interactive analyst environment.

Provides a Metasploit-style interactive console where security
analysts can review flagged AI actions, approve or block them,
escalate incidents, and inspect the audit trail.

Security considerations:
    - Every analyst decision is written to the audit log.
    - Critical actions require explicit typed confirmation.
    - The console cannot modify past audit entries.
    - Session identity is set at startup and cannot be changed.
    - All input is sanitised before processing.

Usage:
    aisec soc
    aisec soc --scenario trading_ai --steps 20
"""

from __future__ import annotations

import time
from pathlib import Path

import click
from prompt_toolkit import PromptSession
from prompt_toolkit.formatted_text import HTML
from prompt_toolkit.styles import Style
from rich.console import Console
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
from aisec.storage.audit import AuditLogger
from aisec.storage.models import Decision

console = Console()

# Rate limiting imports and state
import time as _time

# Minimum seconds between irreversible analyst decisions.
# Prevents automated approval of events faster than human review.
_MIN_DECISION_INTERVAL: float = 1.0
_last_decision_time: float = 0.0

# ── SOC queue ─────────────────────────────────────────────────────────────────


class SOCQueue:
    """
    In-memory queue of events requiring analyst review.

    Tracks pending, approved, blocked, and escalated events.
    All analyst decisions are written to the audit log.
    """

    def __init__(self, audit_logger: AuditLogger) -> None:
        self._pending: list[EngineResult] = []
        self._resolved: list[tuple[EngineResult, str, str]] = []
        self._logger = audit_logger

    def submit(self, result: EngineResult) -> None:
        """Add a flagged event to the review queue."""
        self._pending.append(result)

    def pending(self) -> list[EngineResult]:
        """Return all unresolved events."""
        return list(self._pending)

    def pending_count(self) -> int:
        return len(self._pending)

    def resolve(
        self,
        result: EngineResult,
        analyst_decision: str,
        analyst_id: str,
        reason: str = "",
    ) -> None:
        """
        Record an analyst decision and move event out of pending queue.

        Args:
            result:           The EngineResult being resolved.
            analyst_decision: "approve", "block", or "escalate".
            analyst_id:       Identity of the analyst making the decision.
            reason:           Optional reason for the decision.
        """
        # Validate decision value before logging
        valid = {"approve", "block", "escalate"}
        if analyst_decision not in valid:
            raise ValueError(
                f"Invalid analyst decision '{analyst_decision}'. "
                f"Must be one of: {valid}"
            )

        # Remove from pending
        self._pending = [
            r for r in self._pending if r.event.event_id != result.event.event_id
        ]

        # Record resolution
        self._resolved.append((result, analyst_decision, analyst_id))

        # Write to tamper-evident audit log
        self._logger.log(
            record_type="analyst_decision",
            record_id=result.event.event_id,
            payload={
                "analyst_id": analyst_id,
                "analyst_decision": analyst_decision,
                "reason": reason,
                "action_type": result.event.action_type,
                "agent_id": result.event.agent_id,
                "original_decision": result.analysis.decision.value,
                "risk_score": result.analysis.risk_score,
            },
        )

    def resolved_count(self) -> int:
        return len(self._resolved)

    def verify_chain(self) -> tuple[bool, list[str]]:
        """Verify the audit log hash chain integrity."""
        return self._logger.verify_chain()

    def get_last_entries(self, n: int = 10) -> list:
        """Return the last n audit log entries."""
        return self._logger.get_last(max(1, min(n, 100)))

    def stats(self) -> dict[str, int]:
        """Return counts of each analyst decision type."""
        counts: dict[str, int] = {"approve": 0, "block": 0, "escalate": 0}
        for _, decision, _ in self._resolved:
            counts[decision] = counts.get(decision, 0) + 1
        return counts


# ── Display helpers ───────────────────────────────────────────────────────────

_DECISION_STYLE: dict[Decision, str] = {
    Decision.ALLOW: "bold green",
    Decision.BLOCK: "bold red",
    Decision.ESCALATE: "bold magenta",
    Decision.PENDING_REVIEW: "bold yellow",
}


def _print_queue_table(queue: SOCQueue) -> None:
    """Render the pending review queue as a Rich table."""
    pending = queue.pending()

    if not pending:
        console.print(
            Panel(
                "[bold green]No events pending review.[/bold green]",
                title="[bold]SOC Review Queue[/bold]",
                border_style="green",
            )
        )
        return

    table = Table(
        title=f"[bold yellow]SOC Review Queue — {len(pending)} pending[/bold yellow]",
        show_header=True,
        header_style="bold white on dark_red",
        border_style="yellow",
        expand=True,
        show_lines=True,
    )

    table.add_column("#", width=4, no_wrap=True)
    table.add_column("Agent", width=16, no_wrap=True, style="cyan")
    table.add_column("Action", width=28, no_wrap=True)
    table.add_column("Target", width=22, no_wrap=True, style="dim white")
    table.add_column("Decision", width=12, no_wrap=True)
    table.add_column("Risk", width=8, no_wrap=True)
    table.add_column("Reason", min_width=30)

    for i, result in enumerate(pending, start=1):
        decision = result.analysis.decision
        style = _DECISION_STYLE.get(decision, "white")
        reason = result.analysis.explanation[:55]
        if len(result.analysis.explanation) > 55:
            reason += "…"

        table.add_row(
            str(i),
            result.event.agent_id[:16],
            result.event.action_type[:28],
            result.event.target[:22],
            Text(decision.value, style=style),
            f"{result.analysis.risk_score:.2f}",
            reason,
        )

    console.print(table)


def _print_event_detail(result: EngineResult, index: int) -> None:
    """Print full details of a single event for analyst review."""
    decision = result.analysis.decision
    style = _DECISION_STYLE.get(decision, "white")

    console.print()
    console.print(
        Panel(
            f"[bold]Event ID:[/]        {result.event.event_id}\n"
            f"[bold]Timestamp:[/]       {result.event.timestamp}\n"
            f"[bold]Agent:[/]           {result.event.agent_id}\n"
            f"[bold]Scenario:[/]        {result.event.scenario.value}\n"
            f"[bold]Action:[/]          {result.event.action_type}\n"
            f"[bold]Target:[/]          {result.event.target}\n"
            f"[bold]Payload:[/]         {result.event.raw_payload}\n"
            f"[bold]Risk score:[/]      {result.analysis.risk_score:.4f}\n"
            f"[bold]Decision:[/]        [{style}]{decision.value}[/{style}]\n"
            f"[bold]Rules fired:[/]     {result.analysis.rule_hits or 'none'}\n"
            f"[bold]Explanation:[/]     {result.analysis.explanation}",
            title=f"[bold yellow]Event #{index} — Analyst Review[/bold yellow]",
            border_style="yellow",
        )
    )
    console.print()


def _print_soc_header(queue: SOCQueue, analyst_id: str) -> None:
    """Print the SOC console header panel."""
    stats = queue.stats()
    console.print(
        Panel(
            f"[bold white]Analyst:[/]       {analyst_id}\n"
            f"[bold yellow]Pending:[/]       {queue.pending_count()}\n"
            f"[bold white]Resolved:[/]       {queue.resolved_count()}   "
            f"([green]approved: {stats['approve']}[/green]  "
            f"[red]blocked: {stats['block']}[/red]  "
            f"[magenta]escalated: {stats['escalate']}[/magenta])",
            title="[bold cyan]AISec SOC Console — Analyst Mode[/bold cyan]",
            border_style="cyan",
        )
    )


def _print_help() -> None:
    """Print available SOC console commands."""
    console.print(
        Panel(
            "[bold cyan]queue[/]              Show all pending events\n"
            "[bold cyan]review <N>[/]         Review event number N in detail\n"
            "[bold cyan]approve <N>[/]        Approve event N — allow action to proceed\n"
            "[bold cyan]block <N>[/]          Block event N — action is denied\n"
            "[bold cyan]escalate <N>[/]       Escalate event N to senior analyst\n"
            "[bold cyan]stats[/]              Show session statistics\n"
            "[bold cyan]verify[/]             Verify audit log hash chain integrity\n"
            "[bold cyan]logs [N][/]           Show last N audit log entries (default 10)\n"
            "[bold cyan]help[/]               Show this help message\n"
            "[bold cyan]exit[/]               Exit the SOC console",
            title="[bold]Available Commands[/bold]",
            border_style="dim",
        )
    )


# ── Command parser ────────────────────────────────────────────────────────────


def _parse_index(parts: list[str], queue: SOCQueue) -> int | None:
    """
    Parse and validate an event index from command parts.

    Returns the 0-based index into the pending queue,
    or None if the input is invalid.
    """
    if len(parts) < 2:
        console.print(Text("  Usage: <command> <event number>", style="red"))
        return None

    try:
        n = int(parts[1])
    except ValueError:
        console.print(Text(f"  '{parts[1]}' is not a number.", style="red"))
        return None

    pending = queue.pending()
    if n < 1 or n > len(pending):
        console.print(
            Text(
                f"  Event #{n} does not exist. "
                f"Queue has {len(pending)} pending events.",
                style="red",
            )
        )
        return None

    return n - 1  # Convert to 0-based index


def _confirm_action(action_name: str, event_num: int) -> bool:
    """
    Require explicit typed confirmation for irreversible analyst decisions.

    The analyst must type 'CONFIRM <ACTION>' to proceed.
    Any other input cancels the action.
    """
    expected = f"CONFIRM {action_name.upper()}"
    console.print(
        Text(
            f"  Type '{expected}' to confirm, or anything else to cancel:",
            style="yellow",
        )
    )
    try:
        response = input("  > ").strip()
    except (EOFError, KeyboardInterrupt):
        return False

    if response == expected:
        return True

    console.print(Text("  Action cancelled.", style="dim"))
    return False


def _check_rate_limit() -> bool:
    """
    Enforce minimum time between analyst decisions.

    Returns True if the decision is allowed, False if too fast.
    This prevents automated scripts from approving events
    faster than a human could reasonably review them.
    """
    global _last_decision_time
    now = _time.monotonic()
    elapsed = now - _last_decision_time
    if elapsed < _MIN_DECISION_INTERVAL:
        remaining = _MIN_DECISION_INTERVAL - elapsed
        console.print(
            Text(
                f"  ⚠ Decision rate limit — wait {remaining:.1f}s.",
                style="yellow",
            )
        )
        return False
    _last_decision_time = now
    return True


# ── SOC session ───────────────────────────────────────────────────────────────


def _run_soc_session(queue: SOCQueue, analyst_id: str) -> None:
    """
    Run the interactive SOC analyst console session.

    Accepts typed commands in a loop until the analyst exits.
    Uses prompt_toolkit for a polished readline-style prompt.
    """
    prompt_style = Style.from_dict(
        {
            "prompt": "ansicyan bold",
        }
    )

    session: PromptSession[str] = PromptSession()

    console.print()
    _print_soc_header(queue, analyst_id)
    console.print()
    console.print(
        Text("  Type 'help' for available commands, 'exit' to quit.", style="dim")
    )
    console.print()

    while True:
        try:
            raw = session.prompt(
                HTML("<prompt>soc</prompt>> "),
                style=prompt_style,
            ).strip()
        except (KeyboardInterrupt, EOFError):
            break

        if not raw:
            continue

        # Sanitise — truncate to prevent absurdly long inputs
        raw = raw[:256]
        parts = raw.split()
        cmd = parts[0].lower()

        # ── Commands ──────────────────────────────────────────────────────────

        if cmd == "exit":
            break

        elif cmd == "help":
            _print_help()

        elif cmd == "queue":
            console.print()
            _print_queue_table(queue)
            console.print()

        elif cmd == "review":
            idx = _parse_index(parts, queue)
            if idx is not None:
                _print_event_detail(queue.pending()[idx], idx + 1)

        elif cmd == "approve":
            idx = _parse_index(parts, queue)
            if idx is not None:
                if not _check_rate_limit():
                    continue
                result = queue.pending()[idx]
                _print_event_detail(result, idx + 1)
                if _confirm_action("approve", idx + 1):
                    queue.resolve(
                        result,
                        "approve",
                        analyst_id,
                        reason="Analyst approved after review",
                    )
                    console.print(
                        Text(
                            f"  ✔ Event #{idx+1} approved and logged.",
                            style="bold green",
                        )
                    )

        elif cmd == "block":
            idx = _parse_index(parts, queue)
            if idx is not None:
                if not _check_rate_limit():
                    continue
                result = queue.pending()[idx]
                _print_event_detail(result, idx + 1)
                if _confirm_action("block", idx + 1):
                    queue.resolve(
                        result,
                        "block",
                        analyst_id,
                        reason="Analyst blocked after review",
                    )
                    console.print(
                        Text(
                            f"  ✘ Event #{idx+1} blocked and logged.", style="bold red"
                        )
                    )

        elif cmd == "escalate":
            idx = _parse_index(parts, queue)
            if idx is not None:
                if not _check_rate_limit():
                    continue
                result = queue.pending()[idx]
                _print_event_detail(result, idx + 1)
                if _confirm_action("escalate", idx + 1):
                    queue.resolve(
                        result,
                        "escalate",
                        analyst_id,
                        reason="Escalated to senior analyst",
                    )
                    console.print(
                        Text(
                            f"  ⬆ Event #{idx+1} escalated and logged.",
                            style="bold magenta",
                        )
                    )

        elif cmd == "stats":
            console.print()
            _print_soc_header(queue, analyst_id)
            console.print()

        elif cmd == "verify":
            console.print()
            console.print(Text("  Verifying audit chain...", style="dim"))
            # Use the public wrapper on SOCQueue instead of accessing private logger
            ok, errors = queue.verify_chain()
            if ok:
                console.print(
                    Text(
                        "  ✔ Audit chain INTACT — no tampering detected.",
                        style="bold green",
                    )
                )
            else:
                console.print(
                    Text(
                        f"  ✘ Audit chain BROKEN — {len(errors)} error(s):",
                        style="bold red",
                    )
                )
                for err in errors:
                    console.print(Text(f"    • {err}", style="red"))
            console.print()

        elif cmd == "logs":
            n = 10
            if len(parts) >= 2:
                try:
                    n = max(1, min(int(parts[1]), 100))
                except ValueError:
                    pass
            # Use the public wrapper on SOCQueue instead of accessing private logger
            entries = queue.get_last_entries(n)
            console.print()
            console.print(
                Text(f"  Last {len(entries)} audit log entries:", style="dim")
            )
            for entry in entries:
                ts = entry.timestamp[11:19]
                payload = entry.payload
                action = payload.get(
                    "action_type", payload.get("analyst_decision", "?")
                )
                decision = payload.get("decision", payload.get("analyst_decision", "?"))
                console.print(
                    Text(f"  [{ts}] ", style="dim")
                    + Text(f"{entry.record_type:<18} ", style="cyan")
                    + Text(f"{action:<28} ", style="white")
                    + Text(f"{decision}", style="yellow")
                )
            console.print()

        else:
            console.print(
                Text(
                    f"  Unknown command: '{cmd}'. Type 'help' for commands.",
                    style="red",
                )
            )


# ── Click command ─────────────────────────────────────────────────────────────


@click.command("soc")
@click.option(
    "--analyst",
    default="analyst_01",
    show_default=True,
    help="Analyst identity recorded in the audit log.",
)
@click.option(
    "--scenario",
    type=click.Choice(["trading_ai", "urban_ai", "both"], case_sensitive=False),
    default="both",
    show_default=True,
    help="Scenario to pre-populate the review queue with.",
)
@click.option(
    "--steps",
    type=click.IntRange(min=1, max=200),
    default=20,
    show_default=True,
    help="Number of agent actions to simulate before entering console.",
)
def soc_command(analyst: str, scenario: str, steps: int) -> None:
    """
    Enter the interactive SOC analyst console.

    Simulates AI agent actions, populates the review queue
    with flagged events, then opens an interactive console
    where analysts can approve, block, or escalate them.

    \\b
    Examples:
        aisec soc
        aisec soc --analyst senior_analyst --scenario trading_ai
        aisec soc --steps 30 --scenario urban_ai
    """
    # Sanitise analyst ID — only allow safe characters
    analyst = "".join(c for c in analyst if c.isalnum() or c in "-_")[:32]
    if len(analyst) < 3:
        analyst = "analyst_01"

    log_path = Path(".aisec") / "soc_session.jsonl"
    engine = AnalysisEngine(log_path=log_path)
    logger = AuditLogger(log_path=log_path)
    queue = SOCQueue(audit_logger=logger)

    console.print()
    console.print(
        Text("  Simulating agent actions — populating SOC queue...", style="dim")
    )

    # Run simulation and collect flagged events
    import random  # Non-cryptographic — simulation only, not security-sensitive

    trading_pool = TRADING_SAFE * 2 + TRADING_DANGEROUS
    urban_pool = URBAN_SAFE * 3 + URBAN_DANGEROUS

    trading_agent = TradingAgent(engine)
    urban_agent = UrbanAgent(engine)

    flagged = 0
    for i in range(steps):
        if scenario == "trading_ai":
            action = random.choice(trading_pool)
            result = trading_agent.attempt_action(action)
        elif scenario == "urban_ai":
            action = random.choice(urban_pool)
            result = urban_agent.attempt_action(action)
        else:
            if i % 2 == 0:
                action = random.choice(trading_pool)
                result = trading_agent.attempt_action(action)
            else:
                action = random.choice(urban_pool)
                result = urban_agent.attempt_action(action)

        # Queue events that need human review
        if result.analysis.decision == Decision.PENDING_REVIEW:
            queue.submit(result)
            flagged += 1

    console.print(
        Text(
            f"  Simulation complete — {steps} actions, "
            f"{flagged} events flagged for review.",
            style="dim",
        )
    )

    # Enter interactive console
    _run_soc_session(queue, analyst)

    # Exit summary
    console.print()
    stats = queue.stats()
    console.print(
        Panel(
            f"[bold white]Session complete[/bold white]\n"
            f"Analyst:          {analyst}\n"
            f"Events reviewed:  {queue.resolved_count()}\n"
            f"Approved:         [green]{stats['approve']}[/green]\n"
            f"Blocked:          [red]{stats['block']}[/red]\n"
            f"Escalated:        [magenta]{stats['escalate']}[/magenta]\n"
            f"Still pending:    [yellow]{queue.pending_count()}[/yellow]",
            title="[bold cyan]SOC Session Summary[/bold cyan]",
            border_style="cyan",
        )
    )
    console.print()
