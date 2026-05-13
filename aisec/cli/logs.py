"""
AISec audit log CLI command.

Provides commands to inspect and verify the tamper-evident
audit log directly from the terminal.

Security considerations:
    - All operations are strictly read-only.
    - verify subcommand re-computes every hash in the chain.
    - Any broken link is reported with exact entry location.
    - Export produces a plain JSONL file — no hash modification.

Usage:
    aisec logs
    aisec logs --tail 20
    aisec logs --verify
    aisec logs --export audit_export.jsonl
"""

from __future__ import annotations

import json
from pathlib import Path

import click
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from aisec.storage.audit import AuditLogger, DEFAULT_LOG_PATH

console = Console()


# ── Display helpers ───────────────────────────────────────────────────────────

def _truncate(s: str, n: int) -> str:
    """Truncate string to n characters with ellipsis."""
    s = str(s)
    return s if len(s) <= n else s[: n - 1] + "…"


def _payload_summary(payload: dict) -> str:
    """
    Produce a concise one-line summary of an audit log payload.

    Shows the most relevant fields depending on record type.
    """
    if not payload:
        return "[empty]"

    # Analysis entry
    if "action_type" in payload and "decision" in payload:
        action   = payload.get("action_type", "?")
        decision = payload.get("decision", "?")
        agent    = payload.get("agent_id", "?")
        risk     = payload.get("risk_score", 0.0)
        return f"{agent} | {action} | {decision} | risk={risk:.3f}"

    # Analyst decision entry
    if "analyst_decision" in payload:
        analyst  = payload.get("analyst_id", "?")
        decision = payload.get("analyst_decision", "?")
        action   = payload.get("action_type", "?")
        return f"analyst={analyst} | {decision} | {action}"

    # Generic fallback — show first two key/value pairs
    items = list(payload.items())[:2]
    return "  ".join(f"{k}={v}" for k, v in items)


def _decision_style(payload: dict) -> str:
    """Return a Rich style string based on the decision in the payload."""
    decision = payload.get("decision", payload.get("analyst_decision", ""))
    styles = {
        "BLOCK":          "bold red",
        "ESCALATE":       "bold magenta",
        "PENDING_REVIEW": "bold yellow",
        "ALLOW":          "green",
        "block":          "bold red",
        "approve":        "green",
        "escalate":       "bold magenta",
    }
    return styles.get(decision, "white")


# ── Click command ─────────────────────────────────────────────────────────────

@click.command("logs")
@click.option(
    "--log",
    type=click.Path(exists=False, path_type=Path),
    default=DEFAULT_LOG_PATH,
    show_default=True,
    help="Path to the audit log file.",
)
@click.option(
    "--tail",
    type=click.IntRange(min=1, max=10_000),
    default=20,
    show_default=True,
    help="Number of recent entries to display.",
)
@click.option(
    "--verify",
    "do_verify",
    is_flag=True,
    default=False,
    help="Verify the full hash chain integrity.",
)
@click.option(
    "--export",
    type=click.Path(path_type=Path),
    default=None,
    help="Export all log entries to a JSONL file.",
)
@click.option(
    "--filter-decision",
    "filter_decision",
    type=click.Choice(
        ["ALLOW", "BLOCK", "ESCALATE", "PENDING_REVIEW"],
        case_sensitive=False,
    ),
    default=None,
    help="Show only entries with the given decision.",
)
def logs_command(
    log: Path,
    tail: int,
    do_verify: bool,
    export: Path | None,
    filter_decision: str | None,
) -> None:
    """
    Inspect and verify the AISec tamper-evident audit log.

    Displays recent log entries in a readable table format.
    Use --verify to check the full hash chain integrity.
    Use --export to save a copy of the log to a new file.

    \\b
    Examples:
        aisec logs
        aisec logs --tail 50
        aisec logs --verify
        aisec logs --filter-decision BLOCK
        aisec logs --export backup.jsonl
    """
    console.print()

    # ── Check log exists ──────────────────────────────────────────────────────
    if not log.exists():
        console.print(Panel(
            f"[yellow]Audit log not found: {log}[/yellow]\n\n"
            "[dim]Run 'aisec monitor' or 'aisec soc' to generate audit data.[/dim]",
            title="[bold]AISec Audit Log[/bold]",
            border_style="yellow",
        ))
        return

    logger  = AuditLogger(log_path=log)
    entries = logger.get_all()
    total   = len(entries)

    if total == 0:
        console.print(Panel(
            "[yellow]Audit log is empty.[/yellow]",
            title="[bold]AISec Audit Log[/bold]",
            border_style="yellow",
        ))
        return

    # ── Export ────────────────────────────────────────────────────────────────
    if export is not None:
        _export_log(entries, export)
        return

    # ── Verify chain ──────────────────────────────────────────────────────────
    if do_verify:
        _verify_chain(logger, total)
        return

    # ── Display entries ───────────────────────────────────────────────────────
    display_entries = entries[-tail:]

    # Apply decision filter if requested
    if filter_decision:
        fd = filter_decision.upper()
        display_entries = [
            e for e in display_entries
            if e.payload.get("decision", "").upper() == fd
            or e.payload.get("analyst_decision", "").upper() == fd
        ]

    console.print(
        Text(
            f"  AISec Audit Log — {total} total entries "
            f"(showing {len(display_entries)})",
            style="bold cyan",
        )
    )
    console.print(Text(f"  Log: {log}", style="dim"))
    console.print()

    _display_entries_table(display_entries)

    # Quick chain status at the bottom
    ok, errors = logger.verify_chain()
    if ok:
        console.print(
            Text(
                f"  ✔ Chain intact — {total} entries verified.",
                style="bold green",
            )
        )
    else:
        console.print(
            Text(
                f"  ✘ Chain BROKEN — {len(errors)} error(s). "
                "Run 'aisec logs --verify' for details.",
                style="bold red",
            )
        )
    console.print()


def _display_entries_table(entries: list) -> None:
    """Render audit log entries as a Rich table."""
    if not entries:
        console.print(Text("  No entries match the current filter.", style="dim"))
        return

    table = Table(
        show_header=True,
        header_style="bold white on dark_blue",
        border_style="dim",
        expand=True,
        show_lines=False,
    )

    table.add_column("Time",        width=10,  no_wrap=True, style="dim")
    table.add_column("Type",        width=18,  no_wrap=True, style="cyan")
    table.add_column("Record ID",   width=12,  no_wrap=True, style="dim")
    table.add_column("Summary",     min_width=40)
    table.add_column("Hash",        width=12,  no_wrap=True, style="dim")

    for entry in entries:
        ts       = entry.timestamp[11:19]
        rec_type = _truncate(entry.record_type, 18)
        rec_id   = _truncate(entry.record_id, 12)
        summary  = _payload_summary(entry.payload)
        style    = _decision_style(entry.payload)
        hash_str = entry.current_hash[:10] + "…"

        table.add_row(
            ts,
            rec_type,
            rec_id,
            Text(_truncate(summary, 70), style=style),
            hash_str,
        )

    console.print(table)
    console.print()


def _verify_chain(logger: AuditLogger, total: int) -> None:
    """Run full chain verification and display detailed results."""
    console.print(
        Text(
            f"  Verifying hash chain integrity — {total} entries...",
            style="dim",
        )
    )
    console.print()

    ok, errors = logger.verify_chain()

    if ok:
        console.print(Panel(
            f"[bold green]✔ CHAIN INTACT[/bold green]\n\n"
            f"[white]{total} entries verified.[/white]\n"
            f"[white]No tampering detected.[/white]\n"
            f"[white]SHA-256 hash chain is unbroken.[/white]",
            title="[bold]Audit Chain Verification[/bold]",
            border_style="green",
        ))
    else:
        error_lines = "\n".join(
            f"  [red]• {e}[/red]" for e in errors
        )
        console.print(Panel(
            f"[bold red]✘ CHAIN BROKEN — {len(errors)} error(s)[/bold red]\n\n"
            f"{error_lines}\n\n"
            "[bold red]The audit log has been modified.[/bold red]\n"
            "[red]This is a critical security event.[/red]\n"
            "[red]Preserve this log and escalate immediately.[/red]",
            title="[bold red]⚠ AUDIT CHAIN VIOLATION[/bold red]",
            border_style="red",
        ))

    console.print()


def _export_log(entries: list, export_path: Path) -> None:
    """Export all audit log entries to a JSONL file."""
    try:
        export_path.parent.mkdir(parents=True, exist_ok=True)
        with export_path.open("w", encoding="utf-8") as fh:
            for entry in entries:
                record = {
                    "log_id":       entry.log_id,
                    "timestamp":    entry.timestamp,
                    "record_type":  entry.record_type,
                    "record_id":    entry.record_id,
                    "prev_hash":    entry.prev_hash,
                    "current_hash": entry.current_hash,
                    "payload":      entry.payload,
                }
                fh.write(json.dumps(record) + "\n")

        console.print(Panel(
            f"[bold green]✔ Exported {len(entries)} entries[/bold green]\n"
            f"[white]File: {export_path}[/white]",
            title="[bold]Export Complete[/bold]",
            border_style="green",
        ))

    except OSError as exc:
        console.print(Panel(
            f"[bold red]Export failed: {exc}[/bold red]",
            title="[bold red]Export Error[/bold red]",
            border_style="red",
        ))

    console.print()