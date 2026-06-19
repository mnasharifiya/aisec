"""
AISec audit log inspection CLI command.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import click

from aisec.storage.audit import DEFAULT_LOG_PATH

REDACT_KEYS = {
    "token",
    "secret",
    "password",
    "api_key",
    "apikey",
    "authorization",
    "credential",
    "credentials",
    "private_key",
}


RECORD_TYPES = [
    "all",
    "analysis",
    "temporal_alert",
    "temporal_anomaly",
    "correlation_alert",
    "multi_agent_correlation",
    "safe_state_entry",
    "safe_state_enter",
    "safe_state_exit",
    "analyst_decision",
    "soc_session_start",
    "soc_session_end",
]


def _payload(entry: Any) -> dict[str, Any]:
    payload = getattr(entry, "payload", {}) or {}
    return payload if isinstance(payload, dict) else {}


def _timestamp(entry: Any) -> str:
    return str(getattr(entry, "timestamp", ""))[:19] or "-"


def _record_id(entry: Any) -> str:
    return str(getattr(entry, "record_id", "unknown"))


def _record_type(entry: Any) -> str:
    return str(getattr(entry, "record_type", "unknown"))


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _decision_color(decision: str) -> str:
    decision = decision.upper()
    if decision in {"BLOCK", "ESCALATE"}:
        return "red"
    if decision in {"PENDING_REVIEW", "REVIEW"}:
        return "yellow"
    return "green"


def _redact(value: Any) -> Any:
    if isinstance(value, dict):
        clean: dict[str, Any] = {}
        for key, item in value.items():
            key_text = str(key).lower()
            if any(secret in key_text for secret in REDACT_KEYS):
                clean[key] = "***REDACTED***"
            else:
                clean[key] = _redact(item)
        return clean

    if isinstance(value, list):
        return [_redact(item) for item in value]

    return value


def _short(value: Any, limit: int = 80) -> str:
    text = str(value)
    text = text.replace("\r", " ").replace("\n", " ").replace("\t", " ")
    text = " ".join(text.split())
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 1)] + "…"


def _entry_to_json(entry: Any, *, redact: bool) -> dict[str, Any]:
    payload = _payload(entry)
    if redact:
        payload = _redact(payload)

    return {
        "timestamp": str(getattr(entry, "timestamp", "")),
        "record_type": _record_type(entry),
        "record_id": _record_id(entry),
        "payload": payload,
    }


def _matches_filters(
    entry: Any,
    *,
    record_type: str,
    agent_id: str | None,
    decision: str | None,
    contains: str | None,
) -> bool:
    payload = _payload(entry)

    if record_type != "all" and _record_type(entry) != record_type:
        return False

    if agent_id and str(payload.get("agent_id", "")) != agent_id:
        return False

    if decision and str(payload.get("decision", "")).upper() != decision.upper():
        return False

    if contains:
        haystack = json.dumps(
            _entry_to_json(entry, redact=True),
            sort_keys=True,
            ensure_ascii=False,
        ).lower()
        if contains.lower() not in haystack:
            return False

    return True


def _print_analysis(entry: Any) -> None:
    payload = _payload(entry)
    decision = str(payload.get("decision", "?"))
    risk = _safe_float(payload.get("risk_score", 0.0))

    click.echo(
        f"  {_timestamp(entry)}  "
        f"{_record_type(entry):<22}  "
        f"{_short(payload.get('action_type', '?'), 28):<28}  "
        + click.style(f"{decision:<15}", fg=_decision_color(decision))
        + f"  risk={risk:.3f}"
    )


def _print_alert(entry: Any, *, color: str) -> None:
    payload = _payload(entry)

    click.echo(
        click.style(
            f"  {_timestamp(entry)}  "
            f"{_record_type(entry):<22}  "
            f"{_short(payload.get('threat', '?'), 28):<28}  "
            f"{_short(payload.get('severity', '?'), 16)}",
            fg=color,
        )
    )


def _print_safe_state(entry: Any, *, color: str) -> None:
    payload = _payload(entry)

    click.echo(
        click.style(
            f"  {_timestamp(entry)}  "
            f"{_record_type(entry):<22}  "
            f"agent={_short(payload.get('agent_id', '?'), 20):<20}  "
            f"trigger={_short(payload.get('triggered_by', '?'), 20)}",
            fg=color,
        )
    )


def _print_analyst_decision(entry: Any) -> None:
    payload = _payload(entry)

    click.echo(
        click.style(
            f"  {_timestamp(entry)}  "
            f"{_record_type(entry):<22}  "
            f"analyst={_short(payload.get('analyst_id', '?'), 16):<16}  "
            f"decision={_short(payload.get('analyst_decision', '?'), 20)}",
            fg="cyan",
        )
    )


def _print_default(entry: Any, *, redact: bool) -> None:
    payload = _payload(entry)
    if redact:
        payload = _redact(payload)

    click.echo(
        f"  {_timestamp(entry)}  "
        f"{_record_type(entry):<22}  "
        f"{_short(payload, 80)}"
    )


def _print_entry(entry: Any, *, redact: bool) -> None:
    rtype = _record_type(entry)

    if rtype == "analysis":
        _print_analysis(entry)
    elif rtype in {"temporal_alert", "temporal_anomaly"}:
        _print_alert(entry, color="yellow")
    elif rtype in {"correlation_alert", "multi_agent_correlation"}:
        _print_alert(entry, color="magenta")
    elif rtype in {"safe_state_entry", "safe_state_enter"}:
        _print_safe_state(entry, color="red")
    elif rtype == "safe_state_exit":
        _print_safe_state(entry, color="green")
    elif rtype == "analyst_decision":
        _print_analyst_decision(entry)
    else:
        _print_default(entry, redact=redact)


@click.command("logs")
@click.option(
    "--log-path",
    type=click.Path(path_type=Path),
    default=DEFAULT_LOG_PATH,
    show_default=True,
    help="Path to AISec audit log.",
)
@click.option(
    "--verify",
    is_flag=True,
    help="Verify hash-chain integrity.",
)
@click.option(
    "--tail",
    type=click.IntRange(1, 10_000),
    default=20,
    show_default=True,
    help="Number of recent entries to display.",
)
@click.option(
    "--type",
    "record_type",
    default="all",
    type=click.Choice(RECORD_TYPES),
    show_default=True,
    help="Filter by record type.",
)
@click.option(
    "--agent-id",
    default=None,
    help="Filter by agent ID.",
)
@click.option(
    "--decision",
    default=None,
    help="Filter analysis entries by decision.",
)
@click.option(
    "--contains",
    default=None,
    help="Search within redacted entry JSON.",
)
@click.option(
    "--json",
    "json_mode",
    is_flag=True,
    help="Print entries as JSON.",
)
@click.option(
    "--no-redact",
    is_flag=True,
    help="Show full payload values. Use carefully.",
)
@click.option(
    "--export",
    "export_path",
    type=click.Path(path_type=Path),
    default=None,
    help="Export audit log in CEF format.",
)
@click.option(
    "--fail-on-broken-chain",
    is_flag=True,
    help="Exit with non-zero status if chain verification fails.",
)
def logs_command(
    log_path: Path,
    verify: bool,
    tail: int,
    record_type: str,
    agent_id: str | None,
    decision: str | None,
    contains: str | None,
    json_mode: bool,
    no_redact: bool,
    export_path: Path | None,
    fail_on_broken_chain: bool,
) -> None:
    """
    Inspect the tamper-evident AISec audit log.

    Examples:
        aisec logs
        aisec logs --verify
        aisec logs --tail 50 --type analysis
        aisec logs --agent-id agent_01 --decision BLOCK
        aisec logs --json
        aisec logs --export ./siem_export.cef
    """
    from aisec.core.engine import AnalysisEngine
    from aisec.integrations.siem import SIEMExporter

    if not log_path.exists():
        raise click.ClickException(
            f"Audit log not found: {log_path}. Run events first with aisec monitor."
        )

    engine = AnalysisEngine(log_path=log_path)

    try:
        entries = engine._logger.get_all()
    except Exception as exc:
        raise click.ClickException(f"Failed to read audit log: {exc}") from exc

    filtered = [
        entry
        for entry in entries
        if _matches_filters(
            entry,
            record_type=record_type,
            agent_id=agent_id,
            decision=decision,
            contains=contains,
        )
    ]

    display = filtered[-tail:]
    redact = not no_redact

    if json_mode:
        output = {
            "log_path": str(log_path),
            "total_entries": len(entries),
            "matched_entries": len(filtered),
            "displayed_entries": len(display),
            "entries": [_entry_to_json(entry, redact=redact) for entry in display],
        }

        if verify or fail_on_broken_chain:
            ok, errors = engine.verify_audit_chain()
            output["audit_chain"] = {
                "intact": ok,
                "error_count": len(errors),
                "errors": [str(error) for error in errors[:20]],
            }

        click.echo(json.dumps(output, indent=2, sort_keys=True, ensure_ascii=False))

    else:
        click.echo(f"\n  AISec Audit Log — {log_path}")
        click.echo(f"  Total entries:   {len(entries):>8,}")
        click.echo(f"  Matched entries: {len(filtered):>8,}")

        if verify or fail_on_broken_chain:
            ok, errors = engine.verify_audit_chain()
            if ok:
                click.echo("  Chain status:    " + click.style("INTACT ✔", fg="green"))
            else:
                click.echo(
                    "  Chain status:    "
                    + click.style(f"BROKEN ✘ ({len(errors)} errors)", fg="red")
                )
                for error in errors[:5]:
                    click.echo(f"    • {error}")

        click.echo(f"\n  Showing last {len(display)} of {len(filtered)} entries:\n")
        click.echo(f"  {'─' * 72}")

        for entry in display:
            _print_entry(entry, redact=redact)

        click.echo(f"  {'─' * 72}\n")

    if export_path:
        try:
            exporter = SIEMExporter(output_path=export_path)
            written = exporter.export_audit_log(engine._logger)
        except Exception as exc:
            raise click.ClickException(f"CEF export failed: {exc}") from exc

        if json_mode:
            click.echo(
                json.dumps(
                    {
                        "export_path": str(export_path),
                        "cef_lines_written": written,
                    },
                    indent=2,
                    sort_keys=True,
                )
            )
        else:
            click.echo(f"  Exported {written} CEF lines to {export_path}\n")

    if fail_on_broken_chain:
        ok, errors = engine.verify_audit_chain()
        if not ok:
            raise click.ClickException(
                f"Audit chain is broken: {len(errors)} error(s)."
            )
