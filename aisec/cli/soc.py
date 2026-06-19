"""
AISec SOC Console — Interactive Security Operations Centre.

A terminal-based analyst console for reviewing and resolving AI agent actions
that require human approval.

Enterprise-oriented design:
    - Interactive SOC queue review.
    - Analyst approve / block / escalate decisions.
    - Admin-only safe-state release.
    - RBAC enforcement for every sensitive operation.
    - Audit logging for sessions and decisions.
    - Queue filtering and event inspection.
    - Safe input sanitization for audit/log output.
    - JSON output support for automation and demos.
    - Local audit-log backend for demos and research evaluation.

Usage:
    aisec soc --role analyst
    aisec soc --role admin
    aisec soc --scenario trading_ai --role analyst
    aisec soc --json

Security:
    - All analyst decisions are written to the tamper-evident audit trail.
    - Admin actions require explicit confirmation.
    - Role is fixed at session startup.
    - Local demo role selection is not authentication.
    - Production authentication must happen before this CLI is trusted.
"""

from __future__ import annotations

import json
import re
import sys
import time
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Iterable, Sequence

import click

from aisec.storage.audit import DEFAULT_LOG_PATH
from aisec.utils.logger import get_logger

log = get_logger("aisec.cli.soc")


# ── Constants ─────────────────────────────────────────────────────────────────

MAX_TEXT_LEN = 500
MAX_REASON_LEN = 300
MAX_EVENT_DISPLAY = 50
MAX_AGENT_ID_LEN = 128
MAX_PRINCIPAL_ID_LEN = 128

_SAFE_TEXT_RE = re.compile(r"[^A-Za-z0-9_.:@/\- ,;()[\]{}+=#%!?]")
_SAFE_ID_RE = re.compile(r"^[A-Za-z0-9_.:@-]+$")


# ── Enums / Models ────────────────────────────────────────────────────────────


class SOCDecision(str, Enum):
    """Allowed analyst decisions."""

    APPROVE = "approve"
    BLOCK = "block"
    ESCALATE = "escalate"
    FALSE_POSITIVE = "false_positive"
    DISMISS = "dismiss"


class QueueStatus(str, Enum):
    """Queue status filters."""

    PENDING = "pending"
    ESCALATED = "escalated"
    ALL = "all"


@dataclass(frozen=True)
class SOCQueueItem:
    """Normalized SOC queue item."""

    record_id: str
    decision: str
    action_type: str
    agent_id: str
    risk_score: float
    explanation: str
    payload: dict[str, Any]
    created_at: Any = None


@dataclass(frozen=True)
class SOCSession:
    """SOC console session context."""

    principal_id: str
    role: str
    scenario: str
    started_at: float


# ── Sanitization helpers ──────────────────────────────────────────────────────


def _sanitize_text(value: Any, *, max_len: int = MAX_TEXT_LEN) -> str:
    """Return audit-safe human-readable text."""
    if value is None:
        return ""

    text = str(value)
    text = text.replace("\r", " ").replace("\n", " ").replace("\t", " ")
    text = _SAFE_TEXT_RE.sub("", text)
    text = " ".join(text.split())
    return text[:max_len]


def _sanitize_identifier(value: Any, *, fallback: str = "unknown") -> str:
    """Return safe ID-like text for principal/event/agent IDs."""
    if value is None:
        return fallback

    text = str(value).strip()
    text = text.replace("\r", "").replace("\n", "").replace("\t", "")

    if not text:
        return fallback

    if len(text) > MAX_PRINCIPAL_ID_LEN:
        text = text[:MAX_PRINCIPAL_ID_LEN]

    if not _SAFE_ID_RE.fullmatch(text):
        return fallback

    return text


def _safe_float(value: Any, *, default: float = 0.0) -> float:
    """Convert value to float safely."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _format_percentage(count: int, total: int) -> str:
    if total <= 0:
        return "0.0%"
    return f"{count / total * 100:.1f}%"


def _short(text: Any, length: int = 80) -> str:
    value = _sanitize_text(text, max_len=MAX_TEXT_LEN)
    if len(value) <= length:
        return value
    return value[: max(0, length - 1)] + "…"


def _print_json(data: Any) -> None:
    click.echo(json.dumps(data, indent=2, sort_keys=True, ensure_ascii=False))


# ── Role / principal helpers ──────────────────────────────────────────────────


def _get_role(role_str: str):
    """Parse user role into AISec Role."""
    from aisec.security.rbac import Role

    role_map = {
        "analyst": Role.ANALYST,
        "admin": Role.ADMIN,
    }

    role = role_map.get(role_str.lower())
    if role is None:
        raise click.BadParameter(f"Invalid role '{role_str}'. Choose: analyst, admin")

    return role


def _build_principal(principal_id: str, role: str):
    """Build an immutable RBAC principal."""
    from aisec.security.rbac import Principal

    role_obj = _get_role(role)
    safe_principal_id = _sanitize_identifier(
        principal_id,
        fallback="soc_user",
    )

    if len(safe_principal_id) < 3:
        safe_principal_id = "soc_user"

    return Principal(
        principal_id=safe_principal_id,
        role=role_obj,
        display_name=safe_principal_id,
    )


def _display_header(role_str: str, scenario: str) -> None:
    """Print SOC console header."""
    click.echo("\n" + "=" * 64)
    click.echo("  AISec SOC Console")
    click.echo(f"  Role: {role_str.upper()}  |  Scenario: {scenario}")
    click.echo("=" * 64)


# ── Audit / queue helpers ─────────────────────────────────────────────────────


def _get_entries(engine: Any) -> list[Any]:
    """Read all audit entries safely."""
    try:
        entries = engine._logger.get_all()
        return list(entries)
    except Exception as exc:
        click.echo(
            click.style(f"  Failed to read audit log: {exc}", fg="red"),
            err=True,
        )
        return []


def _analysis_entries(engine: Any) -> list[Any]:
    """Return analysis audit entries only."""
    return [
        entry
        for entry in _get_entries(engine)
        if getattr(entry, "record_type", None) == "analysis"
    ]


def _decision_entries(engine: Any) -> list[Any]:
    """Return analyst decision entries only."""
    return [
        entry
        for entry in _get_entries(engine)
        if getattr(entry, "record_type", None) == "analyst_decision"
    ]


def _resolved_event_ids(engine: Any) -> set[str]:
    """Return event IDs that already have analyst decisions."""
    resolved: set[str] = set()

    for entry in _decision_entries(engine):
        record_id = getattr(entry, "record_id", "")
        payload = getattr(entry, "payload", {}) or {}

        if record_id:
            resolved.add(str(record_id))

        event_id = payload.get("event_id")
        if event_id:
            resolved.add(str(event_id))

    return resolved


def _normalize_queue_item(entry: Any) -> SOCQueueItem:
    """Convert an audit entry into a normalized queue item."""
    payload = getattr(entry, "payload", {}) or {}

    record_id = _sanitize_identifier(
        getattr(entry, "record_id", "unknown"),
        fallback="unknown",
    )
    decision = _sanitize_text(payload.get("decision", "UNKNOWN"))
    action_type = _sanitize_text(payload.get("action_type", "unknown"))
    agent_id = _sanitize_identifier(
        payload.get("agent_id", "?"),
        fallback="unknown_agent",
    )
    risk_score = _safe_float(payload.get("risk_score", 0.0))
    explanation = _sanitize_text(payload.get("explanation", ""))

    return SOCQueueItem(
        record_id=record_id,
        decision=decision,
        action_type=action_type,
        agent_id=agent_id,
        risk_score=risk_score,
        explanation=explanation,
        payload=dict(payload),
        created_at=getattr(entry, "created_at", None),
    )


def _queue_items(
    engine: Any,
    *,
    status: QueueStatus = QueueStatus.PENDING,
    limit: int = MAX_EVENT_DISPLAY,
    agent_id: str | None = None,
) -> list[SOCQueueItem]:
    """Return unresolved queue items matching filter."""
    resolved_ids = _resolved_event_ids(engine)
    items: list[SOCQueueItem] = []

    for entry in _analysis_entries(engine):
        item = _normalize_queue_item(entry)

        if item.record_id in resolved_ids:
            continue

        decision_upper = item.decision.upper()

        if status == QueueStatus.PENDING:
            if decision_upper != "PENDING_REVIEW":
                continue
        elif status == QueueStatus.ESCALATED:
            if decision_upper != "ESCALATE":
                continue
        elif status == QueueStatus.ALL:
            if decision_upper not in {"PENDING_REVIEW", "ESCALATE"}:
                continue

        if agent_id and item.agent_id != agent_id:
            continue

        items.append(item)

    return items[: max(1, min(limit, MAX_EVENT_DISPLAY))]


def _find_queue_item(engine: Any, event_id: str) -> SOCQueueItem | None:
    """Find a queue event by ID."""
    safe_event_id = _sanitize_identifier(event_id, fallback="")
    if not safe_event_id:
        return None

    for item in _queue_items(engine, status=QueueStatus.ALL, limit=MAX_EVENT_DISPLAY):
        if item.record_id == safe_event_id:
            return item

    for entry in _analysis_entries(engine):
        item = _normalize_queue_item(entry)
        if item.record_id == safe_event_id:
            return item

    return None


def _is_event_resolved(engine: Any, event_id: str) -> bool:
    """Return True if an event already has analyst resolution."""
    safe_event_id = _sanitize_identifier(event_id, fallback="")
    return safe_event_id in _resolved_event_ids(engine)


# ── Display helpers ───────────────────────────────────────────────────────────


def _display_queue(
    engine: Any,
    enforcer: Any,
    principal: Any,
    *,
    status: QueueStatus = QueueStatus.ALL,
    limit: int = 10,
    agent_id: str | None = None,
    json_mode: bool = False,
) -> list[Any] | None:
    """Display unresolved SOC queue events."""
    from aisec.security.rbac import Permission

    if not enforcer.check(principal, Permission.VIEW_QUEUE):
        click.echo(click.style("  Access denied: cannot view queue.", fg="red"))
        return None

    items = _queue_items(
        engine,
        status=status,
        limit=limit,
        agent_id=agent_id,
    )

    if json_mode:
        _print_json([item.__dict__ for item in items])
        return items if items else None

    click.echo(f"\n  Pending review: {len(items)} event(s)\n")

    if not items:
        click.echo("  Queue is empty — no events require review.")
        return None

    for index, item in enumerate(items, 1):
        decision_upper = item.decision.upper()
        decision_color = "yellow"
        if decision_upper == "ESCALATE":
            decision_color = "red"

        click.echo(
            f"  [{index}] {item.action_type:<30} "
            f"agent={item.agent_id:<20} "
            f"risk={item.risk_score:.3f}  "
            + click.style(item.decision, fg=decision_color)
        )
        click.echo(f"      {_short(item.explanation, 90)}")
        click.echo(f"      ID: {item.record_id}")
        click.echo()

    return items


def _display_event_detail(
    engine: Any,
    enforcer: Any,
    principal: Any,
    event_id: str,
    *,
    json_mode: bool = False,
) -> None:
    """Display one event in detail."""
    from aisec.security.rbac import Permission

    if not enforcer.check(principal, Permission.VIEW_QUEUE):
        click.echo(click.style("  Access denied: cannot inspect events.", fg="red"))
        return

    item = _find_queue_item(engine, event_id)
    if item is None:
        click.echo(click.style("  Event not found.", fg="red"))
        return

    if json_mode:
        _print_json(item.__dict__)
        return

    click.echo("\n  Event Detail")
    click.echo("  " + "-" * 56)
    click.echo(f"  ID:          {item.record_id}")
    click.echo(f"  Agent:       {item.agent_id}")
    click.echo(f"  Action:      {item.action_type}")
    click.echo(f"  Decision:    {item.decision}")
    click.echo(f"  Risk score:  {item.risk_score:.3f}")
    click.echo(f"  Explanation: {item.explanation}")
    click.echo("\n  Payload:")
    _print_json(item.payload)


def _display_safe_state(
    engine: Any,
    enforcer: Any,
    principal: Any,
    *,
    json_mode: bool = False,
) -> None:
    """Display active safe-state restrictions."""
    from aisec.security.rbac import Permission

    if not enforcer.check(principal, Permission.VIEW_SAFE_STATE):
        click.echo(click.style("  Access denied: cannot view safe state.", fg="red"))
        return

    active = engine.safe_state.list_active()

    if json_mode:
        _print_json([entry.__dict__ for entry in active])
        return

    click.echo(f"\n  Agents in safe state: {len(active)}")

    if not active:
        click.echo("  No agents currently restricted.")
        return

    for entry in active:
        agent_id = _sanitize_identifier(
            getattr(entry, "agent_id", "unknown"),
            fallback="unknown_agent",
        )
        triggered_by = _sanitize_text(getattr(entry, "triggered_by", "unknown"))
        reason = _sanitize_text(getattr(entry, "reason", ""))
        entered_at = getattr(entry, "entered_at", "")

        click.echo(
            f"  • {agent_id:<30} "
            + click.style("RESTRICTED", fg="red")
            + f"  triggered_by={triggered_by}"
        )
        click.echo(f"    Reason: {reason[:100]}")
        click.echo(f"    Since:  {entered_at}")
        click.echo()


def _display_metrics(
    engine: Any,
    enforcer: Any,
    principal: Any,
    *,
    json_mode: bool = False,
) -> None:
    """Display SOC metrics."""
    from aisec.security.rbac import Permission

    if not enforcer.check(principal, Permission.VIEW_METRICS):
        click.echo(click.style("  Access denied: cannot view metrics.", fg="red"))
        return

    analysis_entries = _analysis_entries(engine)
    total = len(analysis_entries)

    if total == 0:
        if json_mode:
            _print_json(
                {
                    "events_analysed": 0,
                    "blocked": 0,
                    "pending_review": 0,
                    "allowed": 0,
                    "audit_chain_intact": True,
                    "safe_state_count": engine.safe_state.active_count(),
                }
            )
            return

        click.echo("\n  No events analysed yet.")
        return

    decisions = [
        str((getattr(entry, "payload", {}) or {}).get("decision", "")).upper()
        for entry in analysis_entries
    ]

    blocked = sum(1 for decision in decisions if decision in {"BLOCK", "ESCALATE"})
    reviewed = sum(1 for decision in decisions if decision == "PENDING_REVIEW")
    allowed = total - blocked - reviewed

    ok, errors = engine.verify_audit_chain()
    safe_state_count = engine.safe_state.active_count()

    if json_mode:
        _print_json(
            {
                "events_analysed": total,
                "blocked": blocked,
                "pending_review": reviewed,
                "allowed": allowed,
                "audit_chain_intact": ok,
                "audit_errors": errors,
                "safe_state_count": safe_state_count,
            }
        )
        return

    chain_label = (
        click.style("INTACT", fg="green") if ok else click.style("BROKEN", fg="red")
    )

    click.echo(f"\n  Events analysed:  {total}")
    click.echo(f"  Blocked:          {blocked} ({_format_percentage(blocked, total)})")
    click.echo(
        f"  Pending review:   {reviewed} ({_format_percentage(reviewed, total)})"
    )
    click.echo(f"  Allowed:          {allowed} ({_format_percentage(allowed, total)})")
    click.echo(f"  Audit chain:      {chain_label}")
    click.echo(f"  Safe state count: {safe_state_count}")


# ── Mutating SOC actions ──────────────────────────────────────────────────────


def _resolve_event(
    event_id: str,
    decision: str,
    engine: Any,
    enforcer: Any,
    principal: Any,
    *,
    reason: str | None = None,
) -> None:
    """Resolve an event from the SOC queue."""
    from aisec.security.rbac import Permission

    if not enforcer.check(principal, Permission.RESOLVE_QUEUE):
        click.echo(click.style("  Access denied: cannot resolve events.", fg="red"))
        return

    safe_event_id = _sanitize_identifier(event_id, fallback="")
    if not safe_event_id:
        click.echo(click.style("  Invalid event ID.", fg="red"))
        return

    normalized_decision = _sanitize_text(decision, max_len=40).lower()
    allowed_decisions = {item.value for item in SOCDecision}
    if normalized_decision not in allowed_decisions:
        click.echo(
            click.style(
                f"  Invalid decision. Choose: {', '.join(sorted(allowed_decisions))}",
                fg="red",
            )
        )
        return

    if _is_event_resolved(engine, safe_event_id):
        click.echo(click.style("  Event is already resolved.", fg="yellow"))
        return

    clean_reason = _sanitize_text(
        reason or f"SOC console decision by {principal.principal_id}",
        max_len=MAX_REASON_LEN,
    )

    engine._logger.log(
        record_type="analyst_decision",
        record_id=safe_event_id,
        payload={
            "analyst_id": principal.principal_id,
            "analyst_role": principal.role.value,
            "analyst_decision": normalized_decision,
            "reason": clean_reason,
            "event_id": safe_event_id,
            "source": "soc_console",
        },
    )

    color = "green" if normalized_decision in {"approve", "false_positive"} else "red"
    if normalized_decision == "escalate":
        color = "yellow"

    click.echo(
        click.style(
            f"  Decision recorded: {normalized_decision.upper()}",
            fg=color,
        )
        + f"  (event={safe_event_id[:24]})"
    )


def _release_safe_state(
    agent_id: str,
    engine: Any,
    enforcer: Any,
    principal: Any,
    *,
    reason: str = "Released via SOC console",
    force_confirm: bool = False,
) -> None:
    """Release an agent from safe state. Admin-only."""
    from aisec.security.rbac import AccessDeniedError, Permission

    safe_agent_id = _sanitize_identifier(
        agent_id,
        fallback="",
    )

    if not safe_agent_id:
        click.echo(click.style("  Invalid agent ID.", fg="red"))
        return

    try:
        enforcer.require(
            principal,
            Permission.MANAGE_SAFE_STATE,
            f"release agent {safe_agent_id} from safe state",
        )
    except AccessDeniedError:
        click.echo(
            click.style(
                "  Access denied: only admins can release agents from safe state.",
                fg="red",
            )
        )
        return

    clean_reason = _sanitize_text(reason, max_len=MAX_REASON_LEN)

    if not force_confirm:
        confirmed = click.confirm(
            f"  Release '{safe_agent_id}' from safe state? "
            "This action is audit logged."
        )
        if not confirmed:
            click.echo("  Cancelled.")
            return

    released = engine.safe_state.exit_safe_state(
        agent_id=safe_agent_id,
        admin_id=principal.principal_id,
        reason=clean_reason,
    )

    if released:
        click.echo(
            click.style(
                f"  Agent '{safe_agent_id}' released from safe state.",
                fg="green",
            )
        )
    else:
        click.echo(f"  Agent '{safe_agent_id}' was not in safe state.")


# ── Session logging ───────────────────────────────────────────────────────────


def _log_session_start(engine: Any, session: SOCSession) -> None:
    """Log SOC session start."""
    engine._logger.log(
        record_type="soc_session_start",
        record_id=session.principal_id,
        payload={
            "principal_id": session.principal_id,
            "role": session.role,
            "scenario": session.scenario,
            "started_at": session.started_at,
            "source": "soc_console",
        },
    )


def _log_session_end(engine: Any, session: SOCSession) -> None:
    """Log SOC session end."""
    engine._logger.log(
        record_type="soc_session_end",
        record_id=session.principal_id,
        payload={
            "principal_id": session.principal_id,
            "role": session.role,
            "scenario": session.scenario,
            "ended_at": time.time(),
            "source": "soc_console",
        },
    )


# ── Interactive command ───────────────────────────────────────────────────────


@click.command("soc")
@click.option(
    "--role",
    default="analyst",
    type=click.Choice(["analyst", "admin"], case_sensitive=False),
    show_default=True,
    help="Local demo role: analyst or admin.",
)
@click.option(
    "--principal-id",
    default="soc_user",
    help="Your identity. Logged in the audit trail.",
)
@click.option(
    "--log-path",
    type=click.Path(path_type=Path),
    default=DEFAULT_LOG_PATH,
    show_default=True,
    help="Path to the audit log.",
)
@click.option(
    "--scenario",
    default="both",
    type=click.Choice(["trading_ai", "urban_ai", "drone", "both"]),
    show_default=True,
)
@click.option(
    "--json",
    "json_mode",
    is_flag=True,
    help="Print machine-readable JSON for supported displays.",
)
def soc_command(
    role: str,
    principal_id: str,
    log_path: Path,
    scenario: str,
    json_mode: bool,
) -> None:
    """
    Interactive SOC analyst console.

    Review and resolve AI agent actions that require human approval.
    All decisions are recorded in the tamper-evident audit log.

    Examples:
        aisec soc
        aisec soc --role admin --principal-id admin_01
        aisec soc --log-path /var/log/aisec/audit.jsonl
    """
    from aisec.core.engine import AnalysisEngine
    from aisec.security.rbac import RBACEnforcer

    principal = _build_principal(principal_id, role)
    enforcer = RBACEnforcer()
    engine = AnalysisEngine(log_path=log_path)

    session = SOCSession(
        principal_id=principal.principal_id,
        role=role.lower(),
        scenario=scenario,
        started_at=time.time(),
    )

    _display_header(role, scenario)
    _log_session_start(engine, session)

    click.echo(f"\n  Logged in as: {principal.principal_id} ({role})")
    click.echo(
        click.style(
            "  Note: local role selection is for demo/research use. "
            "Production must authenticate identity before RBAC.",
            fg="yellow",
        )
    )

    try:
        while True:
            click.echo("\n  " + "─" * 60)
            click.echo("  [q] Queue       [i] Inspect event     [m] Metrics")
            click.echo("  [s] Safe State  [x] Exit")
            if role.lower() == "admin":
                click.echo("  [r] Release agent from safe state")
            click.echo("  " + "─" * 60)

            choice = click.prompt("  Command", default="q").strip().lower()

            if choice == "x":
                click.echo("\n  Session ended. Goodbye.\n")
                break

            if choice == "q":
                status_input = click.prompt(
                    "  Filter",
                    default="all",
                    type=click.Choice(
                        ["pending", "escalated", "all"],
                        case_sensitive=False,
                    ),
                ).lower()

                status = QueueStatus(status_input)

                unresolved = _display_queue(
                    engine,
                    enforcer,
                    principal,
                    status=status,
                    limit=10,
                    json_mode=json_mode,
                )

                if unresolved:
                    resolve = click.prompt(
                        "\n  Enter event number to resolve, or Enter to skip",
                        default="",
                    ).strip()

                    if resolve.isdigit():
                        index = int(resolve) - 1
                        if 0 <= index < len(unresolved):
                            selected = unresolved[index]
                            decision = click.prompt(
                                "  Decision",
                                type=click.Choice(
                                    [
                                        "approve",
                                        "block",
                                        "escalate",
                                        "false_positive",
                                        "dismiss",
                                    ],
                                    case_sensitive=False,
                                ),
                            ).lower()
                            reason = click.prompt(
                                "  Reason",
                                default="Reviewed in SOC console",
                            )

                            _resolve_event(
                                selected.record_id,
                                decision,
                                engine,
                                enforcer,
                                principal,
                                reason=reason,
                            )
                        else:
                            click.echo("  Invalid event number.")

                continue

            if choice == "i":
                event_id = click.prompt("  Event ID").strip()
                _display_event_detail(
                    engine,
                    enforcer,
                    principal,
                    event_id,
                    json_mode=json_mode,
                )
                continue

            if choice == "m":
                _display_metrics(
                    engine,
                    enforcer,
                    principal,
                    json_mode=json_mode,
                )
                continue

            if choice == "s":
                _display_safe_state(
                    engine,
                    enforcer,
                    principal,
                    json_mode=json_mode,
                )
                continue

            if choice == "r" and role.lower() == "admin":
                agent_id = click.prompt("  Agent ID to release").strip()
                reason = click.prompt(
                    "  Reason",
                    default="Admin reviewed and released via SOC console",
                )
                _release_safe_state(
                    agent_id,
                    engine,
                    enforcer,
                    principal,
                    reason=reason,
                )
                continue

            click.echo("  Unknown command.")

    finally:
        _log_session_end(engine, session)


# ── Optional standalone entry point ───────────────────────────────────────────


def main(argv: Sequence[str] | None = None) -> int:
    """Standalone entry point for python -m aisec.cli.soc."""
    try:
        args = list(argv) if argv is not None else None
        soc_command.main(args=args, standalone_mode=True)
        return 0
    except SystemExit as exc:
        code = exc.code
        return int(code) if isinstance(code, int) else 0
    except Exception as exc:
        click.echo(f"ERROR: {exc}", err=True)
        return 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
