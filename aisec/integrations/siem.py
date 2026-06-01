"""
AISec SIEM integration — Common Event Format (CEF) export.

Converts AISec analysis results into CEF log lines compatible
with Splunk, IBM QRadar, Elastic SIEM, and ArcSight.

CEF is the industry standard format for security events.
Every major SIEM platform can ingest CEF without custom parsing.

CEF format:
    CEF:Version|Device Vendor|Device Product|Device Version|
    Signature ID|Name|Severity|Extension

AISec CEF mapping:
    Signature ID  → rule IDs that fired (e.g. TRADING-001)
    Name          → action type (e.g. execute_large_trade)
    Severity      → 0-10 based on risk score and decision
    Extension     → agent_id, risk_score, scenario, decision, etc.

Usage:
    formatter = CEFFormatter()

    # Format a single result
    cef_line = formatter.format_result(engine_result)
    print(cef_line)

    # Write to syslog file for SIEM ingestion
    exporter = SIEMExporter(output_path=Path("/var/log/aisec/siem.log"))
    exporter.export(engine_result)

    # Export entire audit log
    exporter.export_audit_log(audit_logger)

Reference:
    ArcSight Common Event Format (CEF) Guide, Version 23
    https://www.microfocus.com/documentation/arcsight/
"""

from __future__ import annotations

import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from aisec.storage.models import Decision
from aisec.utils.logger import get_logger

log = get_logger("aisec.integrations.siem")

# ── Constants ─────────────────────────────────────────────────────────────────

CEF_VERSION = "CEF:0"
DEVICE_VENDOR = "AISec"
DEVICE_PRODUCT = "AISec Runtime Security"
DEVICE_VERSION = "1.2.0"

# CEF severity mapping (0=lowest, 10=highest)
_SEVERITY_MAP: dict[Decision, int] = {
    Decision.ALLOW: 1,
    Decision.PENDING_REVIEW: 5,
    Decision.ESCALATE: 8,
    Decision.BLOCK: 10,
}

# CEF signature IDs for AISec events
SIG_ANALYSIS_COMPLETE = "AISEC-001"
SIG_ACTION_BLOCKED = "AISEC-002"
SIG_ACTION_ESCALATED = "AISEC-003"
SIG_PENDING_REVIEW = "AISEC-004"
SIG_TEMPORAL_ALERT = "AISEC-005"
SIG_SAFE_STATE_ENTER = "AISEC-006"
SIG_SAFE_STATE_EXIT = "AISEC-007"
SIG_CHAIN_BROKEN = "AISEC-999"


# ── CEF sanitisation ──────────────────────────────────────────────────────────


def _sanitise_cef_value(value: str) -> str:
    """
    Sanitise a string for safe inclusion in CEF extension fields.

    CEF extension values must not contain unescaped:
        = (equals sign)
        \\ (backslash)
        \\n (newline)
        | (pipe — reserved in CEF header)

    Args:
        value: Raw string value.

    Returns:
        CEF-safe string with special characters escaped.
    """
    value = str(value)
    value = value.replace("\\", "\\\\")
    value = value.replace("=", "\\=")
    value = value.replace("\n", "\\n")
    value = value.replace("\r", "")
    value = value.replace("|", "\\|")
    return value[:500]  # CEF values have practical length limits


def _sanitise_cef_header(value: str) -> str:
    """
    Sanitise a string for safe inclusion in CEF header fields.

    Header fields cannot contain | (pipe) or \\ (backslash).
    """
    value = str(value)
    value = value.replace("\\", "\\\\")
    value = value.replace("|", "\\|")
    return value[:100]


# ── CEF formatter ─────────────────────────────────────────────────────────────


class CEFFormatter:
    """
    Formats AISec events as CEF log lines.

    Each CEF line is self-contained and can be shipped directly
    to a SIEM via syslog, file, or HTTP.

    Thread safety: CEFFormatter is stateless — safe for concurrent use.
    """

    def format_result(
        self,
        event_id: str,
        agent_id: str,
        action_type: str,
        decision: Decision,
        risk_score: float,
        rule_hits: list[str],
        scenario: str,
        explanation: str,
        timestamp: str,
    ) -> str:
        """
        Format an analysis result as a CEF log line.

        Args:
            event_id:    Unique event identifier.
            agent_id:    AI agent identifier.
            action_type: The action that was analysed.
            decision:    The enforcement decision.
            risk_score:  Risk score in [0.0, 1.0].
            rule_hits:   List of rule IDs that fired.
            scenario:    Threat scenario name.
            explanation: Human-readable explanation.
            timestamp:   UTC ISO-8601 timestamp.

        Returns:
            Single CEF-formatted log line (no trailing newline).
        """
        severity = _SEVERITY_MAP.get(decision, 5)
        sig_id = self._decision_to_sig_id(decision)
        event_name = _sanitise_cef_header(action_type)

        # Build extension fields
        ext_parts = [
            f"rt={self._format_timestamp(timestamp)}",
            f"src={_sanitise_cef_value(agent_id)}",
            f"act={_sanitise_cef_value(action_type)}",
            f"outcome={_sanitise_cef_value(decision.value)}",
            f"cs1={_sanitise_cef_value(scenario)}",
            f"cs1Label=scenario",
            f"cs2={_sanitise_cef_value(','.join(rule_hits) or 'none')}",
            f"cs2Label=ruleHits",
            f"cs3={_sanitise_cef_value(event_id)}",
            f"cs3Label=eventId",
            f"cn1={round(risk_score * 100)}",
            f"cn1Label=riskScore",
            f"msg={_sanitise_cef_value(explanation[:200])}",
        ]

        extension = " ".join(ext_parts)

        return (
            f"{CEF_VERSION}"
            f"|{_sanitise_cef_header(DEVICE_VENDOR)}"
            f"|{_sanitise_cef_header(DEVICE_PRODUCT)}"
            f"|{_sanitise_cef_header(DEVICE_VERSION)}"
            f"|{_sanitise_cef_header(sig_id)}"
            f"|{event_name}"
            f"|{severity}"
            f"|{extension}"
        )

    def format_temporal_alert(
        self,
        agent_id: str,
        threat: str,
        severity: str,
        description: str,
        timestamp: str,
    ) -> str:
        """Format a temporal anomaly alert as CEF."""
        cef_severity = 8 if severity == "CRITICAL" else 5

        ext_parts = [
            f"rt={self._format_timestamp(timestamp)}",
            f"src={_sanitise_cef_value(agent_id)}",
            f"cs1={_sanitise_cef_value(threat)}",
            f"cs1Label=threatType",
            f"cs2={_sanitise_cef_value(severity)}",
            f"cs2Label=severity",
            f"msg={_sanitise_cef_value(description[:200])}",
        ]

        return (
            f"{CEF_VERSION}"
            f"|{_sanitise_cef_header(DEVICE_VENDOR)}"
            f"|{_sanitise_cef_header(DEVICE_PRODUCT)}"
            f"|{_sanitise_cef_header(DEVICE_VERSION)}"
            f"|{_sanitise_cef_header(SIG_TEMPORAL_ALERT)}"
            f"|TemporalThreatDetected"
            f"|{cef_severity}"
            f"|{' '.join(ext_parts)}"
        )

    def format_audit_entry(self, entry: Any) -> str | None:
        """
        Format an audit log entry as a CEF line.

        Returns None for entry types that do not map to CEF events.
        """
        if entry.record_type == "analysis":
            p = entry.payload
            try:
                decision = Decision(p.get("decision", "ALLOW"))
            except ValueError:
                decision = Decision.ALLOW

            return self.format_result(
                event_id=entry.record_id,
                agent_id=p.get("agent_id", "unknown"),
                action_type=p.get("action_type", "unknown"),
                decision=decision,
                risk_score=float(p.get("risk_score", 0.0)),
                rule_hits=p.get("rule_hits", []),
                scenario=p.get("scenario", "unknown"),
                explanation=p.get("explanation", ""),
                timestamp=entry.timestamp,
            )

        elif entry.record_type == "temporal_alert":
            p = entry.payload
            return self.format_temporal_alert(
                agent_id=p.get("agent_id", "unknown"),
                threat=p.get("threat", "UNKNOWN"),
                severity=p.get("severity", "HIGH"),
                description=p.get("description", ""),
                timestamp=entry.timestamp,
            )

        elif entry.record_type == "safe_state_entry":
            p = entry.payload
            ext = (
                f"rt={self._format_timestamp(entry.timestamp)} "
                f"src={_sanitise_cef_value(p.get('agent_id', ''))} "
                f"cs1={_sanitise_cef_value(p.get('triggered_by', ''))} "
                f"cs1Label=triggeredBy "
                f"msg={_sanitise_cef_value(p.get('reason', '')[:200])}"
            )
            return (
                f"{CEF_VERSION}"
                f"|{DEVICE_VENDOR}|{DEVICE_PRODUCT}|{DEVICE_VERSION}"
                f"|{SIG_SAFE_STATE_ENTER}|AgentSafeStateEntered|9|{ext}"
            )

        return None  # Unknown type — skip

    # ── Private helpers ───────────────────────────────────────────────────────

    @staticmethod
    def _decision_to_sig_id(decision: Decision) -> str:
        return {
            Decision.ALLOW: SIG_ANALYSIS_COMPLETE,
            Decision.BLOCK: SIG_ACTION_BLOCKED,
            Decision.ESCALATE: SIG_ACTION_ESCALATED,
            Decision.PENDING_REVIEW: SIG_PENDING_REVIEW,
        }.get(decision, SIG_ANALYSIS_COMPLETE)

    @staticmethod
    def _format_timestamp(ts: str) -> str:
        """Convert ISO-8601 timestamp to CEF milliseconds since epoch."""
        try:
            dt = datetime.fromisoformat(ts)
            return str(int(dt.timestamp() * 1000))
        except (ValueError, OSError):
            return str(int(datetime.now(timezone.utc).timestamp() * 1000))


# ── SIEM exporter ─────────────────────────────────────────────────────────────


class SIEMExporter:
    """
    Exports AISec events to a CEF log file for SIEM ingestion.

    The output file can be:
    - Tailed by a log shipper (Filebeat, Splunk UF, NXLog)
    - Shipped to syslog with: logger -f /var/log/aisec/siem.log
    - Ingested directly by QRadar log source configuration

    Thread safety:
        File writes are protected by opening in append mode.
        On Linux/Mac this is atomic for small writes.
        For high throughput, use a log shipper instead.
    """

    def __init__(
        self,
        output_path: Path,
        formatter: CEFFormatter | None = None,
    ) -> None:
        self._path = output_path
        self._formatter = formatter or CEFFormatter()
        self._path.parent.mkdir(parents=True, exist_ok=True)

        log.info(
            "siem_exporter_initialized",
            output_path=str(output_path),
        )

    def write_line(self, cef_line: str) -> None:
        """
        Append a single CEF line to the output file.

        Args:
            cef_line: A formatted CEF log line.
        """
        try:
            with self._path.open("a", encoding="utf-8") as fh:
                fh.write(cef_line + "\n")
        except OSError as exc:
            log.error(
                "siem_write_error",
                exc_type=type(exc).__name__,
                detail=str(exc)[:200],
            )

    def export_audit_log(self, audit_logger: Any) -> int:
        """
        Export all entries from an audit log to CEF format.

        Args:
            audit_logger: An AuditLogger instance.

        Returns:
            Number of CEF lines written.
        """
        entries = audit_logger.get_all()
        written = 0

        for entry in entries:
            cef_line = self._formatter.format_audit_entry(entry)
            if cef_line:
                self.write_line(cef_line)
                written += 1

        log.info(
            "siem_export_complete",
            total_entries=len(entries),
            cef_lines_written=written,
            output_path=str(self._path),
        )

        return written
