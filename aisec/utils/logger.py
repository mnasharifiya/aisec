"""
AISec structured JSON logger.

Produces machine-readable JSON log lines that SIEM tools
like Splunk, Elastic, and Grafana Loki can ingest directly.

Every log entry is a single JSON object on one line (JSONL format)
containing a timestamp, level, component, message, and optional
structured context fields.

This is separate from the audit log — the audit log is for
tamper-evident security records. This logger is for operational
observability — what the system is doing and why.

Usage:
    logger = get_logger("aisec.core.engine")
    logger.info("action_evaluated", risk_score=0.87, decision="BLOCK")
    logger.warning("chain_broken", entry_count=42, errors=3)
    logger.error("engine_failed", exc_type="ValueError", detail=str(e))
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timezone
from typing import Any

# ── JSON formatter ────────────────────────────────────────────────────────────


class JSONFormatter(logging.Formatter):
    """
    Formats log records as single-line JSON objects.

    Output example:
        {"ts":"2025-05-03T22:14:05.123456+00:00","level":"INFO",
         "component":"aisec.core.engine","msg":"action_evaluated",
         "risk_score":0.87,"decision":"BLOCK"}

    This format is directly ingestible by:
        - Splunk (JSON source type)
        - Elastic (filebeat JSON input)
        - Grafana Loki (JSON pipeline)
        - AWS CloudWatch (JSON structured logs)
    """

    def format(self, record: logging.LogRecord) -> str:
        """Format a LogRecord as a JSON line."""
        entry: dict[str, Any] = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "component": record.name,
            "msg": record.getMessage(),
        }

        # Include any extra fields passed via the extra= parameter
        # or as keyword arguments through our structured helper
        for key, value in record.__dict__.items():
            if key.startswith("_aisec_"):
                clean_key = key[7:]  # Strip the _aisec_ prefix
                entry[clean_key] = value

        # Include exception info if present
        if record.exc_info:
            entry["exc_type"] = (
                record.exc_info[0].__name__ if record.exc_info[0] else None
            )
            entry["exc_message"] = (
                str(record.exc_info[1]) if record.exc_info[1] else None
            )

        try:
            return json.dumps(entry, default=str, separators=(",", ":"))
        except (TypeError, ValueError):
            # Fallback — never let the logger itself crash
            return json.dumps(
                {
                    "ts": entry["ts"],
                    "level": "ERROR",
                    "component": "aisec.logger",
                    "msg": "Failed to serialize log entry",
                }
            )


# ── Structured logger wrapper ─────────────────────────────────────────────────


class StructuredLogger:
    """
    Wraps a standard Python logger to support structured key-value logging.

    Usage:
        log = get_logger("aisec.core.engine")
        log.info("decision_made", decision="BLOCK", risk=0.94, agent="trading_bot")
        log.warning("high_risk_detected", risk=0.85, action="execute_large_trade")
        log.error("engine_error", exc_type="ValueError", detail="invalid vector")
    """

    def __init__(self, name: str) -> None:
        self._logger = logging.getLogger(name)

    def _log(
        self,
        level: int,
        msg: str,
        **kwargs: Any,
    ) -> None:
        """Emit a structured log record with keyword fields."""
        extra = {f"_aisec_{k}": v for k, v in kwargs.items()}
        self._logger.log(level, msg, extra=extra)

    def debug(self, msg: str, **kwargs: Any) -> None:
        self._log(logging.DEBUG, msg, **kwargs)

    def info(self, msg: str, **kwargs: Any) -> None:
        self._log(logging.INFO, msg, **kwargs)

    def warning(self, msg: str, **kwargs: Any) -> None:
        self._log(logging.WARNING, msg, **kwargs)

    def error(self, msg: str, **kwargs: Any) -> None:
        self._log(logging.ERROR, msg, **kwargs)

    def critical(self, msg: str, **kwargs: Any) -> None:
        self._log(logging.CRITICAL, msg, **kwargs)


# ── Setup ─────────────────────────────────────────────────────────────────────

_configured = False


def configure_logging(
    level: str = "INFO",
    output: str = "stderr",
) -> None:
    """
    Configure AISec structured JSON logging.

    Must be called once at startup before any log messages are emitted.
    Subsequent calls are ignored.

    Args:
        level:  Minimum log level. One of DEBUG, INFO, WARNING, ERROR, CRITICAL.
        output: Where to write logs. "stderr" (default) or "stdout".
                Use "stdout" when running in Docker with log aggregation.
    """
    global _configured
    if _configured:
        return

    root = logging.getLogger("aisec")
    root.setLevel(getattr(logging, level.upper(), logging.INFO))

    handler = logging.StreamHandler(sys.stdout if output == "stdout" else sys.stderr)
    handler.setFormatter(JSONFormatter())
    root.addHandler(handler)

    # Prevent propagation to root logger to avoid duplicate output
    root.propagate = False
    _configured = True


def get_logger(name: str) -> StructuredLogger:
    """
    Return a StructuredLogger for the given component name.

    Args:
        name: Component name, e.g. "aisec.core.engine".
              Use __name__ for automatic module naming.

    Returns:
        StructuredLogger instance.
    """
    return StructuredLogger(name)
