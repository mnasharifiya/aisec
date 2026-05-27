"""
AISec tamper-evident audit logger.

Implements a SHA-256 hash chain where each entry references
the hash of the previous entry. Any modification to any entry
breaks the chain and is detected by verify_chain().

Storage: append-only JSONL file — one JSON object per line.
"""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any

from aisec.storage.models import AuditLogEntry

# ── Constants ─────────────────────────────────────────────────────────────────

GENESIS_HASH = "0" * 64  # Sentinel value for the very first entry
DEFAULT_LOG_PATH = Path(".aisec") / "audit.jsonl"


# ── AuditLogger ───────────────────────────────────────────────────────────────


class AuditLogger:
    """
    Append-only, tamper-evident audit log backed by a JSONL file.

    Usage:
        logger = AuditLogger()
        logger.log("event", event_id, {"action": "buy", "amount": 5000})
        ok, errors = logger.verify_chain()

    The log file is created automatically on first write.
    Never delete or edit the log file manually — doing so
    will break the chain and trigger verification failure.
    """

    def __init__(self, log_path: Path = DEFAULT_LOG_PATH) -> None:
        self._path  = log_path
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._lock  = threading.Lock()   # Serialises all writes — thread safety
        self._last_hash: str = self._load_last_hash()
        self._warn_if_insecure()

    def _warn_if_insecure(self) -> None:
        """
        Warn if the audit log directory may be accessible to other users.

        Full permission enforcement on Windows requires win32security.
        This warning is the minimum viable security notification.
        """
        import os
        import sys

        if sys.platform != "win32":
            try:
                mode = oct(os.stat(self._path.parent).st_mode)[-3:]
                # mode is a string like '700' or '755'; check 'others' permission bit
                if mode[2] != "0":
                    import warnings

                    warnings.warn(
                        f"Audit log directory {self._path.parent} may be "
                        f"readable by other users (permissions: {mode}). "
                        "Consider: chmod 700 .aisec/",
                        UserWarning,
                        stacklevel=2,
                    )
            except OSError:
                # If we cannot stat the directory, silently ignore the warning.
                pass

    # ── Public API ────────────────────────────────────────────────────────────

    def log(
        self,
        record_type: str,
        record_id: str,
        payload: dict[str, Any],
    ) -> AuditLogEntry:
        """
        Append a new entry to the audit log.
        Thread-safe: acquires a lock for the entire read-hash →
        build-entry → write sequence. This ensures:
        1. prev_hash always refers to the immediately preceding entry.
        2. No two threads write simultaneously (no interleaved bytes).
        3. The hash chain is always linear — never forked.
        Args:
            record_type: Category of the record.
            record_id:   Unique identifier of the record.
            payload:     Arbitrary data to store.
        Returns:
            The created AuditLogEntry with its computed hash.
        """
        with self._lock:
            entry = AuditLogEntry(
                record_type=record_type,
                record_id=record_id,
                payload=payload,
                prev_hash=self._last_hash,
            )
            self._append(entry)
            self._last_hash = entry.current_hash
            return entry

    def verify_chain(self) -> tuple[bool, list[str]]:
        """
        Read the entire log and verify the hash chain is intact.

        Returns:
            (True, [])               — chain is intact, no errors.
            (False, [error, ...])    — chain is broken, errors describe where.

        This is an O(n) operation over the entire log file.
        """
        entries = self._load_all()

        if not entries:
            return True, []

        errors: list[str] = []
        expected_prev = GENESIS_HASH

        for i, entry in enumerate(entries):
            if not entry.verify(expected_prev):
                errors.append(
                    f"Chain broken at entry {i}: "
                    f"log_id={entry.log_id}, "
                    f"record_id={entry.record_id}"
                )
            expected_prev = entry.current_hash

        return len(errors) == 0, errors

    def count(self) -> int:
        """Return the number of entries currently in the log."""
        return len(self._load_all())

    def get_all(self) -> list[AuditLogEntry]:
        """Return all entries in chronological order."""
        return self._load_all()

    def get_last(self, n: int = 10) -> list[AuditLogEntry]:
        """Return the most recent n entries."""
        return self._load_all()[-n:]

    # ── Private helpers ───────────────────────────────────────────────────────

    def _append(self, entry: AuditLogEntry) -> None:
        """Serialize the entry to JSON and append it to the log file."""
        record = {
            "log_id": entry.log_id,
            "timestamp": entry.timestamp,
            "record_type": entry.record_type,
            "record_id": entry.record_id,
            "prev_hash": entry.prev_hash,
            "current_hash": entry.current_hash,
            "payload": entry.payload,
        }
        with self._path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record) + "\n")

    def _load_all(self) -> list[AuditLogEntry]:
        """Read and deserialize all entries from the log file."""
        if not self._path.exists():
            return []

        entries: list[AuditLogEntry] = []

        with self._path.open("r", encoding="utf-8") as fh:
            for line_num, line in enumerate(fh, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    entry = AuditLogEntry(
                        log_id=data["log_id"],
                        timestamp=data["timestamp"],
                        record_type=data["record_type"],
                        record_id=data["record_id"],
                        prev_hash=data["prev_hash"],
                        payload=data["payload"],
                        current_hash=data["current_hash"],
                    )
                    entries.append(entry)
                except (json.JSONDecodeError, KeyError) as exc:
                    raise RuntimeError(
                        f"Corrupt audit log at line {line_num}: {exc}"
                    ) from exc

        return entries

    def _load_last_hash(self) -> str:
        """
        Return the hash of the most recent log entry.
        Returns GENESIS_HASH if the log is empty or does not exist.
        """
        entries = self._load_all()
        if not entries:
            return GENESIS_HASH
        return entries[-1].current_hash