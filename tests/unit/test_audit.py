"""
Unit tests for the AuditLogger.
Run with: pytest tests/unit/test_audit.py -v
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from aisec.storage.audit import AuditLogger, GENESIS_HASH

# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture
def logger(tmp_path: Path) -> AuditLogger:
    """Return a fresh AuditLogger backed by a temporary file."""
    return AuditLogger(log_path=tmp_path / "test_audit.jsonl")


# ── Tests ─────────────────────────────────────────────────────────────────────


class TestAuditLogger:

    def test_empty_log_verifies_clean(self, logger: AuditLogger) -> None:
        ok, errors = logger.verify_chain()
        assert ok is True
        assert errors == []

    def test_single_entry_is_logged(self, logger: AuditLogger) -> None:
        entry = logger.log("event", "evt-001", {"action": "buy"})
        assert entry.log_id != ""
        assert entry.current_hash != ""
        assert entry.prev_hash == GENESIS_HASH

    def test_second_entry_links_to_first(self, logger: AuditLogger) -> None:
        first = logger.log("event", "evt-001", {"action": "buy"})
        second = logger.log("event", "evt-002", {"action": "sell"})
        assert second.prev_hash == first.current_hash

    def test_chain_verifies_after_multiple_entries(self, logger: AuditLogger) -> None:
        for i in range(10):
            logger.log("event", f"evt-{i:03}", {"index": i})
        ok, errors = logger.verify_chain()
        assert ok is True
        assert errors == []

    def test_count_returns_correct_number(self, logger: AuditLogger) -> None:
        for i in range(5):
            logger.log("event", f"evt-{i}", {"i": i})
        assert logger.count() == 5

    def test_get_last_returns_recent_entries(self, logger: AuditLogger) -> None:
        for i in range(10):
            logger.log("event", f"evt-{i}", {"i": i})
        last = logger.get_last(3)
        assert len(last) == 3
        assert last[-1].payload == {"i": 9}

    def test_tampered_entry_fails_verification(
        self, logger: AuditLogger, tmp_path: Path
    ) -> None:
        log_path = tmp_path / "test_audit.jsonl"
        logger2 = AuditLogger(log_path=log_path)

        logger2.log("event", "evt-001", {"action": "buy"})
        logger2.log("event", "evt-002", {"action": "sell"})

        # Read the raw file and tamper with the first entry
        lines = log_path.read_text(encoding="utf-8").splitlines()
        record = json.loads(lines[0])
        record["payload"]["action"] = "TAMPERED"
        lines[0] = json.dumps(record)
        log_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

        # Reload and verify — chain must be broken
        reloaded = AuditLogger(log_path=log_path)
        ok, errors = reloaded.verify_chain()
        assert ok is False
        assert len(errors) > 0

    def test_log_persists_across_instances(self, tmp_path: Path) -> None:
        log_path = tmp_path / "persist.jsonl"

        # First instance writes
        a = AuditLogger(log_path=log_path)
        a.log("event", "evt-001", {"action": "buy"})

        # Second instance reads and continues the chain
        b = AuditLogger(log_path=log_path)
        b.log("event", "evt-002", {"action": "sell"})

        ok, errors = b.verify_chain()
        assert ok is True
        assert b.count() == 2
