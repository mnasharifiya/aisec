"""
Unit tests for crypto and time utilities.
Run with: pytest tests/unit/test_crypto.py -v
"""

from __future__ import annotations

import time
from pathlib import Path

import pytest

from aisec.utils.crypto import (
    canonical_json,
    generate_secret_key,
    sha256_hex,
    sha256_file,
    sign,
    sign_policy_file,
    verify_policy_file,
    verify_signature,
)
from aisec.utils.time import (
    from_timestamp,
    is_within_seconds,
    now_utc,
    parse_utc,
    seconds_between,
)


# ── SHA-256 hashing ───────────────────────────────────────────────────────────

class TestSha256Hex:

    def test_known_hash(self) -> None:
        result = sha256_hex("hello")
        assert result == "2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824"

    def test_accepts_bytes(self) -> None:
        assert sha256_hex(b"hello") == sha256_hex("hello")

    def test_different_inputs_produce_different_hashes(self) -> None:
        assert sha256_hex("buy") != sha256_hex("sell")

    def test_output_is_64_chars(self) -> None:
        assert len(sha256_hex("any string")) == 64


class TestSha256File:

    def test_hashes_file_correctly(self, tmp_path: Path) -> None:
        f = tmp_path / "test.txt"
        f.write_text("hello", encoding="utf-8")
        assert sha256_file(f) == sha256_hex("hello")

    def test_raises_for_missing_file(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            sha256_file(tmp_path / "missing.txt")

    def test_different_content_different_hash(self, tmp_path: Path) -> None:
        f1 = tmp_path / "a.txt"
        f2 = tmp_path / "b.txt"
        f1.write_text("buy", encoding="utf-8")
        f2.write_text("sell", encoding="utf-8")
        assert sha256_file(f1) != sha256_file(f2)


# ── HMAC signing ──────────────────────────────────────────────────────────────

class TestSigning:

    def test_sign_and_verify_succeeds(self) -> None:
        sig = sign("buy 1000 shares", "secret-key")
        assert verify_signature("buy 1000 shares", sig, "secret-key") is True

    def test_wrong_key_fails_verification(self) -> None:
        sig = sign("buy 1000 shares", "correct-key")
        assert verify_signature("buy 1000 shares", sig, "wrong-key") is False

    def test_tampered_data_fails_verification(self) -> None:
        sig = sign("buy 1000 shares", "secret")
        assert verify_signature("sell 1000 shares", sig, "secret") is False

    def test_different_data_different_signature(self) -> None:
        assert sign("buy", "key") != sign("sell", "key")

    def test_signature_is_64_chars(self) -> None:
        assert len(sign("data", "key")) == 64


# ── Policy file integrity ─────────────────────────────────────────────────────

class TestPolicyFileIntegrity:

    def test_sign_and_verify_unmodified(self, tmp_path: Path) -> None:
        policy = tmp_path / "policies.yaml"
        policy.write_text("block_large_trades: true\n", encoding="utf-8")
        sig = sign_policy_file(policy, "secret")
        assert verify_policy_file(policy, sig, "secret") is True

    def test_modified_file_fails_verification(self, tmp_path: Path) -> None:
        policy = tmp_path / "policies.yaml"
        policy.write_text("block_large_trades: true\n", encoding="utf-8")
        sig = sign_policy_file(policy, "secret")
        policy.write_text("block_large_trades: false\n", encoding="utf-8")
        assert verify_policy_file(policy, sig, "secret") is False

    def test_missing_file_returns_false(self, tmp_path: Path) -> None:
        assert verify_policy_file(tmp_path / "missing.yaml", "sig", "key") is False

    def test_sign_raises_for_missing_file(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            sign_policy_file(tmp_path / "missing.yaml", "key")


# ── Key generation ────────────────────────────────────────────────────────────

class TestKeyGeneration:

    def test_generates_64_char_key_by_default(self) -> None:
        assert len(generate_secret_key()) == 64

    def test_two_keys_are_unique(self) -> None:
        assert generate_secret_key() != generate_secret_key()

    def test_custom_length(self) -> None:
        assert len(generate_secret_key(16)) == 32


# ── Canonical JSON ────────────────────────────────────────────────────────────

class TestCanonicalJson:

    def test_sorted_keys(self) -> None:
        result = canonical_json({"z": 1, "a": 2})
        assert result == '{"a":2,"z":1}'

    def test_same_data_same_output_regardless_of_insertion_order(self) -> None:
        d1 = {"action": "buy", "amount": 5000}
        d2 = {"amount": 5000, "action": "buy"}
        assert canonical_json(d1) == canonical_json(d2)


# ── Timestamp utilities ───────────────────────────────────────────────────────

class TestTimestamps:

    def test_now_utc_is_string(self) -> None:
        ts = now_utc()
        assert isinstance(ts, str)
        assert "+00:00" in ts or "Z" in ts or "UTC" in ts

    def test_parse_utc_roundtrip(self) -> None:
        ts = now_utc()
        dt = parse_utc(ts)
        assert dt is not None

    def test_parse_utc_rejects_naive_datetime(self) -> None:
        with pytest.raises(ValueError, match="timezone"):
            parse_utc("2025-05-03T22:14:05")

    def test_from_timestamp_produces_utc_string(self) -> None:
        ts_str = from_timestamp(0.0)
        assert "1970" in ts_str

    def test_seconds_between_is_positive_for_past_future(self) -> None:
        t1 = now_utc()
        time.sleep(0.05)
        t2 = now_utc()
        diff = seconds_between(t1, t2)
        assert diff > 0

    def test_is_within_seconds_true_for_recent(self) -> None:
        ts = now_utc()
        assert is_within_seconds(ts, seconds=5.0) is True

    def test_is_within_seconds_false_for_old(self) -> None:
        old_ts = from_timestamp(0.0)
        assert is_within_seconds(old_ts, seconds=10.0) is False