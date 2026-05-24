"""
Unit tests for the pluggable signer interface.
Run with: pytest tests/unit/test_signer.py -v
"""

from __future__ import annotations

import os
import pytest

from aisec.security.signer import HMACSigner, NullSigner, Signer


class TestSignerProtocol:

    def test_hmac_signer_satisfies_protocol(self) -> None:
        signer = HMACSigner(secret_key="a" * 32)
        assert isinstance(signer, Signer)

    def test_null_signer_satisfies_protocol(self) -> None:
        assert isinstance(NullSigner(), Signer)


class TestHMACSigner:

    def test_sign_returns_64_char_hex(self) -> None:
        signer = HMACSigner(secret_key="a" * 32)
        sig = signer.sign(b"test data")
        assert len(sig) == 64
        assert all(c in "0123456789abcdef" for c in sig)

    def test_verify_valid_signature(self) -> None:
        signer = HMACSigner(secret_key="a" * 32)
        data = b"audit entry content"
        sig = signer.sign(data)
        assert signer.verify(data, sig) is True

    def test_verify_rejects_tampered_data(self) -> None:
        signer = HMACSigner(secret_key="a" * 32)
        sig = signer.sign(b"original")
        assert signer.verify(b"tampered", sig) is False

    def test_verify_rejects_wrong_key(self) -> None:
        signer1 = HMACSigner(secret_key="a" * 32)
        signer2 = HMACSigner(secret_key="b" * 32)
        sig = signer1.sign(b"data")
        assert signer2.verify(b"data", sig) is False

    def test_rejects_short_key(self) -> None:
        with pytest.raises(ValueError, match="32 characters"):
            HMACSigner(secret_key="short")

    def test_rejects_missing_key(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("AISEC_SIGNING_KEY", raising=False)
        with pytest.raises(ValueError, match="AISEC_SIGNING_KEY"):
            HMACSigner()

    def test_reads_key_from_environment(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("AISEC_SIGNING_KEY", "e" * 32)
        signer = HMACSigner()
        sig = signer.sign(b"data")
        assert signer.verify(b"data", sig) is True

    def test_repr_does_not_expose_key(self) -> None:
        signer = HMACSigner(secret_key="a" * 32)
        assert "a" * 32 not in repr(signer)
        assert "redacted" in repr(signer)

    def test_two_different_data_produce_different_signatures(self) -> None:
        signer = HMACSigner(secret_key="a" * 32)
        assert signer.sign(b"buy") != signer.sign(b"sell")


class TestNullSigner:

    def test_always_returns_fixed_signature(self) -> None:
        signer = NullSigner()
        assert signer.sign(b"anything") == "null_signature"

    def test_always_verifies_true(self) -> None:
        signer = NullSigner()
        assert signer.verify(b"anything", "anything") is True
