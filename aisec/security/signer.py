"""
AISec pluggable signing interface.

Provides a Protocol that all signing implementations must satisfy.
This allows swapping between local HMAC signing (development),
cloud KMS signing (production), and HSM signing (enterprise)
without changing any calling code.

Current implementations:
    HMACSigner  — HMAC-SHA256, suitable for development and testing.
                  Keys stored in environment variable or config file.

Planned implementations (v2):
    AWSSigner   — AWS KMS backed signing.
    AzureSigner — Azure Key Vault backed signing.
    HSMSigner   — PKCS#11 hardware security module.

Security requirement:
    Private keys must NEVER be stored in source code or committed
    to version control. Use environment variables or a secrets manager.
"""

from __future__ import annotations

import hmac
import hashlib
import os
from typing import Protocol, runtime_checkable


# ── Signer protocol ───────────────────────────────────────────────────────────

@runtime_checkable
class Signer(Protocol):
    """
    Interface that all AISec signing implementations must satisfy.

    Any class that implements sign() and verify() with these
    signatures is a valid Signer — no inheritance required.
    """

    def sign(self, data: bytes) -> str:
        """
        Compute a cryptographic signature over data.

        Args:
            data: Raw bytes to sign.

        Returns:
            Hex-encoded signature string.
        """
        ...

    def verify(self, data: bytes, signature: str) -> bool:
        """
        Verify a signature over data.

        Args:
            data:      The original bytes that were signed.
            signature: The hex-encoded signature to verify.

        Returns:
            True if signature is valid, False otherwise.
            Must use constant-time comparison to prevent
            timing side-channel attacks.
        """
        ...


# ── HMAC implementation ───────────────────────────────────────────────────────

class HMACSigner:
    """
    HMAC-SHA256 signing implementation for development use.

    The secret key is read from:
        1. Constructor argument (highest priority)
        2. AISEC_SIGNING_KEY environment variable
        3. Raises ValueError if neither is available

    Security warning:
        Do not use this in production without storing the key
        in a proper secrets manager. Never commit the key to
        version control.

    Usage:
        # From environment variable (recommended)
        signer = HMACSigner()

        # From explicit key (testing only)
        signer = HMACSigner(secret_key="test-key-never-in-prod")

        sig = signer.sign(b"audit entry content")
        ok  = signer.verify(b"audit entry content", sig)
    """

    ENV_VAR = "AISEC_SIGNING_KEY"

    def __init__(self, secret_key: str | None = None) -> None:
        """
        Args:
            secret_key: Secret key for HMAC signing.
                        If None, reads from AISEC_SIGNING_KEY env var.

        Raises:
            ValueError: If no key is available from any source.
        """
        key = secret_key or os.environ.get(self.ENV_VAR)
        if not key:
            raise ValueError(
                f"No signing key provided. Set the {self.ENV_VAR} "
                "environment variable or pass secret_key explicitly.\n"
                "Generate a key with: python -c \"import secrets; "
                "print(secrets.token_hex(32))\""
            )
        if len(key) < 32:
            raise ValueError(
                "Signing key must be at least 32 characters long. "
                "Use secrets.token_hex(32) to generate a strong key."
            )
        # Store as bytes — never log or expose the key
        self._key: bytes = key.encode("utf-8")

    def sign(self, data: bytes) -> str:
        """
        Compute HMAC-SHA256 signature over data.

        Args:
            data: Raw bytes to sign.

        Returns:
            64-character hex-encoded HMAC signature.
        """
        return hmac.new(self._key, data, hashlib.sha256).hexdigest()

    def verify(self, data: bytes, signature: str) -> bool:
        """
        Verify HMAC-SHA256 signature using constant-time comparison.

        Constant-time comparison prevents timing side-channel attacks.

        Args:
            data:      The original bytes that were signed.
            signature: The hex-encoded signature to verify.

        Returns:
            True if signature is valid, False otherwise.
        """
        expected = self.sign(data)
        return hmac.compare_digest(expected, signature)

    def __repr__(self) -> str:
        """Never expose the key in string representation."""
        return "HMACSigner(key=<redacted>)"


# ── Null signer (testing only) ────────────────────────────────────────────────

class NullSigner:
    """
    No-op signer for unit testing only.

    NEVER use in production. This signer accepts any signature
    as valid, which provides zero security.
    """

    def sign(self, data: bytes) -> str:
        return "null_signature"

    def verify(self, data: bytes, signature: str) -> bool:
        return True

    def __repr__(self) -> str:
        return "NullSigner(WARNING: no-op, testing only)"