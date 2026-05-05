"""
AISec cryptographic utilities.

Provides hashing, policy file signing, and signature
verification used throughout the system to detect tampering.

Design principles:
  - All functions are pure — no side effects, no global state.
  - Signatures use HMAC-SHA256 with a caller-supplied secret key.
  - The secret key is never stored by this module.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import secrets
from pathlib import Path
from typing import Any


# ── Hashing ───────────────────────────────────────────────────────────────────

def sha256_hex(data: str | bytes) -> str:
    """
    Return the SHA-256 hex digest of the given data.

    Args:
        data: A string (UTF-8 encoded) or bytes object.

    Returns:
        64-character lowercase hex string.

    Example:
        >>> sha256_hex("hello")
        '2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824'
    """
    if isinstance(data, str):
        data = data.encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    """
    Return the SHA-256 hex digest of a file's contents.

    Reads the file in chunks to avoid loading large files
    into memory all at once.

    Args:
        path: Path to the file to hash.

    Returns:
        64-character lowercase hex string.

    Raises:
        FileNotFoundError: If the file does not exist.
    """
    if not path.exists():
        raise FileNotFoundError(f"Cannot hash missing file: {path}")

    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(65_536), b""):
            digest.update(chunk)
    return digest.hexdigest()


# ── HMAC signing ──────────────────────────────────────────────────────────────

def sign(data: str | bytes, secret_key: str) -> str:
    """
    Compute an HMAC-SHA256 signature over the given data.

    Args:
        data:       The content to sign.
        secret_key: Caller-supplied secret. Never stored here.

    Returns:
        64-character lowercase hex signature.
    """
    if isinstance(data, str):
        data = data.encode("utf-8")
    key_bytes = secret_key.encode("utf-8")
    return hmac.new(key_bytes, data, hashlib.sha256).hexdigest()


def verify_signature(data: str | bytes, signature: str, secret_key: str) -> bool:
    """
    Verify an HMAC-SHA256 signature using constant-time comparison.

    Constant-time comparison prevents timing side-channel attacks
    where an attacker could deduce the correct signature by measuring
    how long the comparison takes.

    Args:
        data:       The original content that was signed.
        signature:  The signature to verify.
        secret_key: The same secret used when signing.

    Returns:
        True if the signature is valid, False otherwise.
    """
    expected = sign(data, secret_key)
    return hmac.compare_digest(expected, signature)


# ── Policy file integrity ─────────────────────────────────────────────────────

def sign_policy_file(path: Path, secret_key: str) -> str:
    """
    Compute and return an HMAC signature over a policy file's contents.

    The caller is responsible for storing this signature alongside
    the policy file. On startup, call verify_policy_file() to confirm
    the file has not been modified.

    Args:
        path:       Path to the policy file (YAML or JSON).
        secret_key: Secret key for signing.

    Returns:
        HMAC-SHA256 hex signature of the file contents.

    Raises:
        FileNotFoundError: If the file does not exist.
    """
    if not path.exists():
        raise FileNotFoundError(f"Policy file not found: {path}")
    contents = path.read_text(encoding="utf-8")
    return sign(contents, secret_key)


def verify_policy_file(path: Path, signature: str, secret_key: str) -> bool:
    """
    Verify that a policy file has not been modified since it was signed.

    Args:
        path:       Path to the policy file.
        signature:  Previously stored signature from sign_policy_file().
        secret_key: The same secret used when signing.

    Returns:
        True if the file is unmodified, False if it has been tampered with.
    """
    if not path.exists():
        return False
    contents = path.read_text(encoding="utf-8")
    return verify_signature(contents, signature, secret_key)


# ── Key generation ────────────────────────────────────────────────────────────

def generate_secret_key(length: int = 32) -> str:
    """
    Generate a cryptographically secure random secret key.

    Uses Python's secrets module which draws from the OS
    entropy pool — suitable for production use.

    Args:
        length: Number of random bytes before hex encoding.
                Default 32 bytes = 256-bit security.

    Returns:
        Hex-encoded string of length * 2 characters.
    """
    return secrets.token_hex(length)


# ── Canonical serialisation ───────────────────────────────────────────────────

def canonical_json(data: dict[str, Any]) -> str:
    """
    Serialise a dictionary to a canonical JSON string.

    Keys are sorted so that the same data always produces
    the same string regardless of insertion order.
    This is used when hashing structured payloads.

    Args:
        data: Dictionary to serialise.

    Returns:
        Compact, sorted JSON string with no extra whitespace.
    """
    return json.dumps(data, sort_keys=True, separators=(",", ":"))