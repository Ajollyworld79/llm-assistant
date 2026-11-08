"""AES-256-GCM encryption for Qdrant payload chunk text.

Vector embeddings are not encrypted (they are not directly readable as text).
Only the human-readable `text` field stored in payloads gets encrypted at rest.

Ciphertext format (base64-url-safe):
    nonce(12 bytes) || ciphertext || tag(16 bytes)

If PAYLOAD_ENCRYPTION_KEY is not set, encryption is a no-op pass-through so the
demo continues to work in local dev without ceremony. The /backend/api/gdpr
endpoint exposes whether encryption is enabled so the demo viewer reflects it.
"""
from __future__ import annotations

import base64
import logging
import os
from typing import Optional

from Crypto.Cipher import AES
from Crypto.Random import get_random_bytes

logger = logging.getLogger(__name__)

_CIPHERTEXT_PREFIX = "enc:v1:"


_warned_bad_key = False


def _get_key() -> Optional[bytes]:
    """Return the 32-byte key, or None when unset/invalid (encryption disabled)."""
    global _warned_bad_key
    raw = os.getenv("PAYLOAD_ENCRYPTION_KEY", "").strip()
    if not raw:
        return None
    try:
        decoded = base64.b64decode(raw, validate=True)
        if len(decoded) == 32:
            return decoded
    except Exception:
        pass
    if not _warned_bad_key:
        _warned_bad_key = True
        logger.error("PAYLOAD_ENCRYPTION_KEY is not base64 of exactly 32 bytes; payload encryption is DISABLED")
    return None


def is_enabled() -> bool:
    return _get_key() is not None


def encrypt_text(plaintext: str) -> str:
    """Encrypt plaintext with AES-256-GCM. Returns prefixed base64 string.

    If no key is configured, returns plaintext unchanged so the demo runs
    without the env var set. Production deploy must set PAYLOAD_ENCRYPTION_KEY.
    """
    if not plaintext:
        return plaintext
    key = _get_key()
    if key is None:
        return plaintext
    nonce = get_random_bytes(12)
    cipher = AES.new(key, AES.MODE_GCM, nonce=nonce)
    ct, tag = cipher.encrypt_and_digest(plaintext.encode("utf-8"))
    payload = nonce + ct + tag
    return _CIPHERTEXT_PREFIX + base64.urlsafe_b64encode(payload).decode("ascii")


def is_encrypted(value: str) -> bool:
    return isinstance(value, str) and value.startswith(_CIPHERTEXT_PREFIX)


def decrypt_text(value: str) -> str:
    """Decrypt a prefixed ciphertext. Pass-through for plaintext values."""
    if not value or not isinstance(value, str):
        return value
    if not value.startswith(_CIPHERTEXT_PREFIX):
        return value
    key = _get_key()
    if key is None:
        # Encrypted at write-time but key missing at read-time: surface a clear marker
        return "[encrypted — decryption key unavailable]"
    try:
        raw = base64.urlsafe_b64decode(value[len(_CIPHERTEXT_PREFIX):])
        if len(raw) < 12 + 16:
            return value
        nonce, body, tag = raw[:12], raw[12:-16], raw[-16:]
        cipher = AES.new(key, AES.MODE_GCM, nonce=nonce)
        return cipher.decrypt_and_verify(body, tag).decode("utf-8")
    except Exception as e:
        logger.warning("decrypt_text failed: %s", e)
        return "[decryption failed]"
