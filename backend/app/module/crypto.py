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


def _get_key() -> Optional[bytes]:
    raw = os.getenv("PAYLOAD_ENCRYPTION_KEY", "").strip()
    if not raw:
        return None
    # Accept either base64 (32 bytes decoded) or raw 32-char passphrase
    try:
        decoded = base64.b64decode(raw, validate=True)
        if len(decoded) == 32:
            return decoded
    except Exception:
        pass
    # Fallback: hash a non-base64 passphrase to 32 bytes
    from hashlib import sha256
    return sha256(raw.encode("utf-8")).digest()


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
    try:
        nonce = get_random_bytes(12)
        cipher = AES.new(key, AES.MODE_GCM, nonce=nonce)
        ct, tag = cipher.encrypt_and_digest(plaintext.encode("utf-8"))
        payload = nonce + ct + tag
        return _CIPHERTEXT_PREFIX + base64.urlsafe_b64encode(payload).decode("ascii")
    except Exception as e:
        logger.exception("encrypt_text failed; storing plaintext as fallback: %s", e)
        return plaintext


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
