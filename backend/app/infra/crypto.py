"""AES-256-GCM encryption of chunk text stored in Qdrant payloads.

Ciphertext format (``enc:v1:`` + URL-safe base64): ``nonce(12) || ciphertext || tag(16)``.
Only the human-readable ``text`` payload field is encrypted; vectors are not.
Without a key the cipher is a pass-through, which settings validation only allows in DEMO.
"""

from __future__ import annotations

import base64
import binascii
import logging

from Crypto.Cipher import AES
from Crypto.Random import get_random_bytes

logger = logging.getLogger(__name__)

PREFIX = "enc:v1:"
_NONCE_LEN = 12
_TAG_LEN = 16
KEY_UNAVAILABLE = "[encrypted - decryption key unavailable]"
DECRYPT_FAILED = "[decryption failed]"


class CryptoError(ValueError):
    """The configured payload encryption key is unusable."""


def parse_key(raw: str | None) -> bytes | None:
    """Decode a base64 32-byte key. None for unset; CryptoError for a malformed key."""
    if not raw or not raw.strip():
        return None
    try:
        key = base64.b64decode(raw.strip(), validate=True)
    except (binascii.Error, ValueError) as exc:
        raise CryptoError("PAYLOAD_ENCRYPTION_KEY is not valid base64") from exc
    if len(key) != 32:
        raise CryptoError("PAYLOAD_ENCRYPTION_KEY must decode to exactly 32 bytes")
    return key


class PayloadCipher:
    def __init__(self, key: bytes | None) -> None:
        self._key = key

    @classmethod
    def from_setting(cls, raw: str | None) -> PayloadCipher:
        return cls(parse_key(raw))

    @property
    def enabled(self) -> bool:
        return self._key is not None

    @staticmethod
    def is_encrypted(value: object) -> bool:
        return isinstance(value, str) and value.startswith(PREFIX)

    def encrypt(self, plaintext: str) -> str:
        if not plaintext or self._key is None:
            return plaintext
        nonce = get_random_bytes(_NONCE_LEN)
        cipher = AES.new(self._key, AES.MODE_GCM, nonce=nonce)
        body, tag = cipher.encrypt_and_digest(plaintext.encode("utf-8"))
        return PREFIX + base64.urlsafe_b64encode(nonce + body + tag).decode("ascii")

    def decrypt(self, value: str) -> str:
        """Decrypt a prefixed value; plaintext values pass through unchanged."""
        if not value or not value.startswith(PREFIX):
            return value
        if self._key is None:
            return KEY_UNAVAILABLE
        try:
            raw = base64.urlsafe_b64decode(value[len(PREFIX) :])
            if len(raw) < _NONCE_LEN + _TAG_LEN:
                return DECRYPT_FAILED
            nonce, body, tag = raw[:_NONCE_LEN], raw[_NONCE_LEN:-_TAG_LEN], raw[-_TAG_LEN:]
            cipher = AES.new(self._key, AES.MODE_GCM, nonce=nonce)
            return cipher.decrypt_and_verify(body, tag).decode("utf-8")
        except (ValueError, binascii.Error):
            logger.warning("payload decryption failed (wrong key or corrupted data)")
            return DECRYPT_FAILED
