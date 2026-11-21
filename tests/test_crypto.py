import base64

import pytest

from backend.app.infra.crypto import DECRYPT_FAILED, KEY_UNAVAILABLE, CryptoError, PayloadCipher, parse_key

KEY_B64 = "AAECAwQFBgcICQoLDA0ODxAREhMUFRYXGBkaGxwdHh8="  # bytes 0..31
# Produced once with pycryptodome 3.23.0; guards the stored enc:v1 format across library bumps.
KNOWN_VECTOR = "enc:v1:ZGVmZ2hpamtsbW5vANh4SlmilSZcBzGAuxME3GLzNDkJhbWWzsuDv02L7YZn1fA0"
KNOWN_PLAINTEXT = "Hæ, København! 123"


def cipher(raw: str | None = KEY_B64) -> PayloadCipher:
    return PayloadCipher.from_setting(raw)


def test_known_ciphertext_still_decrypts():
    assert cipher().decrypt(KNOWN_VECTOR) == KNOWN_PLAINTEXT


def test_round_trip_and_randomized_nonce():
    c = cipher()
    a, b = c.encrypt("secret text"), c.encrypt("secret text")
    assert a != b and a.startswith("enc:v1:")
    assert c.decrypt(a) == c.decrypt(b) == "secret text"
    assert c.is_encrypted(a) and not c.is_encrypted("plain")


def test_wrong_key_reports_failure():
    other = base64.b64encode(bytes(range(1, 33))).decode()
    assert cipher(other).decrypt(KNOWN_VECTOR) == DECRYPT_FAILED


def test_missing_key_pass_through_and_marker():
    c = cipher(None)
    assert not c.enabled
    assert c.encrypt("plain") == "plain"
    assert c.decrypt("plain") == "plain"
    assert c.decrypt(KNOWN_VECTOR) == KEY_UNAVAILABLE


def test_truncated_or_tampered_ciphertext():
    c = cipher()
    assert c.decrypt("enc:v1:AAAA") == DECRYPT_FAILED
    tampered = KNOWN_VECTOR[:-4] + "AAAA"
    assert c.decrypt(tampered) == DECRYPT_FAILED
    assert c.decrypt("") == ""


@pytest.mark.parametrize("raw", ["not base64!!", base64.b64encode(b"short").decode()])
def test_invalid_key_raises(raw):
    with pytest.raises(CryptoError):
        parse_key(raw)


def test_empty_plaintext_not_encrypted():
    assert cipher().encrypt("") == ""
