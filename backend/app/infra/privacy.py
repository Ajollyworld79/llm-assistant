"""Best-effort PII scrubbing applied before chat history is stored."""

from __future__ import annotations

import re

# Most specific first so a CPR number is not partially eaten by the phone pattern.
_PATTERNS = (
    (re.compile(r"(?<!\w)\d{6}[-\s]?\d{4}(?!\w)"), "[cpr-redacted]"),
    (re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b"), "[email-redacted]"),
    (re.compile(r"(?<!\w)(?:\+45[\s-]?)?(?:\d{2}[\s-]?){3}\d{2}(?!\w)"), "[phone-redacted]"),
)


def redact_pii(text: str) -> str:
    """Replace e-mail addresses, Danish CPR numbers and phone numbers with placeholders."""
    if not text:
        return text
    for pattern, replacement in _PATTERNS:
        text = pattern.sub(replacement, text)
    return text
