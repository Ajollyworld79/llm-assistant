"""Gemini chat completion with structured output, primary -> fallback model, and a demo emulator."""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any

from backend.app.core.config import Settings
from backend.app.services.prompt import SOURCE_TO_INDICATOR, SYSTEM_PROMPT, ChatAnswer

logger = logging.getLogger(__name__)

TRUNCATION_NOTE = "_(svar blev afkortet — overvej at omformulere spørgsmålet for et kortere svar)_"
_ESCAPES = {"n": "\n", "t": "\t", "r": "\r", '"': '"', "\\": "\\", "/": "/"}


@dataclass(frozen=True)
class ImageInput:
    mime_type: str
    data: bytes


@dataclass
class Usage:
    model: str | None = None
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cached_tokens: int = 0
    total_tokens: int = 0
    fallback: bool = False


@dataclass
class LLMResult:
    answer: str
    indicator: int | None = None
    usage: Usage = field(default_factory=Usage)
    blocked: bool = False
    ok: bool = True


def recover_truncated_answer(raw: str) -> tuple[str, int | None]:
    """Salvage ``answer_markdown`` from JSON cut off by the output token limit."""
    logger.warning("Structured output JSON parse failed (raw length %d)", len(raw))
    match = re.search(r'"answer_markdown"\s*:\s*"', raw)
    if not match:
        return raw.strip(), None

    i, n = match.end(), len(raw)
    out: list[str] = []
    while i < n:
        ch = raw[i]
        if ch == "\\" and i + 1 < n:
            nxt = raw[i + 1]
            if nxt in _ESCAPES:
                out.append(_ESCAPES[nxt])
                i += 2
                continue
            if nxt == "u" and i + 5 < n:
                try:
                    out.append(chr(int(raw[i + 2 : i + 6], 16)))
                    i += 6
                    continue
                except ValueError:
                    pass
            out.append(nxt)
            i += 2
            continue
        if ch == '"':
            break
        out.append(ch)
        i += 1

    recovered = "".join(out).strip()
    if recovered:
        return f"{recovered}\n\n{TRUNCATION_NOTE}", 1
    return raw.strip(), None


def parse_structured_response(raw: str) -> tuple[str, int | None]:
    """Parse the model's JSON into (answer_markdown, source indicator)."""
    if not raw:
        return raw, None
    candidate = raw.strip()
    if candidate.startswith("```"):
        candidate = candidate.strip("`")
        if candidate.lower().startswith("json"):
            candidate = candidate[4:]
        candidate = candidate.strip()

    obj: Any
    try:
        obj = json.loads(candidate)
    except ValueError:
        start, end = candidate.find("{"), candidate.rfind("}")
        if start < 0 or end <= start:
            return recover_truncated_answer(raw)
        try:
            obj = json.loads(candidate[start : end + 1])
        except ValueError:
            return recover_truncated_answer(raw)
    if not isinstance(obj, dict):
        return recover_truncated_answer(raw)

    answer = str(obj.get("answer_markdown") or "").strip()
    return answer, SOURCE_TO_INDICATOR.get(obj.get("source") or "general_knowledge", 1)


class LLMService:
    def __init__(self, settings: Settings, client: Any | None) -> None:
        self.settings = settings
        self._client = client

    @property
    def configured(self) -> bool:
        return self._client is not None

    def _build_contents(self, query: str, context: str, images: list[ImageInput]) -> list[Any]:
        from google.genai import types

        limit = self.settings.max_context_chars
        if len(context) > limit:
            context = context[:limit] + "\n\n...[truncated]..."
        parts = [types.Part(text=f"Konteksten:\n{context}\n\nSpørgsmål: {query}")]
        parts.extend(types.Part.from_bytes(data=img.data, mime_type=img.mime_type) for img in images)
        return [types.Content(role="user", parts=parts)]

    async def _call(self, model: str, contents: list[Any]) -> Any:
        from google.genai import types

        config = types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            temperature=self.settings.temperature,
            max_output_tokens=self.settings.max_tokens,
            response_mime_type="application/json",
            response_schema=ChatAnswer,
        )
        if self._client is None:
            raise RuntimeError("Gemini client not configured")
        return await self._client.aio.models.generate_content(model=model, contents=contents, config=config)

    @staticmethod
    def _usage(response: Any, model: str, fallback: bool) -> Usage:
        meta = getattr(response, "usage_metadata", None)
        prompt = int(getattr(meta, "prompt_token_count", 0) or 0)
        completion = int(getattr(meta, "candidates_token_count", 0) or 0)
        total = int(getattr(meta, "total_token_count", 0) or 0) or prompt + completion
        cached = int(getattr(meta, "cached_content_token_count", 0) or 0)
        return Usage(model, prompt, completion, cached, total, fallback)

    @staticmethod
    def _is_block(exc: Exception) -> bool:
        text = str(exc).lower()
        return "blocked" in text or "safety" in text

    async def answer(self, query: str, context: str, images: list[ImageInput] | None = None) -> LLMResult:
        if self._client is None:
            return LLMResult("Sorry, the assistant is currently unavailable.", ok=False)
        s = self.settings
        if not s.gemini_model:
            return LLMResult("Sorry, the assistant is not configured.", ok=False)

        contents = self._build_contents(query, context, images or [])
        models = [(s.gemini_model, False)]
        if s.gemini_model_fallback and s.gemini_model_fallback != s.gemini_model:
            models.append((s.gemini_model_fallback, True))

        blocked = False
        for model, is_fallback in models:
            try:
                response = await self._call(model, contents)
            except Exception as exc:
                blocked = blocked or self._is_block(exc)
                logger.warning("Model %s failed (%s)", model, type(exc).__name__)
                continue
            text, indicator = parse_structured_response(getattr(response, "text", "") or "")
            return LLMResult(text, indicator, self._usage(response, model, is_fallback))

        if blocked:
            return LLMResult("Content blocked by safety filters.", blocked=True, ok=False)
        return LLMResult("Sorry, the assistant is currently unavailable.", ok=False)


class DemoLLM:
    """Deterministic demo emulator: echoes the retrieved context."""

    def __init__(self, max_chars: int = 800) -> None:
        self.max_chars = max_chars

    def generate(self, context: str) -> str:
        if not context.strip():
            return "Jeg har ingen dokumenter at svare ud fra. Upload nogle dokumenter, så hjælper jeg."
        parts = [p.strip() for p in context.split("\n\n") if p.strip()][:3]
        body = "\n\n".join(parts)
        if len(body) > self.max_chars:
            body = body[: self.max_chars - 3].rstrip() + "..."
        return body
