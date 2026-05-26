import json
import logging
import asyncio
from typing import Optional, Tuple, Any, List, Dict

try:
    from google import genai
    from google.genai import types as genai_types
except ImportError:
    genai = None
    genai_types = None

try:
    from .prompt import SYSTEM_PROMPT, SOURCE_TO_INDICATOR, ChatAnswer
except ImportError:
    from prompt import SYSTEM_PROMPT, SOURCE_TO_INDICATOR, ChatAnswer

try:
    from backend.app.config import settings
except ImportError:
    try:
        import config
        settings = config.settings
    except ImportError:
        pass

try:
    from backend.app.module.Functions_module import setup_logger
    logger = setup_logger()
except ImportError:
    try:
        from module.Functions_module import setup_logger
        logger = setup_logger()
    except ImportError:
        logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(name)s %(message)s')
        logger = logging.getLogger(__name__)


class AIManager:
    """Chat completions backed by the free Gemini API (google-genai)."""

    def __init__(self, settings_obj, embeddings_module, client=None):
        self.settings = settings_obj
        self.embeddings = embeddings_module
        self._client = client
        self.last_used_model: Optional[str] = None
        self.last_total_tokens: int = 0
        self.last_cached_tokens: int = 0  # Gemini does not expose this; kept for metric compatibility
        self.fallback_invocation_count: int = 0
        # Pydantic class — google-genai converts to Gemini's schema format internally.
        self._response_schema = ChatAnswer

    def initialize_client(self) -> None:
        if self._client is not None:
            return
        if genai is None:
            logger.warning("google-genai not installed; AI calls will be disabled")
            return
        api_key = getattr(self.settings, "gemini_api_key", None)
        if not api_key:
            logger.warning("GEMINI_API_KEY not set; AI calls will be disabled")
            return
        try:
            self._client = genai.Client(api_key=api_key)
            logger.info("AIManager: Gemini client initialized")
        except Exception as e:
            logger.exception("AIManager: failed to initialize Gemini client: %s", e)
            self._client = None

    def _build_contents(self, query: str, context: str, history: List[Dict[str, str]],
                        images: Optional[List[Dict[str, str]]] = None) -> List[Any]:
        max_ctx = int(getattr(self.settings, "MAX_CONTEXT_CHARS", 4000))
        ctx = (context[:max_ctx] + "\n\n...[truncated]...") if context and len(context) > max_ctx else (context or "")

        contents: List[Any] = []
        max_history = int(getattr(self.settings, "MAX_CONVERSATION_HISTORY", 6))
        for msg in (history or [])[-max_history:]:
            if not isinstance(msg, dict):
                continue
            u = msg.get("user", "")
            a = msg.get("assistant", "")
            if u:
                contents.append(genai_types.Content(role="user", parts=[genai_types.Part(text=u)]))
            if a:
                contents.append(genai_types.Content(role="model", parts=[genai_types.Part(text=a)]))

        user_msg = f"Konteksten:\n{ctx}\n\nSpørgsmål: {query}"
        parts: List[Any] = [genai_types.Part(text=user_msg)]
        import base64
        for img in (images or []):
            data_b64 = img.get("data_base64") or ""
            mime = img.get("mime_type") or "image/png"
            if not data_b64:
                continue
            try:
                raw = base64.b64decode(data_b64)
                parts.append(genai_types.Part.from_bytes(data=raw, mime_type=mime))
            except Exception as e:
                logger.warning("Skipping invalid image attachment: %s", e)
        contents.append(genai_types.Content(role="user", parts=parts))
        return contents

    async def _call_gemini(self, model: str, contents: List[Any]):
        temperature = float(getattr(self.settings, "TEMPERATURE", 0.1))
        max_tokens = int(getattr(self.settings, "MAX_TOKENS", 400))
        config = genai_types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            temperature=temperature,
            max_output_tokens=max_tokens,
            response_mime_type="application/json",
            response_schema=self._response_schema,
        )
        return await self._client.aio.models.generate_content(
            model=model,
            contents=contents,
            config=config,
        )

    def _record_usage(self, response, model_used: str) -> None:
        self.last_used_model = model_used
        try:
            usage = getattr(response, "usage_metadata", None)
            if usage is None:
                return
            prompt_tokens = int(getattr(usage, "prompt_token_count", 0) or 0)
            completion_tokens = int(getattr(usage, "candidates_token_count", 0) or 0)
            self.last_total_tokens = int(getattr(usage, "total_token_count", prompt_tokens + completion_tokens) or 0)
            self.last_cached_tokens = int(getattr(usage, "cached_content_token_count", 0) or 0)
        except Exception as e:
            logger.debug("Failed to record Gemini usage: %s", e)
            return

        try:
            try:
                from backend.app.module.rate_limiter import get_limiter
            except ImportError:
                from module.rate_limiter import get_limiter
            asyncio.create_task(get_limiter().record_tokens(self.last_total_tokens))
        except Exception as e:
            logger.debug("Failed to record tokens to limiter: %s", e)

        try:
            try:
                from backend.app.module.apimonitor import monitor as _mon
            except ImportError:
                from module.apimonitor import monitor as _mon
            _mon.record_cache_stats(prompt_tokens, self.last_cached_tokens)
        except Exception as e:
            logger.debug("Failed to record cache stats: %s", e)

        try:
            try:
                from backend.app.module.usage_db import usage_db
            except ImportError:
                from module.usage_db import usage_db
            is_fallback = (model_used == getattr(self.settings, "gemini_model_fallback", ""))
            usage_db.log_event(
                session_id=None,
                endpoint="/search",
                model=model_used,
                tokens_input=prompt_tokens,
                tokens_output=completion_tokens,
                tokens_cached=self.last_cached_tokens,
                latency_ms=0,
                status_code=200,
                error_kind="fallback" if is_fallback else None,
            )
        except Exception as e:
            logger.debug("Failed to log usage event: %s", e)

    async def ask_agent_with_context(self, query: str, context: str,
                                     conversation_history: Optional[list] = None,
                                     images: Optional[List[Dict[str, str]]] = None) -> Tuple[str, Optional[int]]:
        if conversation_history is None:
            conversation_history = []

        if getattr(self.settings, 'demo', False):
            demo = DemoLLM()
            try:
                raw, _ = await demo.generate(query, context)
                return self._parse_structured_response(raw)
            except Exception as e:
                logger.exception('DemoLLM failed: %s', e)
                return ("(Demo) Sorry, demo generation failed.", None)

        if self._client is None:
            self.initialize_client()
        if self._client is None:
            return ("Sorry, the assistant is currently unavailable.", None)

        primary = getattr(self.settings, "gemini_model", "") or ""
        fallback = getattr(self.settings, "gemini_model_fallback", "") or ""
        if not primary:
            logger.warning("AIManager: no Gemini model configured")
            return ("Sorry, the assistant is not configured.", None)

        contents = self._build_contents(query, context, conversation_history, images=images)

        try:
            response = await self._call_gemini(primary, contents)
            raw = getattr(response, "text", "") or ""
            self._record_usage(response, primary)
            return self._parse_structured_response(raw)
        except Exception as e:
            err_text = str(e).lower()
            if "blocked" in err_text or "safety" in err_text:
                if not fallback:
                    return "__CONTENT_FILTER_BLOCKED__", None
            logger.warning("Primary model %s failed (%s); attempting fallback", primary, e.__class__.__name__)

        if not fallback:
            return ("Sorry, something went wrong with the assistant.", None)

        try:
            self.fallback_invocation_count += 1
            try:
                try:
                    from backend.app.module.apimonitor import monitor as _mon
                except ImportError:
                    from module.apimonitor import monitor as _mon
                _mon.record_fallback_invocation()
            except Exception:
                pass
            response = await self._call_gemini(fallback, contents)
            raw = getattr(response, "text", "") or ""
            self._record_usage(response, fallback)
            return self._parse_structured_response(raw)
        except Exception as e:
            err_text = str(e).lower()
            if "blocked" in err_text or "safety" in err_text:
                return "__CONTENT_FILTER_BLOCKED__", None
            logger.exception("Fallback model %s also failed: %s", fallback, e)
            return ("Sorry, the assistant is currently unavailable.", None)

    def _parse_structured_response(self, raw: str) -> Tuple[str, Optional[int]]:
        if not raw:
            return raw, None

        candidate = raw.strip()
        # Gemini occasionally wraps JSON in ```json ... ``` fences despite response_mime_type.
        if candidate.startswith("```"):
            candidate = candidate.strip("`")
            if candidate.lower().startswith("json"):
                candidate = candidate[4:]
            candidate = candidate.strip()

        try:
            obj = json.loads(candidate)
        except Exception:
            # Last-ditch: try to find the first {...} block inside the text.
            start = candidate.find("{")
            end = candidate.rfind("}")
            if start >= 0 and end > start:
                try:
                    obj = json.loads(candidate[start:end + 1])
                except Exception:
                    return self._recover_truncated_answer(raw)
            else:
                return self._recover_truncated_answer(raw)

        answer = (obj.get("answer_markdown") or "").strip()
        source = obj.get("source") or "general_knowledge"
        indicator = SOURCE_TO_INDICATOR.get(source, 1)
        return answer, indicator

    def _recover_truncated_answer(self, raw: str) -> Tuple[str, Optional[int]]:
        """Salvage answer_markdown from a truncated/incomplete JSON response.

        Gemini sometimes hits max_output_tokens mid-string, producing JSON that
        json.loads cannot parse. Rather than dump raw JSON into the UI, extract
        the answer_markdown value byte-by-byte respecting JSON escape rules.
        """
        import re
        logger.warning(f"Structured output JSON parse failed (raw len={len(raw)}): {raw[:400]!r}")

        m = re.search(r'"answer_markdown"\s*:\s*"', raw)
        if not m:
            return raw.strip(), None

        i = m.end()
        out_chars = []
        n = len(raw)
        while i < n:
            ch = raw[i]
            if ch == '\\' and i + 1 < n:
                nxt = raw[i + 1]
                escapes = {'n': '\n', 't': '\t', 'r': '\r', '"': '"', '\\': '\\', '/': '/'}
                if nxt in escapes:
                    out_chars.append(escapes[nxt])
                    i += 2
                    continue
                if nxt == 'u' and i + 5 < n:
                    try:
                        out_chars.append(chr(int(raw[i + 2:i + 6], 16)))
                        i += 6
                        continue
                    except Exception:
                        pass
                out_chars.append(nxt)
                i += 2
                continue
            if ch == '"':
                break
            out_chars.append(ch)
            i += 1

        recovered = ''.join(out_chars).strip()
        if recovered:
            recovered += '\n\n_(svar blev afkortet — overvej at omformulere spørgsmålet for et kortere svar)_'
            return recovered, 1
        return raw.strip(), None


class DemoLLM:
    """Deterministic demo-mode emulator. Returns structured-output-shaped JSON."""

    def __init__(self, max_chars: int = 800):
        self.max_chars = max_chars

    async def generate(self, query: str, context: str) -> Tuple[str, Optional[int]]:
        if not context or not context.strip():
            payload = {
                "source": "refusal",
                "language": "da",
                "answer_markdown": "Jeg har ingen dokumenter at svare ud fra. Upload nogle dokumenter, så hjælper jeg.",
                "refusal_reason": "no_documents",
            }
            return json.dumps(payload), None

        parts = [p.strip() for p in context.split("\n\n") if p.strip()][:3]
        body = "\n\n".join(parts)
        if len(body) > self.max_chars:
            body = body[: self.max_chars - 3].rstrip() + "..."

        lower = body.lower()
        source = "documents" if not any(x in lower for x in ("microsoft", "windows", "office")) else "general_knowledge"
        payload = {
            "source": source,
            "language": "da",
            "answer_markdown": body,
            "refusal_reason": None,
        }
        return json.dumps(payload), SOURCE_TO_INDICATOR[source]
