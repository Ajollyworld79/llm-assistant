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
    from .prompt import SYSTEM_PROMPT, SUPPORT_ANSWER_SCHEMA, SOURCE_TO_INDICATOR
except ImportError:
    from prompt import SYSTEM_PROMPT, SUPPORT_ANSWER_SCHEMA, SOURCE_TO_INDICATOR

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


def _schema_to_genai(schema_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Strip JSON-schema fields Gemini does not accept while keeping the structure."""
    inner = schema_dict.get("schema", schema_dict)
    drop_keys = {"additionalProperties", "strict"}

    def _clean(node):
        if isinstance(node, dict):
            cleaned = {k: _clean(v) for k, v in node.items() if k not in drop_keys}
            # Gemini does not accept union types like ["string", "null"]; collapse to string.
            t = cleaned.get("type")
            if isinstance(t, list):
                non_null = [x for x in t if x != "null"]
                cleaned["type"] = non_null[0] if non_null else "string"
            return cleaned
        if isinstance(node, list):
            return [_clean(x) for x in node]
        return node

    return _clean(inner)


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
        self._response_schema = _schema_to_genai(SUPPORT_ANSWER_SCHEMA)

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
                except Exception as e:
                    logger.warning("Structured output JSON parse failed (raw len=%d): %r", len(raw), raw[:200])
                    return raw.strip(), None
            else:
                logger.warning("Structured output JSON parse failed (raw len=%d): %r", len(raw), raw[:200])
                return raw.strip(), None

        answer = (obj.get("answer_markdown") or "").strip()
        source = obj.get("source") or "general_knowledge"
        indicator = SOURCE_TO_INDICATOR.get(source, 1)
        return answer, indicator


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
