import json
import logging
import asyncio
from typing import Optional, Tuple, Any, List, Dict, cast

try:
    from openai import AsyncAzureOpenAI
except ImportError:
    AsyncAzureOpenAI = None

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



class AIManager:
    """Central AI manager for chat completions with prompt-caching layout and fallback model."""

    def __init__(self, settings_obj, embeddings_module, client=None):
        self.settings = settings_obj
        self.embeddings = embeddings_module
        self._client = client
        # Aggregate counters surfaced via /monitor and /backend/api/metrics
        self.last_used_model: Optional[str] = None
        self.last_cached_tokens: int = 0
        self.last_total_tokens: int = 0
        self.fallback_invocation_count: int = 0

    def initialize_client(self) -> None:
        if self._client is None:
            if AsyncAzureOpenAI is None:
                logger.warning("AsyncAzureOpenAI is not available; AI calls will be disabled")
                return
            try:
                self._client = AsyncAzureOpenAI(
                    api_key=cast(str, getattr(self.settings, "azure_openai_api_key", "")),
                    api_version=cast(str, getattr(self.settings, "azure_openai_api_version", "")),
                    azure_endpoint=cast(str, getattr(self.settings, "azure_openai_endpoint", "")),
                )
                logger.info("AIManager: AsyncAzureOpenAI client initialized")
            except Exception as e:
                logger.exception("AIManager: failed to initialize client: %s", e)
                self._client = None

    def _build_messages(self, query: str, context: str, history: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """Build messages with stable system prefix to maximize cache hits.

        Layout: system (stable) → history (stable within session) → current user (dynamic).
        Retrieved context is placed INSIDE the current user message, not the system,
        because context varies per query and would otherwise invalidate the cache.
        """
        max_ctx = int(getattr(self.settings, "MAX_CONTEXT_CHARS", 4000))
        ctx = (context[:max_ctx] + "\n\n...[truncated]...") if context and len(context) > max_ctx else (context or "")

        messages: List[Dict[str, str]] = [
            {"role": "system", "content": SYSTEM_PROMPT}
        ]

        max_history = int(getattr(self.settings, "MAX_CONVERSATION_HISTORY", 6))
        for msg in (history or [])[-max_history:]:
            if not isinstance(msg, dict):
                continue
            u = msg.get("user", "")
            a = msg.get("assistant", "")
            if u:
                messages.append({"role": "user", "content": u})
            if a:
                messages.append({"role": "assistant", "content": a})

        user_msg = f"Konteksten:\n{ctx}\n\nSpørgsmål: {query}"
        messages.append({"role": "user", "content": user_msg})
        return messages

    async def _call_azure(self, model: str, messages: List[Dict[str, str]]):
        temperature = float(getattr(self.settings, "TEMPERATURE", 0.1))
        max_tokens = int(getattr(self.settings, "MAX_TOKENS", 400))
        response_format = {"type": "json_schema", "json_schema": SUPPORT_ANSWER_SCHEMA}

        # gpt-5 family prefers max_completion_tokens; older models use max_tokens.
        # Try the new name first; fall back if SDK or model rejects it.
        try:
            return await self._client.chat.completions.create(
                model=model,
                messages=cast(Any, messages),
                temperature=temperature,
                max_completion_tokens=max_tokens,
                response_format=response_format,
            )
        except TypeError:
            return await self._client.chat.completions.create(
                model=model,
                messages=cast(Any, messages),
                temperature=temperature,
                max_tokens=max_tokens,
                response_format=response_format,
            )

    def _record_usage(self, completion, model_used: str) -> None:
        self.last_used_model = model_used
        try:
            usage = getattr(completion, "usage", None)
            if usage is None:
                return
            self.last_total_tokens = int(getattr(usage, "total_tokens", 0) or 0)
            details = getattr(usage, "prompt_tokens_details", None)
            cached = 0
            if details is not None:
                cached = int(getattr(details, "cached_tokens", 0) or 0)
            self.last_cached_tokens = cached
        except Exception as e:
            logger.debug("Failed to record usage: %s", e)
            return

        # Decrement the persistent daily budget so subsequent requests see the cost
        try:
            try:
                from backend.app.module.rate_limiter import get_limiter
            except ImportError:
                from module.rate_limiter import get_limiter
            asyncio.create_task(get_limiter().record_tokens(self.last_total_tokens))
        except Exception as e:
            logger.debug("Failed to record tokens to limiter: %s", e)

        # Surface prompt-cache ratio for /backend/api/metrics
        try:
            try:
                from backend.app.module.apimonitor import monitor as _mon
            except ImportError:
                from module.apimonitor import monitor as _mon
            prompt_tokens = int(getattr(usage, "prompt_tokens", 0) or 0)
            _mon.record_cache_stats(prompt_tokens, self.last_cached_tokens)
        except Exception as e:
            logger.debug("Failed to record cache stats: %s", e)

    async def ask_agent_with_context(self, query: str, context: str,
                                     conversation_history: Optional[list] = None) -> Tuple[str, Optional[int]]:
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

        primary = cast(str, getattr(self.settings, "azure_deployment_chat", ""))
        fallback = cast(str, getattr(self.settings, "azure_deployment_chat_fallback", "") or "")
        if not primary:
            logger.warning("AIManager: no chat model configured")
            return ("Sorry, the assistant is not configured.", None)

        messages = self._build_messages(query, context, conversation_history)

        # Primary attempt
        try:
            completion = await self._call_azure(primary, messages)
            raw = getattr(completion.choices[0].message, "content", "") or ""
            self._record_usage(completion, primary)
            return self._parse_structured_response(raw)
        except Exception as e:
            err_text = str(e).lower()
            is_content_filter = "content_filter" in err_text or "responsibleaipolicyviolation" in err_text
            if is_content_filter and not fallback:
                # Content filter on primary, no fallback configured — don't retry, surface block
                return "__CONTENT_FILTER_BLOCKED__", None
            logger.warning("Primary model %s failed (%s); attempting fallback", primary, e.__class__.__name__)

        # Fallback attempt
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
            completion = await self._call_azure(fallback, messages)
            raw = getattr(completion.choices[0].message, "content", "") or ""
            self._record_usage(completion, fallback)
            return self._parse_structured_response(raw)
        except Exception as e:
            err_text = str(e).lower()
            if "content_filter" in err_text or "responsibleaipolicyviolation" in err_text:
                return "__CONTENT_FILTER_BLOCKED__", None
            logger.exception("Fallback model %s also failed: %s", fallback, e)
            return ("Sorry, the assistant is currently unavailable.", None)

    def _parse_structured_response(self, raw: str) -> Tuple[str, Optional[int]]:
        """Parse JSON returned by the structured-output completion.

        Returns (markdown_answer, source_indicator) where source_indicator
        matches the legacy 0/1 convention so callers don't need to change.
        Falls back to treating raw as plain text if JSON parsing fails.
        """
        if not raw:
            return raw, None
        try:
            obj = json.loads(raw)
            answer = (obj.get("answer_markdown") or "").strip()
            source = obj.get("source") or "general_knowledge"
            indicator = SOURCE_TO_INDICATOR.get(source, 1)
            return answer, indicator
        except Exception as e:
            logger.warning("Structured output JSON parse failed: %s — returning raw", e)
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
