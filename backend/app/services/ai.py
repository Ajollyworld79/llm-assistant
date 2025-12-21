import logging
import re
import asyncio
from typing import Optional, Tuple, Any, cast

try:
    from openai import AsyncAzureOpenAI
except ImportError:
    AsyncAzureOpenAI = None

try:
    from backend.app.config import settings
except ImportError:
    try:
        import config
        settings = config.settings
    except ImportError:
        pass

# Setup logging
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
    """Central AI manager for chat completions and parsing.

    Provides:
    - English system prompt tailored for a public portfolio demo
    - Robust [SOURCE:X] parsing on the first line
    - Graceful handling of content filter blocks
    """

    def __init__(self, settings_obj, embeddings_module, client=None):
        self.settings = settings_obj
        self.embeddings = embeddings_module
        self._client = client

    def initialize_client(self) -> None:
        """Lazily initialize AsyncAzureOpenAI client if available."""
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

    async def ask_agent_with_context(self, query: str, context: str, conversation_history: Optional[list] = None) -> Tuple[str, Optional[int]]:
        """Send query + context + (optional) history to the chat model.

        Returns cleaned_text and source_indicator (0,1 or None). In demo mode, uses DemoLLM.
        """
        if conversation_history is None:
            conversation_history = []

        # Demo mode short-circuit
        if getattr(self.settings, 'demo', False):
            demo = DemoLLM()
            try:
                raw, src = await demo.generate(query, context)
                cleaned, src2 = self._parse_and_clean_source(raw)
                return cleaned, src2
            except Exception as e:
                logger.exception('DemoLLM failed: %s', e)
                return ("(Demo) Sorry, demo generation failed.", None)

        # Ensure client
        if self._client is None:
            self.initialize_client()
        if self._client is None:
            return ("Sorry, the assistant is currently unavailable.", None)

        # Build system prompt and messages
        max_ctx = int(getattr(self.settings, "MAX_CONTEXT_CHARS", 4000))
        ctx = (context[:max_ctx] + "\n\n...[Truncated context]...") if context and len(context) > max_ctx else (context or "")

        system_prompt = self._build_system_prompt(ctx)
        messages = [{"role": "system", "content": system_prompt}]

        # Conversation history
        max_history = int(getattr(self.settings, "MAX_CONVERSATION_HISTORY", 6))
        history_limit = min(len(conversation_history), max_history)
        for msg in (conversation_history or [])[-history_limit:]:
            if isinstance(msg, dict):
                u = msg.get("user", "")
                a = msg.get("assistant", "")
                if u:
                    messages.append({"role": "user", "content": u})
                if a:
                    messages.append({"role": "assistant", "content": a})

        messages.append({"role": "user", "content": query})

        try:
            model = cast(str, getattr(self.settings, "azure_deployment_chat", ""))
            if not model:
                logger.warning("AIManager: no chat model configured; aborting request")
                return ("Sorry, the assistant is not configured.", None)

            temperature = float(getattr(self.settings, "TEMPERATURE", 0.3))
            max_tokens = int(getattr(self.settings, "MAX_TOKENS", 512))

            completion = await self._client.chat.completions.create(
                model=model,
                messages=cast(Any, messages),
                temperature=temperature,
                max_tokens=max_tokens,
            )

            try:
                raw = getattr(completion.choices[0].message, "content", "") or ""
            except Exception:
                raw = str(completion) if completion else ""

            cleaned, source = self._parse_and_clean_source(raw)
            return cleaned, source

        except Exception as e:
            txt = str(e).lower()
            if "content_filter" in txt or "responsibleaipolicyviolation" in txt or (hasattr(e, "status_code") and getattr(e, "status_code", None) == 400):
                return "__CONTENT_FILTER_BLOCKED__", None
            logger.exception("AIManager: ask_agent_with_context failed: %s", e)
            return ("Sorry, something went wrong with the assistant.", None)

    def _parse_and_clean_source(self, response_text: str) -> Tuple[str, Optional[int]]:
        """Parse a leading [SOURCE:X] marker (only when it occurs on the FIRST non-empty line).

        Returns (cleaned_text, source_index) where source_index is 0,1 or None.
        """
        if not response_text:
            return response_text, None

        # Only consider a SOURCE marker on the first non-empty line
        lines = response_text.splitlines()
        for i, line in enumerate(lines):
            if line.strip() == '':
                continue
            m = re.match(r'^\s*\**\s*\[SOURCE:(?P<s>[01])\]\s*\**\s*(?P<rest>.*)', line, flags=re.S)
            if m:
                rest = m.group('rest') or ''
                # Rebuild remaining lines after the first line
                remaining = '\n'.join([rest] + lines[i+1:])
                return remaining.strip(), int(m.group('s'))
            break
        return response_text.strip(), None

    def _build_system_prompt(self, context: str) -> str:
        """English system prompt for the public demo assistant."""
        base = (
            "You are a concise, factual, and safety-aware assistant for a public demo. "
            "Answer clearly in English and use Markdown when helpful. Do not invent links or confidential info.\n\n"
        )
        base += "CONTEXT: " + (context if context else "[None]")
        base += "\n\nIf the answer comes from provided documents, PREFIX the response on the first line with [SOURCE:0]. "
        base += "If it comes from general knowledge, PREFIX the response on the first line with [SOURCE:1].\n"
        return base


class DemoLLM:
    """Deterministic demo-mode LLM emulator.

    Returns short, deterministic answers built from provided context and marks
    a SOURCE (0 or 1) heuristically.
    """

    def __init__(self, max_chars: int = 800):
        self.max_chars = max_chars

    async def generate(self, query: str, context: str) -> tuple[str, Optional[int]]:
        # If no context, return a helpful demo message
        if not context or not context.strip():
            return ("I have no documents to answer from. Try uploading or adding documents.", None)

        # Pick up to 3 blocks separated by paragraphs
        parts = [p.strip() for p in context.split("\n\n") if p.strip()][:3]
        body = "\n\n".join(parts)
        if len(body) > self.max_chars:
            body = body[: self.max_chars - 3].rstrip() + "..."

        # Heuristic source selection
        lower = body.lower()
        source = 1 if "microsoft" in lower or "windows" in lower or "office" in lower else 0

        # Return in the same style as the real LLM (with SOURCE on first line)
        text = f"[SOURCE:{source}]\n{body}"
        return text, source
