import asyncio
import dataclasses

import pytest

from backend.app.core.config import ConfigError
from backend.app.services.llm import (
    TRUNCATION_NOTE,
    DemoLLM,
    ImageInput,
    LLMService,
    parse_structured_response,
    recover_truncated_answer,
)
from tests.conftest import fake_gemini

GOOD = '{"source":"documents","language":"en","answer_markdown":"Hello **world**","refusal_reason":null}'


def test_parse_plain_json():
    assert parse_structured_response(GOOD) == ("Hello **world**", 0)


def test_parse_fenced_json():
    assert parse_structured_response(f"```json\n{GOOD}\n```") == ("Hello **world**", 0)


def test_parse_json_surrounded_by_prose():
    assert parse_structured_response(f"Here you go: {GOOD} thanks") == ("Hello **world**", 0)


def test_unknown_source_defaults_to_general_knowledge():
    assert parse_structured_response('{"source":"weird","answer_markdown":"x"}') == ("x", 1)


def test_empty_and_non_object():
    assert parse_structured_response("") == ("", None)
    answer, indicator = parse_structured_response("[1, 2]")
    assert answer == "[1, 2]" and indicator is None


def test_truncated_json_is_recovered_with_note():
    raw = '{"source":"documents","language":"da","answer_markdown":"Line one\\nSecond \\u00e6 \\"quoted\\" part'
    answer, indicator = parse_structured_response(raw)
    assert answer.startswith('Line one\nSecond æ "quoted" part')
    assert answer.endswith(TRUNCATION_NOTE) and indicator == 1


def test_unrecoverable_text_returned_as_is():
    assert recover_truncated_answer("  just text  ") == ("just text", None)
    assert recover_truncated_answer('{"answer_markdown":"') == ('{"answer_markdown":"', None)


def test_recover_handles_odd_escapes():
    answer, _ = recover_truncated_answer('{"answer_markdown":"a\\qb \\u12')
    assert answer.startswith("aqb")


def make(settings, behaviours=None):
    client = fake_gemini(behaviours=behaviours)
    return LLMService(settings, client), client.aio.models


async def test_primary_model_success_reports_usage(prod_settings):
    llm, models = make(prod_settings)
    result = await llm.answer("q", "ctx", [ImageInput("image/png", b"\x89PNG")])
    assert result.answer == "fake answer" and result.indicator == 0 and result.ok
    assert result.usage.model == prod_settings.gemini_model and not result.usage.fallback
    assert (result.usage.prompt_tokens, result.usage.total_tokens, result.usage.cached_tokens) == (100, 120, 10)
    assert models.models_called == [prod_settings.gemini_model]


@pytest.mark.parametrize(("level", "expected"), [("low", "LOW"), ("minimal", "MINIMAL"), ("", None)])
async def test_thinking_level_is_sent_to_gemini(prod_settings, level, expected):
    llm, models = make(dataclasses.replace(prod_settings, gemini_thinking_level=level))
    await llm.answer("q", "ctx")
    thinking = models.configs[0].thinking_config
    assert (thinking.thinking_level.value if thinking else None) == expected


def test_invalid_thinking_level_is_rejected(prod_settings):
    with pytest.raises(ConfigError, match="GEMINI_THINKING_LEVEL"):
        dataclasses.replace(prod_settings, gemini_thinking_level="extreme").validate()


async def test_primary_failure_uses_fallback(prod_settings):
    llm, models = make(prod_settings, {prod_settings.gemini_model: RuntimeError("503")})
    result = await llm.answer("q", "x" * 10000)
    assert result.answer == "fake answer" and result.usage.fallback
    assert models.models_called == [prod_settings.gemini_model, prod_settings.gemini_model_fallback]


async def test_hanging_primary_times_out_to_fallback(prod_settings):
    settings = dataclasses.replace(prod_settings, gemini_timeout=0.05)
    llm, models = make(settings, {settings.gemini_model: "hang"})
    result = await asyncio.wait_for(llm.answer("q", "ctx"), 2)
    assert result.ok and result.answer == "fake answer" and result.usage.fallback
    assert models.models_called == [settings.gemini_model, settings.gemini_model_fallback]


async def test_both_models_failing_is_unavailable(prod_settings):
    boom = RuntimeError("down")
    llm, _ = make(prod_settings, {prod_settings.gemini_model: boom, prod_settings.gemini_model_fallback: boom})
    result = await llm.answer("q", "c")
    assert not result.ok and not result.blocked and "unavailable" in result.answer


async def test_safety_block_is_reported(prod_settings):
    err = RuntimeError("Response was blocked: SAFETY")
    llm, _ = make(prod_settings, {prod_settings.gemini_model: err, prod_settings.gemini_model_fallback: err})
    result = await llm.answer("q", "c")
    assert result.blocked and result.answer == "Content blocked by safety filters."


async def test_no_client_and_no_model(prod_settings):
    assert not (await LLMService(prod_settings, None).answer("q", "c")).ok
    from dataclasses import replace

    llm = LLMService(replace(prod_settings, gemini_model=""), fake_gemini())
    assert "not configured" in (await llm.answer("q", "c")).answer


def test_demo_llm():
    demo = DemoLLM(max_chars=40)
    assert "ingen dokumenter" in demo.generate("  ")
    assert demo.generate("a\n\nb") == "a\n\nb"
    assert demo.generate("x" * 100).endswith("...")
