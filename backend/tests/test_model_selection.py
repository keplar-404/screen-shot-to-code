from llm import MODEL_SPECS, Llm, get_openrouter_slug, get_reasoning_effort
from routes.model_choice_sets import (
    ALL_KEYS_MODELS_DEFAULT,
    ALL_KEYS_MODELS_UPDATE,
    VIDEO_VARIANT_MODELS,
    select_variant_models,
)


def test_every_model_has_openrouter_slug() -> None:
    for model in Llm:
        spec = MODEL_SPECS[model]
        assert "/" in spec.slug
        assert spec.family in {"openai", "anthropic", "gemini"}


def test_effort_parsing_matches_names() -> None:
    assert get_reasoning_effort(Llm.GPT_5_5_NONE) == "none"
    assert get_reasoning_effort(Llm.CLAUDE_OPUS_5_MAX) == "max"
    assert get_reasoning_effort(Llm.GEMINI_3_FLASH_PREVIEW_MINIMAL) == "minimal"
    assert get_openrouter_slug(Llm.CLAUDE_OPUS_4_8_MEDIUM) == "anthropic/claude-opus-4.8"
    assert get_openrouter_slug(Llm.GEMINI_3_1_PRO_PREVIEW_HIGH) == "google/gemini-3.1-pro-preview"


def test_variant_selection_cycles(monkeypatch) -> None:
    monkeypatch.delenv("S2C_MODEL_SET", raising=False)
    assert select_variant_models("create", "image", 4) == list(ALL_KEYS_MODELS_DEFAULT)
    assert select_variant_models("update", "image", 2) == list(ALL_KEYS_MODELS_UPDATE)
    assert select_variant_models("create", "video", 2) == list(VIDEO_VARIANT_MODELS)
    five = select_variant_models("update", "text", 5)
    assert five == [ALL_KEYS_MODELS_UPDATE[i % 2] for i in range(5)]


def test_model_set_override(monkeypatch) -> None:
    monkeypatch.setenv("S2C_MODEL_SET", "gemini")
    models = select_variant_models("create", "image", 4)
    assert all(MODEL_SPECS[m].family == "gemini" for m in models)
    monkeypatch.setenv("S2C_MODEL_SET", "openai")
    assert all(MODEL_SPECS[m].family == "openai" for m in select_variant_models("create", "text", 2))


def test_factory_builds_chat_openrouter(monkeypatch) -> None:
    monkeypatch.setattr("models.factory.S2C_FAKE_MODEL", False)
    from models.factory import MissingOpenRouterKeyError, make_chat_model

    model = make_chat_model(Llm.CLAUDE_OPUS_5_MEDIUM, "sk-or-test")
    params = model._default_params  # type: ignore[attr-defined]
    assert params["model"] == "anthropic/claude-opus-5"
    assert params["reasoning"] == {"effort": "medium"}
    assert params["max_tokens"] == 50000
    assert params["provider"]["require_parameters"] is True
    assert params["cache_control"] == {"type": "ephemeral"}  # Anthropic prompt caching
    gpt = make_chat_model(Llm.GPT_5_5_HIGH, "sk-or-test")._default_params  # type: ignore[attr-defined]
    assert "cache_control" not in gpt

    try:
        make_chat_model(Llm.GPT_5_5_HIGH, None)
    except MissingOpenRouterKeyError as exc:
        assert "OPENROUTER_API_KEY" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected MissingOpenRouterKeyError")
