# pyright: reportUnknownVariableType=false, reportUnknownMemberType=false, reportUnknownArgumentType=false
"""Build the chat model used by the deep agent.

All production models are ``ChatOpenRouter`` instances (Chat Completions via
OpenRouter). ``S2C_FAKE_MODEL=1`` swaps in a scripted local model so the app
can run end-to-end without any API key.
"""

from typing import Any, Optional

from langchain_core.language_models import BaseChatModel
from pydantic import SecretStr

from config import (
    OPENROUTER_APP_TITLE,
    OPENROUTER_APP_URL,
    OPENROUTER_BASE_URL,
    S2C_FAKE_MODEL,
)
from llm import Llm, get_model_spec

# Parity with the original providers: every vendor was called with a 50k
# output-token ceiling.
MAX_OUTPUT_TOKENS = 50000
# Milliseconds; long generations legitimately take minutes.
REQUEST_TIMEOUT_MS = 10 * 60 * 1000


class MissingOpenRouterKeyError(Exception):
    def __init__(self) -> None:
        super().__init__(
            "No OpenRouter API key found. Add OPENROUTER_API_KEY to backend/.env "
            "or enter it in the settings dialog. If you add it to .env, restart the backend."
        )


def build_reasoning(model: Llm) -> Optional[dict[str, Any]]:
    effort = get_model_spec(model).effort
    if effort is None:
        return None
    return {"effort": effort}


def build_provider_preferences(model: Llm) -> dict[str, Any]:
    """OpenRouter ``provider`` routing preferences.

    ``require_parameters`` keeps requests on endpoints that support every
    parameter we send (tools, reasoning, response_format). Azure is ignored
    because OpenRouter's reasoning passthrough is stateless there and breaks
    multi-turn tool loops with OpenAI reasoning models (the same default the
    deepagents OpenRouter profile applies).
    """
    prefs: dict[str, Any] = {"require_parameters": True, "ignore": ["azure"]}
    return prefs


def make_chat_model(
    model: Llm,
    api_key: Optional[str],
    *,
    base_url: Optional[str] = None,
) -> BaseChatModel:
    if S2C_FAKE_MODEL:
        from models.fake import ScriptedCodegenModel

        return ScriptedCodegenModel()

    if not api_key:
        raise MissingOpenRouterKeyError()

    # Imported lazily so unit tests that never touch OpenRouter don't need it.
    from langchain_openrouter import ChatOpenRouter

    spec = get_model_spec(model)
    model_kwargs: dict[str, Any] = {}
    if spec.family == "anthropic":
        # Top-level cache_control = OpenRouter's "automatic" Anthropic prompt
        # caching (caches the last cacheable block), matching the original
        # provider's request.
        model_kwargs["cache_control"] = {"type": "ephemeral"}
    return ChatOpenRouter(
        model=spec.slug,
        api_key=SecretStr(api_key),
        base_url=base_url or OPENROUTER_BASE_URL,
        max_tokens=MAX_OUTPUT_TOKENS,
        reasoning=build_reasoning(model),
        openrouter_provider=build_provider_preferences(model),
        model_kwargs=model_kwargs,
        app_url=OPENROUTER_APP_URL,
        app_title=OPENROUTER_APP_TITLE,
        timeout=REQUEST_TIMEOUT_MS,
        max_retries=2,
    )
