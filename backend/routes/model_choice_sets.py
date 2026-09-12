"""Which models run for each variant.

Every model is reached through OpenRouter, so a single key unlocks every set.
``S2C_MODEL_SET`` can pin a cheaper/narrower set for local development
(one of: ``default``, ``gemini``, ``anthropic``, ``openai``).
"""

import os
from typing import List, Literal

from custom_types import InputMode
from llm import Llm

# Video variants always use Gemini (the only family with video input).
VIDEO_VARIANT_MODELS = (
    Llm.GEMINI_3_FLASH_PREVIEW_MINIMAL,
    Llm.GEMINI_3_1_PRO_PREVIEW_HIGH,
)

# Image (Create)
#
# Refreshed 2026-07-27 from judged evals on the jun-21 set (see the
# "Jul 24 session" matrix): sol max is the quality anchor (8.2/10 avg),
# opus 5 medium the best non-anchor (7.8), 3.1 pro the third take (7.2),
# and 3-flash high the fast slot (zero failures across the set).
ALL_KEYS_MODELS_DEFAULT = (
    Llm.CLAUDE_OPUS_5_MEDIUM,
    Llm.GEMINI_3_FLASH_PREVIEW_HIGH,
    Llm.GEMINI_3_1_PRO_PREVIEW_HIGH,
    Llm.GPT_5_6_SOL_MAX,
)

# Text (Create)
ALL_KEYS_MODELS_TEXT_CREATE = (
    Llm.GEMINI_3_FLASH_PREVIEW_MINIMAL,
    Llm.GPT_5_6_SOL_HIGH,
    Llm.CLAUDE_OPUS_5_HIGH,
    Llm.GEMINI_3_1_PRO_PREVIEW_LOW,
)

# Image + Text (Update)
ALL_KEYS_MODELS_UPDATE = (
    Llm.GEMINI_3_FLASH_PREVIEW_MINIMAL,
    Llm.GPT_5_6_TERRA_LOW,
)

# Narrower sets, selectable with S2C_MODEL_SET.
GEMINI_ONLY_MODELS = (
    Llm.GEMINI_3_FLASH_PREVIEW_MINIMAL,
    Llm.GEMINI_3_1_PRO_PREVIEW_LOW,
    Llm.GEMINI_3_FLASH_PREVIEW_HIGH,
    Llm.GEMINI_3_1_PRO_PREVIEW_HIGH,
)
ANTHROPIC_ONLY_MODELS = (
    Llm.CLAUDE_OPUS_4_8_MEDIUM,
    Llm.CLAUDE_SONNET_4_6,
)
OPENAI_ONLY_MODELS = (
    Llm.GPT_5_5_HIGH,
    Llm.GPT_5_5_LOW,
)

ModelSetName = Literal["default", "gemini", "anthropic", "openai"]


def get_model_set_name() -> str:
    return os.environ.get("S2C_MODEL_SET", "default").strip().lower() or "default"


def _base_models(
    generation_type: Literal["create", "update"],
    input_mode: InputMode,
) -> tuple[Llm, ...]:
    set_name = get_model_set_name()
    if set_name == "gemini":
        return GEMINI_ONLY_MODELS
    if set_name == "anthropic":
        return ANTHROPIC_ONLY_MODELS
    if set_name == "openai":
        return OPENAI_ONLY_MODELS
    if input_mode == "text" and generation_type == "create":
        return ALL_KEYS_MODELS_TEXT_CREATE
    if generation_type == "update":
        return ALL_KEYS_MODELS_UPDATE
    return ALL_KEYS_MODELS_DEFAULT


def select_variant_models(
    generation_type: Literal["create", "update"],
    input_mode: InputMode,
    num_variants: int,
) -> List[Llm]:
    """Cycle through the configured set: [A, B] with num=5 becomes [A, B, A, B, A]."""
    if input_mode == "video":
        video_models: List[Llm] = list(VIDEO_VARIANT_MODELS)
        return video_models[:num_variants] or video_models

    models: List[Llm] = list(_base_models(generation_type, input_mode))
    return [models[i % len(models)] for i in range(num_variants)]
