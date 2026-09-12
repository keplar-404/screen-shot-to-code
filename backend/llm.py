"""Model registry.

Every code-generation model is reached through OpenRouter with a single API
key. Each ``Llm`` member keeps the human-readable id used in run logs, eval
folders and the frontend (unchanged from the original project) and maps to an
OpenRouter model slug plus a ``reasoning.effort`` value.

OpenRouter maps ``reasoning.effort`` per vendor (see
https://openrouter.ai/docs/guides/best-practices/reasoning-tokens):

- OpenAI GPT-5 family: ``none | minimal | low | medium | high | xhigh | max``
- Anthropic Claude: ``low | medium | high | xhigh | max``
- Google Gemini 3: ``minimal | low | medium | high`` (maps to ``thinkingLevel``)
"""

from dataclasses import dataclass
from enum import Enum
from typing import Literal, Optional, TypedDict


ModelFamily = Literal["openai", "anthropic", "gemini"]
ReasoningEffort = Literal["none", "minimal", "low", "medium", "high", "xhigh", "max"]
ImageDetail = Literal["high", "original"]


# Human-readable model ids. These strings are stored in run logs and eval
# output folder names, so keep them stable.
class Llm(Enum):
    # GPT
    GPT_5_4_MINI_LOW = "gpt-5.4-mini (low thinking)"
    GPT_5_4_2026_03_05_NONE = "gpt-5.4-2026-03-05 (no thinking)"
    GPT_5_4_2026_03_05_LOW = "gpt-5.4-2026-03-05 (low thinking)"
    GPT_5_4_2026_03_05_MEDIUM = "gpt-5.4-2026-03-05 (medium thinking)"
    GPT_5_4_2026_03_05_HIGH = "gpt-5.4-2026-03-05 (high thinking)"
    GPT_5_4_2026_03_05_XHIGH = "gpt-5.4-2026-03-05 (xhigh thinking)"
    GPT_5_5_NONE = "gpt-5.5 (no thinking)"
    GPT_5_5_LOW = "gpt-5.5 (low thinking)"
    GPT_5_5_MEDIUM = "gpt-5.5 (medium thinking)"
    GPT_5_5_HIGH = "gpt-5.5 (high thinking)"
    GPT_5_5_XHIGH = "gpt-5.5 (xhigh thinking)"
    GPT_5_6_SOL_NONE = "gpt-5.6-sol (no thinking)"
    GPT_5_6_SOL_LOW = "gpt-5.6-sol (low thinking)"
    GPT_5_6_SOL_MEDIUM = "gpt-5.6-sol (medium thinking)"
    GPT_5_6_SOL_HIGH = "gpt-5.6-sol (high thinking)"
    GPT_5_6_SOL_XHIGH = "gpt-5.6-sol (xhigh thinking)"
    GPT_5_6_SOL_MAX = "gpt-5.6-sol (max thinking)"
    GPT_5_6_TERRA_LOW = "gpt-5.6-terra (low thinking)"
    # Claude
    CLAUDE_SONNET_4_6 = "claude-sonnet-4-6"
    CLAUDE_OPUS_5_LOW = "claude-opus-5 (low effort)"
    CLAUDE_OPUS_5_MEDIUM = "claude-opus-5 (medium effort)"
    CLAUDE_OPUS_5_HIGH = "claude-opus-5 (high effort)"
    CLAUDE_OPUS_5_XHIGH = "claude-opus-5 (xhigh effort)"
    CLAUDE_OPUS_5_MAX = "claude-opus-5 (max effort)"
    CLAUDE_OPUS_4_8_LOW = "claude-opus-4-8 (low effort)"
    CLAUDE_OPUS_4_8_MEDIUM = "claude-opus-4-8 (medium effort)"
    CLAUDE_OPUS_4_8_HIGH = "claude-opus-4-8 (high effort)"
    CLAUDE_OPUS_4_8_XHIGH = "claude-opus-4-8 (xhigh effort)"
    CLAUDE_OPUS_4_8_MAX = "claude-opus-4-8 (max effort)"
    CLAUDE_FABLE_5_LOW = "claude-fable-5 (low effort)"
    CLAUDE_FABLE_5_MEDIUM = "claude-fable-5 (medium effort)"
    CLAUDE_FABLE_5_HIGH = "claude-fable-5 (high effort)"
    CLAUDE_FABLE_5_XHIGH = "claude-fable-5 (xhigh effort)"
    CLAUDE_FABLE_5_MAX = "claude-fable-5 (max effort)"
    # Gemini
    GEMINI_3_FLASH_PREVIEW_HIGH = "gemini-3-flash-preview (high thinking)"
    GEMINI_3_FLASH_PREVIEW_MINIMAL = "gemini-3-flash-preview (minimal thinking)"
    GEMINI_3_1_PRO_PREVIEW_HIGH = "gemini-3.1-pro-preview (high thinking)"
    GEMINI_3_1_PRO_PREVIEW_MEDIUM = "gemini-3.1-pro-preview (medium thinking)"
    GEMINI_3_1_PRO_PREVIEW_LOW = "gemini-3.1-pro-preview (low thinking)"
    GEMINI_3_5_FLASH_HIGH = "gemini-3.5-flash (high thinking)"
    GEMINI_3_5_FLASH_MEDIUM = "gemini-3.5-flash (medium thinking)"
    GEMINI_3_5_FLASH_LOW = "gemini-3.5-flash (low thinking)"
    GEMINI_3_5_FLASH_MINIMAL = "gemini-3.5-flash (minimal thinking)"
    GEMINI_3_6_FLASH_HIGH = "gemini-3.6-flash (high thinking)"
    GEMINI_3_6_FLASH_MEDIUM = "gemini-3.6-flash (medium thinking)"
    GEMINI_3_6_FLASH_LOW = "gemini-3.6-flash (low thinking)"
    GEMINI_3_6_FLASH_MINIMAL = "gemini-3.6-flash (minimal thinking)"


class Completion(TypedDict):
    duration: float
    code: str


@dataclass(frozen=True)
class ModelSpec:
    """How one ``Llm`` member is invoked through OpenRouter."""

    family: ModelFamily
    slug: str
    effort: Optional[ReasoningEffort]
    # ``original`` is an OpenRouter extension requesting full-resolution image
    # input (downgraded to ``high`` by providers without that tier). The
    # original project used it for gpt-5.5 / gpt-5.6.
    image_detail: ImageDetail = "high"
    # Whether the model accepts ``video_url`` input through OpenRouter.
    video: bool = False


def _openai(slug: str, effort: ReasoningEffort, *, detail: ImageDetail = "high") -> ModelSpec:
    return ModelSpec(family="openai", slug=slug, effort=effort, image_detail=detail)


def _anthropic(slug: str, effort: ReasoningEffort) -> ModelSpec:
    return ModelSpec(family="anthropic", slug=slug, effort=effort)


def _gemini(slug: str, effort: ReasoningEffort) -> ModelSpec:
    return ModelSpec(family="gemini", slug=slug, effort=effort, video=True)


MODEL_SPECS: dict[Llm, ModelSpec] = {
    # --- OpenAI ---
    Llm.GPT_5_4_MINI_LOW: _openai("openai/gpt-5.4-mini", "low"),
    Llm.GPT_5_4_2026_03_05_NONE: _openai("openai/gpt-5.4-2026-03-05", "none"),
    Llm.GPT_5_4_2026_03_05_LOW: _openai("openai/gpt-5.4-2026-03-05", "low"),
    Llm.GPT_5_4_2026_03_05_MEDIUM: _openai("openai/gpt-5.4-2026-03-05", "medium"),
    Llm.GPT_5_4_2026_03_05_HIGH: _openai("openai/gpt-5.4-2026-03-05", "high"),
    Llm.GPT_5_4_2026_03_05_XHIGH: _openai("openai/gpt-5.4-2026-03-05", "xhigh"),
    Llm.GPT_5_5_NONE: _openai("openai/gpt-5.5", "none", detail="original"),
    Llm.GPT_5_5_LOW: _openai("openai/gpt-5.5", "low", detail="original"),
    Llm.GPT_5_5_MEDIUM: _openai("openai/gpt-5.5", "medium", detail="original"),
    Llm.GPT_5_5_HIGH: _openai("openai/gpt-5.5", "high", detail="original"),
    Llm.GPT_5_5_XHIGH: _openai("openai/gpt-5.5", "xhigh", detail="original"),
    Llm.GPT_5_6_SOL_NONE: _openai("openai/gpt-5.6-sol", "none", detail="original"),
    Llm.GPT_5_6_SOL_LOW: _openai("openai/gpt-5.6-sol", "low", detail="original"),
    Llm.GPT_5_6_SOL_MEDIUM: _openai("openai/gpt-5.6-sol", "medium", detail="original"),
    Llm.GPT_5_6_SOL_HIGH: _openai("openai/gpt-5.6-sol", "high", detail="original"),
    Llm.GPT_5_6_SOL_XHIGH: _openai("openai/gpt-5.6-sol", "xhigh", detail="original"),
    Llm.GPT_5_6_SOL_MAX: _openai("openai/gpt-5.6-sol", "max", detail="original"),
    Llm.GPT_5_6_TERRA_LOW: _openai("openai/gpt-5.6-terra", "low", detail="original"),
    # --- Anthropic ---
    Llm.CLAUDE_SONNET_4_6: _anthropic("anthropic/claude-sonnet-4.6", "high"),
    Llm.CLAUDE_OPUS_5_LOW: _anthropic("anthropic/claude-opus-5", "low"),
    Llm.CLAUDE_OPUS_5_MEDIUM: _anthropic("anthropic/claude-opus-5", "medium"),
    Llm.CLAUDE_OPUS_5_HIGH: _anthropic("anthropic/claude-opus-5", "high"),
    Llm.CLAUDE_OPUS_5_XHIGH: _anthropic("anthropic/claude-opus-5", "xhigh"),
    Llm.CLAUDE_OPUS_5_MAX: _anthropic("anthropic/claude-opus-5", "max"),
    Llm.CLAUDE_OPUS_4_8_LOW: _anthropic("anthropic/claude-opus-4.8", "low"),
    Llm.CLAUDE_OPUS_4_8_MEDIUM: _anthropic("anthropic/claude-opus-4.8", "medium"),
    Llm.CLAUDE_OPUS_4_8_HIGH: _anthropic("anthropic/claude-opus-4.8", "high"),
    Llm.CLAUDE_OPUS_4_8_XHIGH: _anthropic("anthropic/claude-opus-4.8", "xhigh"),
    Llm.CLAUDE_OPUS_4_8_MAX: _anthropic("anthropic/claude-opus-4.8", "max"),
    Llm.CLAUDE_FABLE_5_LOW: _anthropic("anthropic/claude-fable-5", "low"),
    Llm.CLAUDE_FABLE_5_MEDIUM: _anthropic("anthropic/claude-fable-5", "medium"),
    Llm.CLAUDE_FABLE_5_HIGH: _anthropic("anthropic/claude-fable-5", "high"),
    Llm.CLAUDE_FABLE_5_XHIGH: _anthropic("anthropic/claude-fable-5", "xhigh"),
    Llm.CLAUDE_FABLE_5_MAX: _anthropic("anthropic/claude-fable-5", "max"),
    # --- Gemini ---
    Llm.GEMINI_3_FLASH_PREVIEW_HIGH: _gemini("google/gemini-3-flash-preview", "high"),
    Llm.GEMINI_3_FLASH_PREVIEW_MINIMAL: _gemini("google/gemini-3-flash-preview", "minimal"),
    Llm.GEMINI_3_1_PRO_PREVIEW_HIGH: _gemini("google/gemini-3.1-pro-preview", "high"),
    Llm.GEMINI_3_1_PRO_PREVIEW_MEDIUM: _gemini("google/gemini-3.1-pro-preview", "medium"),
    Llm.GEMINI_3_1_PRO_PREVIEW_LOW: _gemini("google/gemini-3.1-pro-preview", "low"),
    Llm.GEMINI_3_5_FLASH_HIGH: _gemini("google/gemini-3.5-flash", "high"),
    Llm.GEMINI_3_5_FLASH_MEDIUM: _gemini("google/gemini-3.5-flash", "medium"),
    Llm.GEMINI_3_5_FLASH_LOW: _gemini("google/gemini-3.5-flash", "low"),
    Llm.GEMINI_3_5_FLASH_MINIMAL: _gemini("google/gemini-3.5-flash", "minimal"),
    Llm.GEMINI_3_6_FLASH_HIGH: _gemini("google/gemini-3.6-flash", "high"),
    Llm.GEMINI_3_6_FLASH_MEDIUM: _gemini("google/gemini-3.6-flash", "medium"),
    Llm.GEMINI_3_6_FLASH_LOW: _gemini("google/gemini-3.6-flash", "low"),
    Llm.GEMINI_3_6_FLASH_MINIMAL: _gemini("google/gemini-3.6-flash", "minimal"),
}

assert set(MODEL_SPECS) == set(Llm), "Every Llm member needs a ModelSpec"

# Provider family for each model (kept for run logs / eval grouping).
MODEL_PROVIDER: dict[Llm, str] = {model: spec.family for model, spec in MODEL_SPECS.items()}

OPENAI_MODELS = {m for m, p in MODEL_PROVIDER.items() if p == "openai"}
ANTHROPIC_MODELS = {m for m, p in MODEL_PROVIDER.items() if p == "anthropic"}
GEMINI_MODELS = {m for m, p in MODEL_PROVIDER.items() if p == "gemini"}


def get_model_spec(model: Llm) -> ModelSpec:
    return MODEL_SPECS[model]


def get_openrouter_slug(model: Llm) -> str:
    """The OpenRouter model slug sent in the ``model`` request field."""
    return MODEL_SPECS[model].slug


def get_reasoning_effort(model: Llm) -> Optional[ReasoningEffort]:
    return MODEL_SPECS[model].effort


def get_image_detail(model: Llm) -> ImageDetail:
    return MODEL_SPECS[model].image_detail


def supports_video(model: Llm) -> bool:
    return MODEL_SPECS[model].video
