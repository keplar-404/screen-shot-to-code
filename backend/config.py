import os

NUM_VARIANTS = 4
NUM_VARIANTS_VIDEO = 2

# --- LLM access: everything goes through OpenRouter with one key ---
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", None)
OPENROUTER_BASE_URL = os.environ.get("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
# Optional app attribution shown on the OpenRouter dashboard
# (https://openrouter.ai/docs/app-attribution).
OPENROUTER_APP_URL = os.environ.get("OPENROUTER_APP_URL", "https://github.com/abi/screenshot-to-code")
OPENROUTER_APP_TITLE = os.environ.get("OPENROUTER_APP_TITLE", "screenshot-to-code")

# Image generation / editing through OpenRouter's Images API
# (https://openrouter.ai/docs/guides/overview/multimodal/image-generation).
OPENROUTER_IMAGE_MODEL = os.environ.get("OPENROUTER_IMAGE_MODEL", "google/gemini-2.5-flash-image")
OPENROUTER_IMAGE_EDIT_MODEL = os.environ.get("OPENROUTER_IMAGE_EDIT_MODEL", OPENROUTER_IMAGE_MODEL)

# Asset extraction (bounding boxes) — a Gemini model reached through OpenRouter.
ASSET_EXTRACTION_MODEL = os.environ.get("ASSET_EXTRACTION_MODEL", "google/gemini-3.6-flash")
# Asset-description planning model used by the standalone extraction API.
ASSET_DESCRIPTION_MODEL = os.environ.get("ASSET_DESCRIPTION_MODEL", "anthropic/claude-sonnet-5")

# Optional: Replicate powers `remove_backgrounds` (no OpenRouter equivalent).
REPLICATE_API_KEY = os.environ.get("REPLICATE_API_KEY", None)

# Development aid: when truthy the backend uses a scripted fake model instead of
# calling OpenRouter, so the whole app can be exercised without any API key.
S2C_FAKE_MODEL = os.environ.get("S2C_FAKE_MODEL", "").strip().lower() in {"1", "true", "yes", "on"}

# Debugging-related
IS_DEBUG_ENABLED = bool(os.environ.get("IS_DEBUG_ENABLED", False))
DEBUG_DIR = os.environ.get("DEBUG_DIR", "")

# Hard per-generation spend ceiling; a run that would continue past this is
# aborted. Applies per variant/eval run. Cost comes from OpenRouter's usage
# accounting; models without a reported cost are not bounded.
GENERATION_MAX_COST_USD = float(os.environ.get("GENERATION_MAX_COST_USD", "3.0"))

# Maximum model calls (tool turns) per variant.
AGENT_MAX_STEPS = int(os.environ.get("AGENT_MAX_STEPS", "30"))

# When enabled, every LLM request is written to run_logs/prompt_reports as a
# JSON report viewable at /evals/prompt-reports.
PROMPT_REPORTS_ENABLED = os.environ.get(
    "PROMPT_REPORTS_ENABLED", ""
).strip().lower() in {"1", "true", "yes", "on"}
LOCAL_ASSET_DIR = os.environ.get(
    "LOCAL_ASSET_DIR", os.path.join(os.path.dirname(__file__), "local_assets")
)
# Base URL the backend serves /local-assets from. The live (websocket) path
# infers this per-request; the evals path has no request, so it uses this.
LOCAL_ASSET_BASE_URL = os.environ.get("LOCAL_ASSET_BASE_URL", "http://127.0.0.1:7001")

# Set to True when running in production (on the hosted version)
# Used as a feature flag to enable or disable certain features
IS_PROD = os.environ.get("IS_PROD", False)
