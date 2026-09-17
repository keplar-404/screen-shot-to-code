import os

# --- Required ---
OPENROUTER_API_KEY: str | None = os.environ.get("OPENROUTER_API_KEY")

# --- OpenRouter gateway ---
OPENROUTER_BASE_URL: str = os.environ.get(
    "OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"
)
OPENROUTER_APP_URL: str = os.environ.get(
    "OPENROUTER_APP_URL", "https://github.com/abi/screenshot-to-code"
)
OPENROUTER_APP_TITLE: str = os.environ.get(
    "OPENROUTER_APP_TITLE", "asset-extraction-standalone"
)

# --- Detection model (bounding boxes + crop, reached via OpenRouter) ---
ASSET_EXTRACTION_MODEL: str = os.environ.get(
    "ASSET_EXTRACTION_MODEL", "google/gemini-3.6-flash"
)
