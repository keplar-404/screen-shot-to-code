from fastapi import APIRouter
from pydantic import BaseModel

from config import OPENROUTER_API_KEY, REPLICATE_API_KEY, S2C_FAKE_MODEL
from preview_screenshot import probe_screenshot_preview

router = APIRouter()


class Capabilities(BaseModel):
    screenshot_preview: bool
    # Whether the backend has an OpenRouter key configured (the settings dialog
    # can still supply one per request).
    openrouter_key_configured: bool = False
    # remove_backgrounds needs Replicate; everything else runs via OpenRouter.
    background_removal: bool = False
    fake_model: bool = False


@router.get("/api/capabilities", response_model=Capabilities)
async def get_capabilities() -> Capabilities:
    """Backend feature availability for the frontend to reflect in settings."""
    return Capabilities(
        screenshot_preview=await probe_screenshot_preview(),
        openrouter_key_configured=bool(OPENROUTER_API_KEY),
        background_removal=bool(REPLICATE_API_KEY),
        fake_model=S2C_FAKE_MODEL,
    )
