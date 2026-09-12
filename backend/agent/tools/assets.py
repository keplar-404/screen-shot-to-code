# pyright: reportUnknownVariableType=false, reportUnknownMemberType=false, reportUnknownArgumentType=false
"""Asset tools: ``extract_assets`` (crop real assets out of the screenshot via
a Gemini detector reached through OpenRouter) and ``save_assets`` (promote
uploaded images to permanent URLs)."""

from typing import List

from langchain.tools import ToolRuntime, tool
from langgraph.types import Command
from pydantic import BaseModel, Field

from agent.state import S2CContext, S2CState
from agent.tools.emit import emit_result
from agent.tools.extract_assets import run_extract_assets
from uploaded_assets.tools import run_save_assets


class ExtractAssetsArgs(BaseModel):
    asset_descriptions: List[str] = Field(
        description=(
            "Each item identifies exactly one visual-asset occurrence. Include its "
            "distinctive appearance (colors, shape, content, or visible "
            "wordmark), precise location, nearby UI/context, and the "
            "1-based screenshot number when multiple screenshots are "
            "available. For repeated lookalikes, use a separate item for "
            "each wanted instance and distinguish it (for example, "
            "leftmost vs. rightmost); do not give only a generic category."
        )
    )


class SaveAssetsArgs(BaseModel):
    asset_ids: List[str] = Field(
        description=(
            "Opaque temporary asset IDs for uploaded images that should be "
            "promoted to permanent asset URLs."
        )
    )


@tool(
    "extract_assets",
    args_schema=ExtractAssetsArgs,
    description=(
        "Extract one or more tightly cropped visual assets from the input "
        "screenshots or reference images using Gemini. Describe exactly "
        "one occurrence per list item with distinctive appearance, precise "
        "location, nearby context, and the 1-based screenshot number; "
        "distinguish repeated lookalikes instead of naming a generic asset. "
        "Returns each asset in request order with a permanent, embeddable "
        "public_url and an attached crop preview; genuinely absent or "
        "unisolatable items are unresolved. These assets are already saved "
        "— do NOT call save_assets on them (save_assets is only for "
        "user-uploaded images)."
    ),
)
async def extract_assets(
    asset_descriptions: List[str], runtime: ToolRuntime[S2CContext, S2CState]
) -> Command:
    ctx = runtime.context
    result = await run_extract_assets(
        {"asset_descriptions": asset_descriptions},
        api_key=ctx.openrouter_api_key,
        input_images=ctx.input_images,
        asset_base_url=ctx.asset_base_url,
        user_id=ctx.user_id,
    )
    return emit_result(runtime, "extract_assets", result)


@tool(
    "save_assets",
    args_schema=SaveAssetsArgs,
    description=(
        "Promote one or more uploaded temporary image asset IDs to permanent URLs. "
        "Use this before embedding any uploaded images in code. "
        "Returns permanent public_url values to use in the generated code."
    ),
)
async def save_assets(asset_ids: List[str], runtime: ToolRuntime[S2CContext, S2CState]) -> Command:
    result = await run_save_assets({"asset_ids": asset_ids}, user_id=runtime.context.user_id)
    return emit_result(runtime, "save_assets", result)
