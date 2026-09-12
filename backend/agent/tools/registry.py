# pyright: reportUnknownVariableType=false, reportUnknownMemberType=false, reportUnknownArgumentType=false
"""Which tools a variant gets, mirroring the original provider factory rules."""

from dataclasses import dataclass
from typing import List

from langchain_core.tools import BaseTool

from agent.tools.assets import extract_assets, save_assets
from agent.tools.files import create_file, edit_file
from agent.tools.images import edit_images, generate_images, remove_backgrounds
from agent.tools.options import retrieve_option
from agent.tools.preview import screenshot_preview
from config import REPLICATE_API_KEY
from preview_screenshot import is_screenshot_preview_available


@dataclass(frozen=True)
class ToolCaps:
    image_generation: bool = True
    image_editing: bool = True
    background_removal: bool = True
    asset_extraction: bool = True
    screenshot: bool = True


def resolve_tool_caps(
    *,
    should_generate_images: bool,
    should_extract_assets: bool,
    has_input_images: bool,
    openrouter_api_key: str | None,
    replicate_api_key: str | None,
) -> ToolCaps:
    return ToolCaps(
        image_generation=should_generate_images,
        # Image editing runs through OpenRouter's Images API.
        image_editing=bool(openrouter_api_key),
        # Background removal is the one Replicate-only capability.
        background_removal=bool(replicate_api_key or REPLICATE_API_KEY),
        # Only advertise extraction when the request actually contains a still
        # image the runtime can crop (videos share the message shape but are
        # not valid extractor inputs).
        asset_extraction=should_extract_assets and has_input_images and bool(openrouter_api_key),
        screenshot=is_screenshot_preview_available(),
    )


def build_tools(caps: ToolCaps) -> List[BaseTool]:
    tools: List[BaseTool] = [create_file, edit_file]
    if caps.image_generation:
        tools.append(generate_images)
    # Always offered (parity with the original backend); without a Replicate
    # key the tool returns a clear error the model can recover from.
    tools.append(remove_backgrounds)
    if caps.image_editing:
        tools.append(edit_images)
    if caps.asset_extraction:
        tools.append(extract_assets)
    if caps.screenshot:
        tools.append(screenshot_preview)
    tools.extend([save_assets, retrieve_option])
    return tools
