"""Standalone asset-extraction endpoint for manual/API testing."""

import base64
import json
from typing import Any, List, cast

from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile
from pydantic import BaseModel, Field, SecretStr

from agent.tools.extract_assets import run_extract_assets
from config import (
    ASSET_DESCRIPTION_MODEL,
    ASSET_EXTRACTION_MODEL,
    OPENROUTER_APP_TITLE,
    OPENROUTER_APP_URL,
    OPENROUTER_API_KEY,
    OPENROUTER_BASE_URL,
)

router = APIRouter()

_MAX_IMAGE_BYTES = 20 * 1024 * 1024
_SUPPORTED_IMAGE_TYPES = {"image/png", "image/jpeg", "image/webp"}


class AssetDescriptionBatch(BaseModel):
    asset_descriptions: list[str] = Field(
        description="Specific visible visual assets worth extracting from the screenshot."
    )


async def _generate_asset_descriptions(image_data_url: str, prompt: str) -> list[str]:
    from langchain_core.messages import HumanMessage, SystemMessage
    from langchain_openrouter import ChatOpenRouter

    model = cast(Any, ChatOpenRouter(
        model=ASSET_DESCRIPTION_MODEL,
        api_key=SecretStr(OPENROUTER_API_KEY or ""),
        base_url=OPENROUTER_BASE_URL,
        reasoning={"effort": "medium"},
        openrouter_provider={"require_parameters": True},
        app_url=OPENROUTER_APP_URL,
        app_title=OPENROUTER_APP_TITLE,
        timeout=3 * 60 * 1000,
    ).with_structured_output(
        AssetDescriptionBatch,
        method="json_schema",
        strict=True,
    ))
    response = cast(AssetDescriptionBatch, await model.ainvoke(
        [
            SystemMessage(
                content=(
                   " Use the user's request as the primary instruction. Extract every visual asset the user explicitly asks to preserve, including icons or CSS-like elements when requested. Only exclude an item when the user does not request it or when it cannot be located in the screenshot."




# alternative prmopt 
# "Identify the visible raster/image assets in the screenshot that "
#                     "should be extracted and reused in a website. Return concise, "
#                     "specific descriptions. Exclude ordinary text, CSS shapes, and "
#                     "simple icons that can be recreated with HTML/CSS."\




                )
            ),
            HumanMessage(
                content=[
                    {"type": "image_url", "image_url": {"url": image_data_url}},
                    {
                        "type": "text",
                        "text": f"User request:\n{prompt}\n\nList the assets needed for this request.",
                    },
                ]
            ),
        ]
    ))
    descriptions = [
        description.strip()
        for description in response.asset_descriptions
        if description.strip()
    ]
    if not descriptions:
        raise HTTPException(status_code=422, detail="Claude generated no asset descriptions")
    return descriptions


def _request_base_url(request: Request) -> str:
    forwarded_host = request.headers.get("x-forwarded-host")
    host = (forwarded_host or request.headers.get("host") or request.url.netloc).split(",", 1)[0].strip()
    forwarded_proto = request.headers.get("x-forwarded-proto")
    scheme = (forwarded_proto or request.url.scheme).split(",", 1)[0].strip()
    return f"{scheme}://{host}".rstrip("/")


def _parse_descriptions(raw: str) -> List[str]:
    try:
        parsed: Any = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise HTTPException(
            status_code=400,
            detail="descriptions must be a JSON array of strings",
        ) from exc

    if not isinstance(parsed, list):
        raise HTTPException(status_code=400, detail="descriptions must be a JSON array of strings")

    parsed_items = cast(list[object], parsed)
    descriptions = [item.strip() for item in parsed_items if isinstance(item, str) and item.strip()]
    if not descriptions:
        raise HTTPException(status_code=400, detail="descriptions must contain at least one non-empty string")
    return descriptions


@router.post("/api/asset-extraction")
async def extract_assets_for_testing(
    request: Request,
    image: UploadFile = File(...),
    prompt: str = Form(...),
) -> dict[str, Any]:
    """Generate asset descriptions with Claude, then extract those assets.

    Postman form fields:
    - ``image``: PNG, JPEG, or WebP file
    - ``prompt``: the user's website request
    """
    if not OPENROUTER_API_KEY:
        raise HTTPException(status_code=503, detail="OPENROUTER_API_KEY is not configured")
    if image.content_type not in _SUPPORTED_IMAGE_TYPES:
        raise HTTPException(status_code=400, detail="image must be PNG, JPEG, or WebP")

    if not prompt.strip():
        raise HTTPException(status_code=400, detail="prompt must not be empty")
    image_bytes = await image.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="image is empty")
    if len(image_bytes) > _MAX_IMAGE_BYTES:
        raise HTTPException(status_code=413, detail="image exceeds the 20 MB limit")

    mime_type = image.content_type or "image/png"
    image_data_url = f"data:{mime_type};base64,{base64.b64encode(image_bytes).decode('ascii')}"
    try:
        description_list = await _generate_asset_descriptions(image_data_url, prompt)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=502,
            detail=f"Asset description generation failed: {exc}",
        ) from exc

    result = await run_extract_assets(
        {"asset_descriptions": description_list},
        api_key=OPENROUTER_API_KEY,
        input_images=[image_data_url],
        asset_base_url=_request_base_url(request),
        user_id=None,
    )

    return {
        "ok": result.ok,
        "description_model": ASSET_DESCRIPTION_MODEL,
        "extraction_model": ASSET_EXTRACTION_MODEL,
        "source": {"filename": image.filename, "content_type": mime_type, "bytes": len(image_bytes)},
        "prompt": prompt,
        "descriptions": description_list,
        "result": result.result,
        "summary": result.summary,
    }
