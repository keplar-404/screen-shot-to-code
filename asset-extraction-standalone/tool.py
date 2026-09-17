"""
LangChain tool: ``extract_assets``

Matches the backend design exactly:
  - Pass ONE or MORE images (the sources to search)
  - Pass ONE or MORE labels (the assets to find across those images)

Gemini looks for all labels across all images in a single API call.
Each label can resolve to any of the provided images.

Usage:
    extract_assets(
        images=[{"path": "screenshot.png"}],
        labels=["company logo", "hero background", "nav icon", "CTA button"],
    )

    # Multiple source images, many labels
    extract_assets(
        images=[
            {"path": "home.png"},
            {"url":  "https://example.com/about.png"},
        ],
        labels=["company logo", "hero photo", "team photo"],
    )

Each image can be:
  - ``path``     – local filesystem path (absolute or relative)
  - ``data_url`` – base64 ``"data:image/...;base64,..."`` string
  - ``url``      – publicly reachable http/https URL (fetched automatically)

Returns a JSON-serialisable dict with an ``"assets"`` list. Each entry has:
  - ``label``        – the description you supplied
  - ``data_url``     – base64 PNG crop of the detected asset, or null
  - ``status``       – "ok" | "missing" | "error"
  - ``box_2d``       – [ymin, xmin, ymax, xmax] normalised 0-1000, or null
  - ``image_index``  – which input image (1-based), or null
"""

from __future__ import annotations

import asyncio
import base64
import mimetypes
from pathlib import Path
from typing import Any, Dict, List, Optional

import httpx
from langchain.tools import tool
from pydantic import BaseModel, Field

from config import OPENROUTER_API_KEY
from extraction import extract_assets_from_images


# ---------------------------------------------------------------------------
# Input schema
# ---------------------------------------------------------------------------


class ImageSource(BaseModel):
    """One image source to search. Supply exactly one of: path, data_url, url."""

    path: Optional[str] = Field(
        default=None,
        description="Local filesystem path (absolute or relative to cwd).",
    )
    data_url: Optional[str] = Field(
        default=None,
        description='Base64 data-URL, e.g. "data:image/png;base64,...".',
    )
    url: Optional[str] = Field(
        default=None,
        description="Publicly reachable http/https URL — fetched automatically.",
    )


class ExtractAssetsInput(BaseModel):
    images: List[ImageSource] = Field(
        description=(
            "One or more source images to search. "
            "All labels are searched across all images; Gemini picks the best match."
        ),
        min_length=1,
    )
    labels: List[str] = Field(
        description=(
            "One or more asset descriptions to locate. "
            "Each label identifies one asset occurrence. "
            "Examples: 'company logo top-left', 'hero background photo', 'nav icon set'."
        ),
        min_length=1,
    )
    api_key: Optional[str] = Field(
        default=None,
        description="OpenRouter API key. Falls back to the OPENROUTER_API_KEY env var.",
    )


# ---------------------------------------------------------------------------
# Image loading helpers
# ---------------------------------------------------------------------------


def _path_to_data_url(path: str) -> str:
    p = Path(path).expanduser().resolve()
    if not p.exists():
        raise FileNotFoundError(f"Image file not found: {p}")
    mime, _ = mimetypes.guess_type(str(p))
    mime = mime or "image/png"
    encoded = base64.b64encode(p.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{encoded}"


async def _url_to_data_url(url: str) -> str:
    async with httpx.AsyncClient(follow_redirects=True, timeout=30) as client:
        response = await client.get(url)
        response.raise_for_status()
        content_type = response.headers.get("content-type", "image/png").split(";")[0].strip()
        encoded = base64.b64encode(response.content).decode("ascii")
        return f"data:{content_type};base64,{encoded}"


async def _resolve_image_source(image: ImageSource, index: int) -> str:
    sources_set = sum([
        image.path is not None,
        image.data_url is not None,
        image.url is not None,
    ])
    if sources_set == 0:
        raise ValueError(f"images[{index}] must provide one of: path, data_url, url")
    if sources_set > 1:
        raise ValueError(f"images[{index}] must provide exactly ONE of: path, data_url, url")

    if image.path is not None:
        return _path_to_data_url(image.path)
    if image.url is not None:
        return await _url_to_data_url(image.url)
    assert image.data_url is not None
    return image.data_url


# ---------------------------------------------------------------------------
# The LangChain tool
# ---------------------------------------------------------------------------


@tool("extract_assets", args_schema=ExtractAssetsInput)
async def extract_assets(
    images: List[ImageSource],
    labels: List[str],
    api_key: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Locate and crop visual assets from one or more images.

    Supply the source images and a list of labels describing what to find.
    All labels are searched across all images in a single Gemini call —
    each label resolves to whichever image it appears in.

    Args:
        images: One or more image sources (path / data_url / url).
        labels: One or more asset descriptions, e.g.:
                  ["company logo top-left", "hero background photo", "nav icon set"]
        api_key: Optional OpenRouter key (falls back to env var).

    Returns:
        {
          "ok": bool,
          "assets": [
            {
              "label":       "company logo top-left",
              "data_url":    "data:image/png;base64,...",  # PNG crop or null
              "status":      "ok",                         # ok | missing | error
              "box_2d":      [12.0, 18.0, 80.0, 210.0],   # ymin xmin ymax xmax (0-1000)
              "image_index": 1,                            # which image (1-based)
            },
            ...
          ],
          "error": "..." # only present when ok=False
        }

    Example — one image, many labels:
        extract_assets(
            images=[{"path": "screenshot.png"}],
            labels=["logo", "hero photo", "nav icon", "footer CTA"],
        )

    Example — multiple images, many labels (Gemini finds each label in the right image):
        extract_assets(
            images=[{"path": "home.png"}, {"url": "https://..."}],
            labels=["company logo", "team photo", "product shot"],
        )
    """
    resolved_key = api_key or OPENROUTER_API_KEY
    if not resolved_key:
        return {
            "ok": False,
            "error": "No OpenRouter API key found. Set OPENROUTER_API_KEY or pass api_key.",
            "assets": [],
        }

    # Resolve all image sources concurrently
    try:
        data_urls: List[str] = await asyncio.gather(
            *(_resolve_image_source(img, i) for i, img in enumerate(images))
        )
    except (FileNotFoundError, ValueError, httpx.HTTPError) as exc:
        return {"ok": False, "error": str(exc), "assets": []}

    # Clean labels
    clean_labels = [lbl.strip() for lbl in labels if lbl.strip()]
    if not clean_labels:
        return {"ok": False, "error": "No valid labels provided.", "assets": []}

    try:
        result = await extract_assets_from_images(
            image_data_urls=data_urls,
            asset_descriptions=clean_labels,
            api_key=resolved_key,
        )
    except Exception as exc:
        return {"ok": False, "error": f"Extraction failed: {exc}", "assets": []}

    # Rename "description" → "label" for consistency with input
    for asset in result.get("assets", []):
        if "description" in asset:
            asset["label"] = asset.pop("description")

    result["ok"] = not result.get("error")
    return result
