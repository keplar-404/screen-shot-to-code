"""
Asset extraction — pure function entry point.

Public API
----------
    run_extraction(images, labels, api_key=None)      →  dict   (async)
    run_extraction_sync(images, labels, api_key=None) →  dict  (sync wrapper)

``images``  – list of image source dicts. Each dict has exactly ONE of:
                path      – local file path (str)
                data_url  – base64 data-URL string
                url       – public http/https URL (fetched automatically)

``labels``  – list of asset description strings to find across those images.
              Gemini finds each label in whichever image it appears in.

Returns:
  {
    "ok": bool,
    "assets": [
      {
        "label":       str,
        "data_url":    str | None,   # base64 PNG crop
        "status":      "ok" | "missing" | "error",
        "box_2d":      [ymin, xmin, ymax, xmax] | None,
        "image_index": int | None,   # which image (1-based)
      },
      ...
    ],
    "error": str | None,   # only present when ok=False
  }

Example (async) — one image, multiple labels
---------------------------------------------
    import asyncio
    from agent import run_extraction

    result = asyncio.run(run_extraction(
        images=[{"path": "screenshot.png"}],
        labels=["company logo", "hero background", "nav icon", "CTA button"],
    ))
    for asset in result["assets"]:
        print(asset["label"], asset["status"])

Example (sync) — multiple images, multiple labels
--------------------------------------------------
    from agent import run_extraction_sync

    result = run_extraction_sync(
        images=[{"path": "home.png"}, {"url": "https://…"}],
        labels=["company logo", "team photo", "product shot"],
    )

CLI
---
    # images come first (all paths), then labels (all descriptions)
    poetry run python agent.py --images home.png about.png --labels "logo" "hero photo"
"""

from __future__ import annotations

import asyncio
import sys
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv

load_dotenv()

from deepagents import create_deep_agent
from langchain_openrouter import ChatOpenRouter
from pydantic import SecretStr

from config import (
    OPENROUTER_API_KEY,
    OPENROUTER_APP_TITLE,
    OPENROUTER_APP_URL,
    OPENROUTER_BASE_URL,
    ASSET_EXTRACTION_MODEL,
)
from tool import ImageInput, extract_assets


# ---------------------------------------------------------------------------
# Agent factory (internal)
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """\
You are an asset extraction assistant.

You have one tool: `extract_assets`.

When the user provides images and labels, call `extract_assets` with those inputs
and return the result exactly as the tool returns it.
"""


def _build_agent() -> Any:
    if not OPENROUTER_API_KEY:
        raise RuntimeError(
            "OPENROUTER_API_KEY is not set. "
            "Copy .env.example to .env and fill in your key."
        )
    chat_model = ChatOpenRouter(
        model=ASSET_EXTRACTION_MODEL,
        api_key=SecretStr(OPENROUTER_API_KEY),
        base_url=OPENROUTER_BASE_URL,
        app_url=OPENROUTER_APP_URL,
        app_title=OPENROUTER_APP_TITLE,
    )
    return create_deep_agent(
        model=chat_model,
        tools=[extract_assets],
        system_prompt=_SYSTEM_PROMPT,
        name="asset-extractor",
    )


# ---------------------------------------------------------------------------
# Public async function
# ---------------------------------------------------------------------------


async def run_extraction(
    images: List[Dict[str, str]],
    labels: List[str],
    api_key: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Locate and crop visual assets.

    Parameters
    ----------
    images:
        List of image source dicts. Each dict has exactly ONE of:
          ``path``     – local filesystem path
          ``data_url`` – base64 ``"data:image/...;base64,"`` string
          ``url``      – public http/https URL (fetched automatically)

    labels:
        List of asset descriptions to find across those images.
        Gemini searches all images for all labels in one call.
        Example: ["company logo top-left", "hero background photo", "nav icon"]

    api_key:
        OpenRouter API key. Falls back to ``OPENROUTER_API_KEY`` env var.

    Returns
    -------
    dict with keys:
        ``ok``      – bool
        ``assets``  – list of asset dicts (label, data_url, status, box_2d, image_index)
        ``error``   – str or None
    """
    from tool import ImageSource

    image_sources: List[ImageSource] = [
        ImageSource(
            path=item.get("path"),
            data_url=item.get("data_url"),
            url=item.get("url"),
        )
        for item in images
    ]

    effective_key = api_key or OPENROUTER_API_KEY
    result: Dict[str, Any] = await extract_assets.ainvoke(
        {"images": image_sources, "labels": labels, "api_key": effective_key}
    )
    return result


# ---------------------------------------------------------------------------
# Sync convenience wrapper
# ---------------------------------------------------------------------------


def run_extraction_sync(
    images: List[Dict[str, str]],
    labels: List[str],
    api_key: Optional[str] = None,
) -> Dict[str, Any]:
    """Synchronous wrapper around :func:`run_extraction`."""
    return asyncio.run(run_extraction(images, labels, api_key=api_key))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser(
        description="Extract visual assets from images.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            '  # One image, multiple labels\n'
            '  python agent.py --images shot.png --labels "logo" "hero photo" "nav icon"\n\n'
            '  # Multiple images, multiple labels (Gemini picks the right image per label)\n'
            '  python agent.py --images home.png about.png --labels "logo" "team photo"'
        ),
    )
    parser.add_argument(
        "--images", nargs="+", required=True, metavar="PATH",
        help="One or more local image file paths",
    )
    parser.add_argument(
        "--labels", nargs="+", required=True, metavar="LABEL",
        help='One or more asset description labels, e.g. "company logo" "hero photo"',
    )
    parsed = parser.parse_args()

    image_list = [{"path": p} for p in parsed.images]
    result = asyncio.run(run_extraction(image_list, parsed.labels))
    print(json.dumps(result, indent=2, default=str))
