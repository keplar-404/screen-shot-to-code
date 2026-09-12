# pyright: reportUnknownVariableType=false, reportUnknownMemberType=false, reportUnknownArgumentType=false
"""Image generation and editing through OpenRouter's Images API.

Docs: https://openrouter.ai/docs/guides/overview/multimodal/image-generation

``POST /api/v1/images`` returns base64 image bytes. Generated images are
persisted as served ``/local-assets/...`` files (content-addressed) so the
URLs the model embeds in the page are stable, unlike the expiring delivery
URLs the old Replicate integration produced.
"""

import asyncio
import base64
import time
from typing import Any, List, Literal, Optional

from config import OPENROUTER_APP_TITLE, OPENROUTER_APP_URL, OPENROUTER_BASE_URL, OPENROUTER_IMAGE_EDIT_MODEL, OPENROUTER_IMAGE_MODEL
from uploaded_assets.store import persist_data_url_as_asset

ImageEditAspectRatio = Literal[
    "match_input_image", "1:1", "16:9", "9:16", "4:3", "3:4", "3:2", "2:3"
]
IMAGE_EDIT_ASPECT_RATIOS: tuple[ImageEditAspectRatio, ...] = (
    "match_input_image",
    "1:1",
    "16:9",
    "9:16",
    "4:3",
    "3:4",
    "3:2",
    "2:3",
)
OPENROUTER_IMAGE_BATCH_SIZE = 20
IMAGE_REQUEST_TIMEOUT_MS = 5 * 60 * 1000


def _client(api_key: str) -> Any:
    # Imported lazily: the SDK is only needed when an image tool actually runs.
    from openrouter import OpenRouter

    return OpenRouter(
        api_key=api_key,
        server_url=OPENROUTER_BASE_URL,
        timeout_ms=IMAGE_REQUEST_TIMEOUT_MS,
        http_referer=OPENROUTER_APP_URL,
        x_open_router_title=OPENROUTER_APP_TITLE,
    )


async def _persist_first_image(response: Any, asset_base_url: str, user_id: Optional[str]) -> str:
    data = getattr(response, "data", None) or []
    if not data:
        raise ValueError("OpenRouter returned no image data.")
    first = data[0]
    b64 = getattr(first, "b64_json", None)
    if not isinstance(b64, str) or not b64:
        raise ValueError("OpenRouter image response is missing b64_json.")
    media_type = getattr(first, "media_type", None) or "image/png"
    # Validate the payload before persisting.
    base64.b64decode(b64, validate=True)
    data_url = f"data:{media_type};base64,{b64}"
    saved = await persist_data_url_as_asset(data_url, asset_base_url, user_id=user_id)
    if saved is None:
        raise ValueError(f"Unsupported generated image type: {media_type}")
    return saved.public_url


async def generate_image_openrouter(
    prompt: str,
    api_key: str,
    *,
    asset_base_url: str,
    user_id: Optional[str] = None,
    model: Optional[str] = None,
) -> str:
    client = _client(api_key)
    response = await client.images.generate_async(
        model=model or OPENROUTER_IMAGE_MODEL,
        prompt=prompt,
        n=1,
        aspect_ratio="1:1",
        output_format="png",
    )
    return await _persist_first_image(response, asset_base_url, user_id)


async def edit_image_openrouter(
    prompt: str,
    image_urls: List[str],
    api_key: str,
    *,
    asset_base_url: str,
    aspect_ratio: ImageEditAspectRatio = "match_input_image",
    user_id: Optional[str] = None,
    model: Optional[str] = None,
) -> str:
    """Image-to-image edit: the first URL is the main image, the rest are references."""
    client = _client(api_key)
    kwargs: dict[str, Any] = {}
    if aspect_ratio != "match_input_image":
        kwargs["aspect_ratio"] = aspect_ratio
    response = await client.images.generate_async(
        model=model or OPENROUTER_IMAGE_EDIT_MODEL,
        prompt=prompt,
        n=1,
        output_format="png",
        input_references=[{"type": "image_url", "image_url": {"url": url}} for url in image_urls],
        **kwargs,
    )
    return await _persist_first_image(response, asset_base_url, user_id)


async def generate_images_openrouter(
    prompts: List[str],
    api_key: str,
    *,
    asset_base_url: str,
    user_id: Optional[str] = None,
) -> List[Optional[str]]:
    """Generate one image per prompt; ``None`` marks a failed prompt."""
    start_time = time.time()
    results: list[str | BaseException] = []
    for i in range(0, len(prompts), OPENROUTER_IMAGE_BATCH_SIZE):
        batch = prompts[i : i + OPENROUTER_IMAGE_BATCH_SIZE]
        tasks = [
            generate_image_openrouter(p, api_key, asset_base_url=asset_base_url, user_id=user_id)
            for p in batch
        ]
        results.extend(await asyncio.gather(*tasks, return_exceptions=True))
    print(f"Image generation time: {time.time() - start_time:.2f} seconds")

    processed: List[Optional[str]] = []
    for result in results:
        if isinstance(result, BaseException):
            print(f"Image generation failed: {result}")
            processed.append(None)
        else:
            processed.append(result)
    return processed
