# pyright: reportUnknownVariableType=false, reportUnknownMemberType=false, reportUnknownArgumentType=false
"""
Core asset extraction engine.

Two-step pipeline:
  1. ``extract_assets_from_images`` — sends image data-URLs + descriptions to
     Gemini (via OpenRouter) which returns bounding boxes, then crops each box
     into a PNG data-URL.
  2. ``build_detector`` — builds the structured-output LangChain runnable used
     in step 1.

This module has NO dependency on FastAPI or any other HTTP framework and can be
used standalone or imported into other projects.
"""

import asyncio
import base64
import io
import json
import math
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Dict, List, Sequence, cast

import pillow_heif
from langchain_core.messages import HumanMessage, SystemMessage
from PIL import Image, ImageOps
from pydantic import BaseModel, Field, SecretStr, ValidationError

from config import (
    ASSET_EXTRACTION_MODEL,
    OPENROUTER_APP_TITLE,
    OPENROUTER_APP_URL,
    OPENROUTER_BASE_URL,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ASSET_EXTRACTION_GEMINI_MODEL = ASSET_EXTRACTION_MODEL
MAX_ASSETS_PER_GEMINI_REQUEST = 25
DEFAULT_ASSET_IMAGE_DETAIL = "original"
SUPPORTED_IMAGE_MIME_TYPES = frozenset(
    {
        "image/png",
        "image/jpeg",
        "image/webp",
        "image/heic",
        "image/heif",
    }
)

# Register HEIC/HEIF support with Pillow.
_register_heif_opener = cast(Callable[[], None], pillow_heif.register_heif_opener)
_register_heif_opener()

ASSET_DETECTION_SYSTEM_INSTRUCTION = (
    "You are a precise 2D object detector for screenshot asset extraction.\n"
    "Return only one schema-valid JSON object. Never return masks, segmentation, "
    "markdown fences, prose, or invented detections. "
    "Process no more than 25 requested objects per response."
)


# ---------------------------------------------------------------------------
# Pydantic schemas
# ---------------------------------------------------------------------------


class AssetDetection(BaseModel):
    """One schema-constrained answer, correlated by a stable request ID."""

    request_id: str = Field(
        description="The unchanged request_id from the corresponding input request."
    )
    image_index: int | None = Field(
        description=(
            "1-based source image number containing the requested asset, or null when "
            "the exact asset is absent, ambiguous, or cannot be isolated."
        )
    )
    box_2d: list[float] | None = Field(
        description=(
            "Tight [ymin, xmin, ymax, xmax] bounds normalized to 0-1000, or null "
            "when image_index is null."
        ),
        min_length=4,
        max_length=4,
    )
    label: str | None = Field(
        description=(
            "A short label that distinguishes this occurrence from lookalikes, or "
            "null when the asset was not found."
        )
    )


class AssetDetectionBatch(BaseModel):
    detections: list[AssetDetection] = Field(
        description="Exactly one detection for every requested request_id.",
        max_length=MAX_ASSETS_PER_GEMINI_REQUEST,
    )


# ---------------------------------------------------------------------------
# Internal data types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SourceImage:
    data_url: str
    image: Image.Image
    mime_type: str
    image_index: int


@dataclass(frozen=True)
class AssetRequest:
    request_id: str
    description: str


def _empty_float_list() -> list[float]:
    return []


def _empty_string_list() -> list[str]:
    return []


@dataclass
class AssetExtractionMetrics:
    """Optional per-call telemetry."""

    request_count: int = 0
    prompt_token_count: int = 0
    candidate_token_count: int = 0
    thoughts_token_count: int = 0
    total_token_count: int = 0
    request_latencies_seconds: list[float] = field(default_factory=_empty_float_list)
    response_ids: list[str] = field(default_factory=_empty_string_list)

    def record_response(self, response: Any, latency_seconds: float) -> None:
        self.request_latencies_seconds.append(latency_seconds)
        response_id = getattr(response, "id", None) or (
            getattr(response, "response_metadata", {}) or {}
        ).get("id")
        if isinstance(response_id, str) and response_id:
            self.response_ids.append(response_id)

        usage = getattr(response, "usage_metadata", None)
        if not usage:
            return
        self.prompt_token_count += int(usage.get("input_tokens") or 0)
        output_tokens = int(usage.get("output_tokens") or 0)
        details = usage.get("output_token_details") or {}
        reasoning = int(details.get("reasoning") or 0)
        self.candidate_token_count += max(output_tokens - reasoning, 0)
        self.thoughts_token_count += reasoning
        self.total_token_count += int(usage.get("total_tokens") or 0)


# ---------------------------------------------------------------------------
# Image helpers
# ---------------------------------------------------------------------------


def _normalize_image_for_detection(image: Image.Image) -> Image.Image:
    """Apply EXIF orientation and convert to stable PNG-compatible mode."""
    oriented = ImageOps.exif_transpose(image)
    has_alpha = "A" in oriented.getbands() or "transparency" in oriented.info
    normalized = oriented.convert("RGBA" if has_alpha else "RGB")
    normalized.load()
    return normalized


def _data_url_to_source_image(
    data_url: str,
    *,
    image_index: int = 1,
) -> SourceImage | None:
    if not data_url.startswith("data:image/") or "," not in data_url:
        return None

    header, encoded = data_url.split(",", 1)
    mime_type = header.removeprefix("data:").split(";", 1)[0].lower()
    if mime_type not in SUPPORTED_IMAGE_MIME_TYPES:
        return None

    try:
        image_bytes = base64.b64decode(encoded, validate=True)
        with Image.open(io.BytesIO(image_bytes)) as opened_image:
            opened_image.seek(0)
            normalized_image = _normalize_image_for_detection(opened_image)
    except Exception:
        return None

    normalized_output = io.BytesIO()
    normalized_image.save(normalized_output, format="PNG")
    normalized_bytes = normalized_output.getvalue()
    normalized_mime_type = "image/png"

    normalized_data_url = (
        f"data:{normalized_mime_type};base64,"
        + base64.b64encode(normalized_bytes).decode("ascii")
    )
    return SourceImage(
        data_url=normalized_data_url,
        image=normalized_image,
        mime_type=normalized_mime_type,
        image_index=image_index,
    )


def _normalize_box(value: object) -> list[float] | None:
    if not isinstance(value, list) or len(value) != 4:
        return None

    coordinates: list[float] = []
    for coordinate in cast(list[object], value):
        if isinstance(coordinate, bool) or not isinstance(coordinate, (int, float)):
            return None
        number = float(coordinate)
        if not math.isfinite(number):
            return None
        coordinates.append(number)

    raw_ymin, raw_xmin, raw_ymax, raw_xmax = coordinates
    ymin, ymax = sorted((raw_ymin, raw_ymax))
    xmin, xmax = sorted((raw_xmin, raw_xmax))
    ymin = max(0.0, min(1000.0, ymin))
    xmin = max(0.0, min(1000.0, xmin))
    ymax = max(0.0, min(1000.0, ymax))
    xmax = max(0.0, min(1000.0, xmax))

    if ymax <= ymin or xmax <= xmin:
        return None
    return [ymin, xmin, ymax, xmax]


def _crop_box_to_data_url(image: Image.Image, box_2d: list[float]) -> str | None:
    normalized_box = _normalize_box(box_2d)
    if normalized_box is None:
        return None

    width, height = image.size
    ymin, xmin, ymax, xmax = normalized_box
    left = max(0, min(width, math.floor(xmin / 1000 * width)))
    top = max(0, min(height, math.floor(ymin / 1000 * height)))
    right = max(0, min(width, math.ceil(xmax / 1000 * width)))
    bottom = max(0, min(height, math.ceil(ymax / 1000 * height)))

    if right <= left or bottom <= top:
        return None

    cropped = image.crop((left, top, right, bottom))
    if cropped.width <= 0 or cropped.height <= 0:
        return None

    output = io.BytesIO()
    cropped.save(output, format="PNG")
    encoded = base64.b64encode(output.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


# ---------------------------------------------------------------------------
# Detector builder
# ---------------------------------------------------------------------------


def build_detector(api_key: str, *, model: str | None = None) -> Any:
    """Build the structured-output LangChain runnable for bounding-box detection."""
    from langchain_openrouter import ChatOpenRouter

    detector = ChatOpenRouter(
        model=model or ASSET_EXTRACTION_GEMINI_MODEL,
        api_key=SecretStr(api_key),
        base_url=OPENROUTER_BASE_URL,
        temperature=0.5,
        reasoning={"effort": "minimal"},
        openrouter_provider={"require_parameters": True},
        app_url=OPENROUTER_APP_URL,
        app_title=OPENROUTER_APP_TITLE,
        timeout=3 * 60 * 1000,
    )
    return detector.with_structured_output(
        AssetDetectionBatch, method="json_schema", strict=True, include_raw=True
    )


# ---------------------------------------------------------------------------
# Detection prompt helpers
# ---------------------------------------------------------------------------


def _stable_request_id(original_index: int) -> str:
    return f"asset-{original_index + 1:04d}"


def _chunk_requests(
    requests: Sequence[AssetRequest],
) -> list[list[AssetRequest]]:
    return [
        list(requests[start : start + MAX_ASSETS_PER_GEMINI_REQUEST])
        for start in range(0, len(requests), MAX_ASSETS_PER_GEMINI_REQUEST)
    ]


def _build_detection_prompt(
    source_images: Sequence[SourceImage],
    requests: Sequence[AssetRequest],
) -> str:
    source_mapping = "\n".join(
        f"- attached image {attachment_index} = source image {source.image_index} "
        f"({source.image.width}x{source.image.height} pixels after EXIF normalization)"
        for attachment_index, source in enumerate(source_images, start=1)
    )
    request_json = json.dumps(
        [
            {
                "request_id": request.request_id,
                "description": request.description,
            }
            for request in requests
        ],
        ensure_ascii=False,
        indent=2,
    )

    return f"""Locate each requested visual asset in the attached source images.

SOURCE IMAGE MAPPING (image_index must use the 1-based source image number):
{source_mapping}

REQUESTS:
{request_json}

BOUNDING-BOX RULES:
- Return exactly one detections record per request, retaining each request_id unchanged.
- box_2d is [ymin, xmin, ymax, xmax], relative to the selected source image, normalized 0-1000.
- Return the smallest axis-aligned box that contains the whole visible requested asset.
- Exclude surrounding UI: cards, containers, page backgrounds, padding, borders, nearby text.
- If absent or ambiguous, return that request_id with image_index=null, box_2d=null, label=null.
- Return JSON only through the supplied schema.
"""


def _parse_detection_response(parsed: Any, raw_text: str | None = None) -> AssetDetectionBatch:
    try:
        if isinstance(parsed, AssetDetectionBatch):
            return parsed
        if parsed is not None:
            return AssetDetectionBatch.model_validate(parsed)
        if raw_text:
            return AssetDetectionBatch.model_validate_json(raw_text)
    except (ValidationError, TypeError, ValueError):
        pass
    return AssetDetectionBatch(detections=[])


# ---------------------------------------------------------------------------
# Core async functions
# ---------------------------------------------------------------------------


async def _locate_asset_batch(
    detector: Any,
    source_images: Sequence[SourceImage],
    requests: Sequence[AssetRequest],
    metrics: AssetExtractionMetrics | None = None,
    *,
    image_detail: str = DEFAULT_ASSET_IMAGE_DETAIL,
) -> AssetDetectionBatch:
    prompt = _build_detection_prompt(source_images, requests)
    if metrics is not None:
        metrics.request_count += 1
    started_at = time.perf_counter()

    user_content: list[Any] = [
        {"type": "image_url", "image_url": {"url": source.data_url, "detail": image_detail}}
        for source in source_images
    ]
    user_content.append({"type": "text", "text": prompt})

    output = await detector.ainvoke(
        [
            SystemMessage(content=ASSET_DETECTION_SYSTEM_INSTRUCTION),
            HumanMessage(content=user_content),
        ]
    )
    raw = output.get("raw") if isinstance(output, dict) else None
    parsed = output.get("parsed") if isinstance(output, dict) else output
    if metrics is not None and raw is not None:
        metrics.record_response(raw, time.perf_counter() - started_at)
    raw_text = raw.text if raw is not None and isinstance(getattr(raw, "text", None), str) else None
    return _parse_detection_response(parsed, raw_text)


async def extract_assets_from_images(
    image_data_urls: List[str],
    asset_descriptions: List[str],
    api_key: str,
    *,
    image_detail: str = DEFAULT_ASSET_IMAGE_DETAIL,
    metrics: AssetExtractionMetrics | None = None,
    detector: Any | None = None,
) -> Dict[str, Any]:
    """
    Locate and crop described assets from input screenshots.

    Args:
        image_data_urls:    List of base64 data-URL strings (PNG / JPEG / WebP / HEIC).
        asset_descriptions: Human-readable descriptions of each asset to find.
        api_key:            OpenRouter API key.
        image_detail:       OpenRouter image detail level ("high" | "original").
        metrics:            Optional telemetry accumulator.
        detector:           Pre-built LangChain runnable (for testing / benchmarks).

    Returns:
        Dict with an ``"assets"`` list. Each asset has:
            - ``description``  – the original description string
            - ``data_url``     – base64 PNG crop, or None if not found
            - ``status``       – "ok" | "missing" | "error"
            - ``box_2d``       – [ymin, xmin, ymax, xmax] (0-1000) or None
            - ``image_index``  – 1-based source image number or None
            - ``label``        – short disambiguating label or None
    """
    source_images: list[SourceImage] = []
    for image_index, data_url in enumerate(image_data_urls, start=1):
        source = _data_url_to_source_image(data_url, image_index=image_index)
        if source is not None:
            source_images.append(source)

    if not source_images:
        return {
            "assets": [],
            "error": "No valid input images were available for asset extraction.",
        }

    requests = [
        AssetRequest(
            request_id=_stable_request_id(original_index),
            description=description,
        )
        for original_index, description in enumerate(asset_descriptions)
    ]
    if not requests:
        return {"assets": []}

    if detector is None:
        detector = build_detector(api_key)

    request_chunks = _chunk_requests(requests)
    batch_results = await asyncio.gather(
        *(
            _locate_asset_batch(detector, source_images, chunk, metrics, image_detail=image_detail)
            for chunk in request_chunks
        )
    )

    detections_by_request_id: dict[str, AssetDetection] = {}
    for request_chunk, batch in zip(request_chunks, batch_results):
        expected_ids = {request.request_id for request in request_chunk}
        for detection in batch.detections:
            if (
                detection.request_id in expected_ids
                and detection.request_id not in detections_by_request_id
            ):
                detections_by_request_id[detection.request_id] = detection

    source_by_image_index = {source.image_index: source for source in source_images}
    assets: List[Dict[str, Any]] = []
    for request in requests:
        detection = detections_by_request_id.get(request.request_id)
        image_index = detection.image_index if detection is not None else None
        if isinstance(image_index, bool):
            image_index = None
        box = _normalize_box(detection.box_2d) if detection is not None else None
        label = detection.label if detection is not None else None

        data_url = None
        status = "missing"
        if isinstance(image_index, int) and box is not None:
            source = source_by_image_index.get(image_index)
            if source is not None:
                data_url = _crop_box_to_data_url(source.image, box)
                status = "ok" if data_url else "error"

        assets.append(
            {
                "description": request.description,
                "data_url": data_url,
                "status": status,
                "box_2d": box,
                "image_index": image_index,
                "label": label,
            }
        )

    result: Dict[str, Any] = {"assets": assets}
    if any(asset["status"] != "ok" for asset in assets):
        result["error"] = (
            "Gemini did not return usable bounding boxes for every requested asset."
        )

    return result
