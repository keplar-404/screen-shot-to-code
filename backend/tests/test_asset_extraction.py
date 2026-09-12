"""Asset extraction through an injected structured-output detector."""

import base64
import io
from typing import Any

import pytest
from PIL import Image

from asset_extraction import (
    AssetDetection,
    AssetDetectionBatch,
    _build_detection_prompt,
    _data_url_to_source_image,
    _normalize_box,
    extract_assets_from_images,
)


def _png(width: int = 100, height: int = 50) -> str:
    image = Image.new("RGB", (width, height), "white")
    image.paste(Image.new("RGB", (20, 10), "red"), (10, 5))
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


class FakeDetector:
    """Stands in for ChatOpenRouter(...).with_structured_output(..., include_raw=True)."""

    def __init__(self, detections: list[AssetDetection]) -> None:
        self.detections = detections
        self.requests: list[Any] = []

    async def ainvoke(self, messages: Any) -> dict[str, Any]:
        self.requests.append(messages)
        return {"raw": None, "parsed": AssetDetectionBatch(detections=self.detections)}


@pytest.mark.asyncio
async def test_extracts_crop_from_detection() -> None:
    detector = FakeDetector(
        [AssetDetection(request_id="asset-0001", image_index=1, box_2d=[100, 100, 300, 300], label="logo")]
    )
    result = await extract_assets_from_images([_png()], ["the red logo"], api_key="k", detector=detector)
    [asset] = result["assets"]
    assert asset["status"] == "ok"
    assert asset["data_url"].startswith("data:image/png;base64,")
    crop = Image.open(io.BytesIO(base64.b64decode(asset["data_url"].split(",", 1)[1])))
    assert crop.size == (20, 10)
    assert "error" not in result

    # the detector received the image first, then the prompt
    [system, human] = detector.requests[0]
    assert human.content[0]["type"] == "image_url"
    assert human.content[0]["image_url"]["detail"] == "original"
    assert human.content[-1]["type"] == "text"
    assert "asset-0001" in human.content[-1]["text"]


@pytest.mark.asyncio
async def test_missing_detection_is_reported() -> None:
    detector = FakeDetector([AssetDetection(request_id="asset-0001", image_index=None, box_2d=None, label=None)])
    result = await extract_assets_from_images([_png()], ["a unicorn"], api_key="k", detector=detector)
    assert result["assets"][0]["status"] == "missing"
    assert "error" in result


@pytest.mark.asyncio
async def test_requests_are_chunked_at_25() -> None:
    detector = FakeDetector([])
    await extract_assets_from_images([_png()], [f"asset {i}" for i in range(30)], api_key="k", detector=detector)
    assert len(detector.requests) == 2


def test_normalize_box_clamps_and_orders() -> None:
    assert _normalize_box([900, 50, 100, 1200]) == [100, 50, 900, 1000]
    assert _normalize_box([1, 2, 3]) is None
    assert _normalize_box([10, 10, 10, 20]) is None


def test_prompt_lists_source_mapping() -> None:
    source = _data_url_to_source_image(_png(120, 80), image_index=2)
    assert source is not None and source.image.size == (120, 80)
    from asset_extraction import AssetRequest

    prompt = _build_detection_prompt([source], [AssetRequest("asset-0001", "logo")])
    assert "attached image 1 = source image 2 (120x80" in prompt
