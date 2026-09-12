"""Image generation through OpenRouter's Images API (SDK mocked)."""

import base64
import io
from types import SimpleNamespace

import pytest
from PIL import Image

import image_generation.openrouter_images as images


def _png_b64() -> str:
    buf = io.BytesIO()
    Image.new("RGB", (8, 8), "green").save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


class FakeImages:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    async def generate_async(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(data=[SimpleNamespace(b64_json=_png_b64(), media_type="image/png")])


@pytest.mark.asyncio
async def test_generate_persists_as_local_asset(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    fake = FakeImages()
    monkeypatch.setattr(images, "_client", lambda api_key: SimpleNamespace(images=fake))
    monkeypatch.setattr("uploaded_assets.store.LOCAL_ASSET_DIR", str(tmp_path))

    urls = await images.generate_images_openrouter(
        ["a red cat", "a blue dog"], "key", asset_base_url="http://127.0.0.1:7001"
    )
    assert len(urls) == 2 and all(u and u.startswith("http://127.0.0.1:7001/local-assets/asset_") for u in urls)
    assert fake.calls[0]["model"] == images.OPENROUTER_IMAGE_MODEL
    assert fake.calls[0]["prompt"] == "a red cat"
    assert len(list(tmp_path.iterdir())) == 1  # identical bytes dedupe to one file


@pytest.mark.asyncio
async def test_edit_passes_reference_images(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    fake = FakeImages()
    monkeypatch.setattr(images, "_client", lambda api_key: SimpleNamespace(images=fake))
    monkeypatch.setattr("uploaded_assets.store.LOCAL_ASSET_DIR", str(tmp_path))

    url = await images.edit_image_openrouter(
        prompt="remove the text",
        image_urls=["data:image/png;base64,AAAA", "https://example.com/ref.png"],
        api_key="key",
        asset_base_url="http://127.0.0.1:7001",
        aspect_ratio="16:9",
    )
    assert url.startswith("http://127.0.0.1:7001/local-assets/")
    call = fake.calls[0]
    assert call["input_references"] == [
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}},
        {"type": "image_url", "image_url": {"url": "https://example.com/ref.png"}},
    ]
    assert call["aspect_ratio"] == "16:9"


@pytest.mark.asyncio
async def test_failed_prompt_yields_none(monkeypatch: pytest.MonkeyPatch) -> None:
    class Exploding:
        async def generate_async(self, **kwargs):
            raise RuntimeError("boom")

    monkeypatch.setattr(images, "_client", lambda api_key: SimpleNamespace(images=Exploding()))
    assert await images.generate_images_openrouter(["x"], "key", asset_base_url="http://h") == [None]
