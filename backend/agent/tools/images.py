# pyright: reportUnknownVariableType=false, reportUnknownMemberType=false, reportUnknownArgumentType=false
"""Image tools: ``generate_images`` (OpenRouter Images API), ``edit_images``
(OpenRouter Images API with reference images) and ``remove_backgrounds``
(Replicate — the one capability OpenRouter does not offer)."""

import asyncio
from typing import Any, Dict, List, Optional, cast

from langchain.tools import ToolRuntime, tool
from langgraph.types import Command
from pydantic import BaseModel, Field

from agent.state import S2CContext, S2CState, ensure_str
from agent.tools.emit import emit_result
from agent.tools.local_assets import guess_image_mime, local_asset_url_to_bytes, local_asset_url_to_data_url
from agent.tools.summaries import summarize_text
from agent.tools.types import ToolExecutionResult, ToolMultimodalPart
from config import REPLICATE_API_KEY
from image_generation.openrouter_images import (
    IMAGE_EDIT_ASPECT_RATIOS,
    ImageEditAspectRatio,
    edit_image_openrouter,
    generate_images_openrouter,
)
from image_generation.replicate import remove_background as remove_background_once

IMAGE_TOOL_BATCH_SIZE = 20


class GenerateImagesArgs(BaseModel):
    prompts: List[str] = Field(description="Prompts, one per image to generate.")


class RemoveBackgroundsArgs(BaseModel):
    image_urls: List[str] = Field(description="URLs of images to remove the background from.")


class ImageEdit(BaseModel):
    prompt: str = Field(
        description=(
            "Clear instruction for this independent edit. Refer to inputs as "
            "image 1, image 2, and so on when multiple images are provided."
        )
    )
    image_urls: List[str] = Field(
        description=(
            "Ordered image URLs for this edit: put the main image first, "
            "followed by any reference images."
        )
    )
    aspect_ratio: Optional[str] = Field(
        default="match_input_image",
        description=(
            "Optional aspect ratio for this edited image. One of: "
            + ", ".join(IMAGE_EDIT_ASPECT_RATIOS)
            + ". Use match_input_image to match its main image."
        ),
    )


class EditImagesArgs(BaseModel):
    edits: List[ImageEdit] = Field(
        min_length=1,
        description=(
            "Independent image edits to run in parallel. Results are returned "
            "in this same order."
        ),
    )


def _local_image_part(url: str, display_name: str) -> ToolMultimodalPart:
    """Local /local-assets URLs are not reachable by cloud models — attach bytes."""
    read = local_asset_url_to_bytes(url)
    if read is not None:
        data, mime = read
        return ToolMultimodalPart(display_name=display_name, mime_type=mime, data=data)
    return ToolMultimodalPart(display_name=display_name, mime_type=guess_image_mime(url), image_url=url)


async def generate_images_impl(ctx: S2CContext, prompts: List[str]) -> ToolExecutionResult:
    if not ctx.should_generate_images:
        return ToolExecutionResult(
            ok=False,
            result={"error": "Image generation is disabled."},
            summary={"error": "Image generation disabled"},
        )
    if not isinstance(prompts, list) or not prompts:
        return ToolExecutionResult(
            ok=False,
            result={"error": "generate_images requires a non-empty prompts list"},
            summary={"error": "Missing prompts"},
        )
    cleaned = [prompt.strip() for prompt in prompts if isinstance(prompt, str)]
    unique_prompts = list(dict.fromkeys([p for p in cleaned if p]))
    if not unique_prompts:
        return ToolExecutionResult(
            ok=False,
            result={"error": "No valid prompts provided"},
            summary={"error": "No valid prompts"},
        )
    if not ctx.openrouter_api_key:
        return ToolExecutionResult(
            ok=False,
            result={"error": "No API key available for image generation."},
            summary={"error": "Missing image generation API key"},
        )

    generated = await generate_images_openrouter(
        unique_prompts, ctx.openrouter_api_key, asset_base_url=ctx.asset_base_url, user_id=ctx.user_id
    )
    merged_results = {prompt: url for prompt, url in zip(unique_prompts, generated)}
    summary_items = [
        {"prompt": prompt, "url": url, "status": "ok" if url else "error"}
        for prompt, url in merged_results.items()
    ]
    multimodal_parts = [
        _local_image_part(url, f"generated_{index}.png")
        for index, url in enumerate(merged_results.values())
        if url
    ]
    return ToolExecutionResult(
        ok=True,
        result={"images": merged_results},
        summary={"images": summary_items},
        multimodal_parts=multimodal_parts,
    )


async def remove_backgrounds_impl(ctx: S2CContext, image_urls: List[str]) -> ToolExecutionResult:
    replicate_api_key = ctx.replicate_api_key or REPLICATE_API_KEY
    if not replicate_api_key:
        return ToolExecutionResult(
            ok=False,
            result={"error": "Background removal requires REPLICATE_API_KEY."},
            summary={"error": "Missing Replicate API key"},
        )
    if not isinstance(image_urls, list) or not image_urls:
        return ToolExecutionResult(
            ok=False,
            result={"error": "remove_backgrounds requires a non-empty image_urls list"},
            summary={"error": "Missing image_urls"},
        )
    cleaned = [url.strip() for url in image_urls if isinstance(url, str)]
    unique_urls = list(dict.fromkeys([u for u in cleaned if u]))
    if not unique_urls:
        return ToolExecutionResult(
            ok=False,
            result={"error": "No valid image URLs provided"},
            summary={"error": "No valid image_urls"},
        )

    raw_results: list[str | BaseException] = []
    for i in range(0, len(unique_urls), IMAGE_TOOL_BATCH_SIZE):
        batch = unique_urls[i : i + IMAGE_TOOL_BATCH_SIZE]
        # Replicate can't fetch localhost; inline local assets as data URLs.
        tasks = [
            remove_background_once(local_asset_url_to_data_url(url), replicate_api_key)
            for url in batch
        ]
        raw_results.extend(await asyncio.gather(*tasks, return_exceptions=True))

    results: List[Dict[str, Any]] = []
    for url, raw in zip(unique_urls, raw_results):
        if isinstance(raw, BaseException):
            print(f"Background removal failed for {url}: {raw}")
            results.append({"image_url": url, "result_url": None, "status": "error"})
        else:
            results.append({"image_url": url, "result_url": raw, "status": "ok"})

    summary_items = [
        {
            "image_url": summarize_text(r["image_url"], 100),
            "result_url": r["result_url"],
            "status": r["status"],
        }
        for r in results
    ]
    multimodal_parts = [
        ToolMultimodalPart(
            display_name=f"no_bg_{index}.png",
            mime_type=guess_image_mime(result["result_url"]),
            image_url=result["result_url"],
        )
        for index, result in enumerate(results)
        if result["status"] == "ok" and result["result_url"]
    ]
    return ToolExecutionResult(
        ok=True,
        result={"images": results},
        summary={"images": summary_items},
        multimodal_parts=multimodal_parts,
    )


async def edit_images_impl(ctx: S2CContext, raw_edits: List[Any]) -> ToolExecutionResult:
    if not ctx.openrouter_api_key:
        return ToolExecutionResult(
            ok=False,
            result={"error": "Image editing requires an OpenRouter API key."},
            summary={"error": "Missing OpenRouter API key"},
        )
    if not isinstance(raw_edits, list) or not raw_edits:
        return ToolExecutionResult(
            ok=False,
            result={"error": "edit_images requires a non-empty edits list"},
            summary={"error": "Missing edits"},
        )

    results: List[Dict[str, Any]] = []
    valid_indexes: List[int] = []
    for index, raw_edit in enumerate(raw_edits):
        if isinstance(raw_edit, ImageEdit):
            raw_edit = raw_edit.model_dump()
        if not isinstance(raw_edit, dict):
            results.append(
                {
                    "prompt": "",
                    "image_urls": [],
                    "result_url": None,
                    "status": "error",
                    "aspect_ratio": "match_input_image",
                    "error": f"Edit {index + 1} must be an object.",
                }
            )
            continue

        edit = cast(Dict[str, Any], raw_edit)
        prompt = ensure_str(edit.get("prompt")).strip()
        raw_image_urls = edit.get("image_urls") or []
        image_urls = (
            [url.strip() for url in cast(List[object], raw_image_urls) if isinstance(url, str) and url.strip()]
            if isinstance(raw_image_urls, list)
            else []
        )
        aspect_ratio_value = ensure_str(edit.get("aspect_ratio") or "match_input_image")
        if aspect_ratio_value not in IMAGE_EDIT_ASPECT_RATIOS:
            aspect_ratio_value = "match_input_image"

        item: Dict[str, Any] = {
            "prompt": prompt,
            "image_urls": image_urls,
            "result_url": None,
            "status": "pending",
            "aspect_ratio": aspect_ratio_value,
        }
        errors: List[str] = []
        if not prompt:
            errors.append("prompt must be non-empty")
        if not image_urls:
            errors.append("image_urls must contain at least one valid URL")
        if errors:
            item["status"] = "error"
            item["error"] = "; ".join(errors)
        else:
            valid_indexes.append(index)
        results.append(item)

    async def execute_single_edit(item: Dict[str, Any]) -> str:
        image_urls = cast(List[str], item["image_urls"])
        return await edit_image_openrouter(
            prompt=ensure_str(item["prompt"]),
            image_urls=[local_asset_url_to_data_url(url) for url in image_urls],
            api_key=ctx.openrouter_api_key or "",
            aspect_ratio=cast(ImageEditAspectRatio, item["aspect_ratio"]),
            asset_base_url=ctx.asset_base_url,
            user_id=ctx.user_id,
        )

    for i in range(0, len(valid_indexes), IMAGE_TOOL_BATCH_SIZE):
        batch_indexes = valid_indexes[i : i + IMAGE_TOOL_BATCH_SIZE]
        raw_results = await asyncio.gather(
            *(execute_single_edit(results[index]) for index in batch_indexes),
            return_exceptions=True,
        )
        for index, raw in zip(batch_indexes, raw_results):
            item = results[index]
            if isinstance(raw, BaseException):
                print(f"Image edit failed for {item['image_urls']}: {raw}")
                item["status"] = "error"
                item["error"] = str(raw)
            else:
                item["result_url"] = raw
                item["status"] = "ok"

    summary_items = [
        {
            "prompt": item["prompt"],
            "image_urls": cast(List[str], item["image_urls"]),
            "result_url": item["result_url"],
            "status": item["status"],
            "aspect_ratio": item["aspect_ratio"],
            **({"error": item["error"]} if item.get("error") else {}),
        }
        for item in results
    ]
    multimodal_parts = [
        _local_image_part(item["result_url"], f"edited_{index}.png")
        for index, item in enumerate(results)
        if item["status"] == "ok" and item["result_url"]
    ]
    return ToolExecutionResult(
        ok=True,
        result={"images": results},
        summary={"images": summary_items},
        multimodal_parts=multimodal_parts,
    )


@tool(
    "generate_images",
    args_schema=GenerateImagesArgs,
    description=(
        "Generate image URLs from prompts using an image generation model. Prompt in "
        "detail, and when prompting for people, include details about their appearance "
        "such as their ethnicity, hair color, features, etc. You can pass multiple prompts at once."
    ),
)
async def generate_images(prompts: List[str], runtime: ToolRuntime[S2CContext, S2CState]) -> Command:
    result = await generate_images_impl(runtime.context, prompts)
    return emit_result(runtime, "generate_images", result)


@tool(
    "remove_backgrounds",
    args_schema=RemoveBackgroundsArgs,
    description=(
        "Remove the backgrounds from one or more images in one batch. Returns "
        "URLs to the processed images with transparent backgrounds in input order."
    ),
)
async def remove_backgrounds(image_urls: List[str], runtime: ToolRuntime[S2CContext, S2CState]) -> Command:
    result = await remove_backgrounds_impl(runtime.context, image_urls)
    return emit_result(runtime, "remove_backgrounds", result)


@tool(
    "edit_images",
    args_schema=EditImagesArgs,
    description=(
        "Edit or upscale one or more images by running independent edits in "
        "one batch. Each edit has its own prompt, ordered main/reference "
        "image URLs, and optional aspect ratio. Results are returned in edit order."
    ),
)
async def edit_images(edits: List[ImageEdit], runtime: ToolRuntime[S2CContext, S2CState]) -> Command:
    result = await edit_images_impl(runtime.context, list(edits))
    return emit_result(runtime, "edit_images", result)
