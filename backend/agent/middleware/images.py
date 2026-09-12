# pyright: reportUnknownVariableType=false, reportUnknownMemberType=false, reportUnknownArgumentType=false
"""Claude vision limits, enforced on the outgoing request.

OpenRouter forwards image bytes unchanged, so the per-image limits Anthropic
documents (5 MB, 8000 px; 2000 px once a request carries more than 20 images)
still apply and are handled here exactly as the original Anthropic provider
did. Applied only to variants running an Anthropic model.
"""

import base64
from typing import Any, Awaitable, Callable, Dict, List, cast

from langchain.agents.middleware.types import AgentMiddleware, ModelRequest
from langchain_core.messages import BaseMessage

from agent.images import (
    CLAUDE_MANY_IMAGE_MAX_DIMENSION,
    CLAUDE_MANY_IMAGE_THRESHOLD,
    CLAUDE_MAX_IMAGE_DIMENSION,
    process_image_bytes,
)


def _iter_image_parts(messages: List[BaseMessage]) -> List[Dict[str, Any]]:
    parts: List[Dict[str, Any]] = []
    for message in messages:
        content = message.content
        if not isinstance(content, list):
            continue
        for part in content:
            if isinstance(part, dict) and part.get("type") == "image_url":
                parts.append(part)
    return parts


def _resize_data_url(url: str, max_dimension: int) -> str:
    header, encoded = url.split(",", 1)
    media_type = header.removeprefix("data:").split(";", 1)[0]
    try:
        raw = base64.b64decode(encoded)
    except ValueError:
        return url
    new_type, new_b64 = process_image_bytes(raw, media_type, max_dimension=max_dimension)
    return f"data:{new_type};base64,{new_b64}"


def enforce_claude_image_limits(messages: List[BaseMessage]) -> List[BaseMessage]:
    """Return copies of ``messages`` with base64 images resized to Claude's limits."""
    image_parts = _iter_image_parts(messages)
    if not image_parts:
        return messages
    max_dimension = (
        CLAUDE_MANY_IMAGE_MAX_DIMENSION
        if len(image_parts) > CLAUDE_MANY_IMAGE_THRESHOLD
        else CLAUDE_MAX_IMAGE_DIMENSION
    )

    updated: List[BaseMessage] = []
    for message in messages:
        content = message.content
        if not isinstance(content, list):
            updated.append(message)
            continue
        new_content: List[Any] = []
        changed = False
        for part in content:
            if (
                isinstance(part, dict)
                and part.get("type") == "image_url"
                and isinstance(part.get("image_url"), dict)
                and str(part["image_url"].get("url", "")).startswith("data:image/")
            ):
                url = str(part["image_url"]["url"])
                new_url = _resize_data_url(url, max_dimension)
                if new_url != url:
                    part = {**part, "image_url": {**part["image_url"], "url": new_url}}
                    changed = True
            new_content.append(part)
        if changed:
            updated.append(message.model_copy(update={"content": new_content}))
        else:
            updated.append(message)
    return updated


class AnthropicImageLimitsMiddleware(AgentMiddleware):
    def wrap_model_call(self, request: ModelRequest, handler: Callable[[ModelRequest], Any]) -> Any:
        return handler(request.override(messages=cast(Any, enforce_claude_image_limits(list(request.messages)))))

    async def awrap_model_call(
        self, request: ModelRequest, handler: Callable[[ModelRequest], Awaitable[Any]]
    ) -> Any:
        return await handler(
            request.override(messages=cast(Any, enforce_claude_image_limits(list(request.messages))))
        )
