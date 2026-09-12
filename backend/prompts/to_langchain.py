# pyright: reportUnknownVariableType=false, reportUnknownMemberType=false, reportUnknownArgumentType=false
"""Convert the provider-neutral prompt dicts built by ``prompts/`` into
LangChain messages for the deep agent.

- ``system`` → ``SystemMessage``
- ``user`` → ``HumanMessage`` whose image parts stay in OpenAI/OpenRouter
  ``image_url`` wire format (``ChatOpenRouter`` passes them through) with the
  model's preferred ``detail``; video data URLs become LangChain ``video``
  content blocks, which ``ChatOpenRouter`` serialises as OpenRouter
  ``video_url`` parts.
- ``assistant`` → ``AIMessage``
"""

import base64
from typing import Any, Dict, List

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage

from prompts.prompt_types import PromptMessage


def _detect_mime_type_from_base64(base64_data: str) -> str | None:
    try:
        decoded = base64.b64decode(base64_data[:32])
        if decoded[:8] == b"\x89PNG\r\n\x1a\n":
            return "image/png"
        if decoded[:2] == b"\xff\xd8":
            return "image/jpeg"
        if decoded[:6] in (b"GIF87a", b"GIF89a"):
            return "image/gif"
        if decoded[:4] == b"RIFF" and decoded[8:12] == b"WEBP":
            return "image/webp"
        if decoded[4:8] == b"ftyp":
            return "video/mp4"
        if decoded[:4] == b"\x1aE\xdf\xa3":
            return "video/webm"
    except Exception:
        pass
    return None


def split_data_url(url: str) -> tuple[str, str] | None:
    """``data:<mime>;base64,<payload>`` → ``(mime, payload)``; ``None`` otherwise."""
    if not url.startswith("data:") or "," not in url:
        return None
    header, payload = url.split(",", 1)
    mime = header.removeprefix("data:").split(";", 1)[0].lower()
    if mime == "application/octet-stream":
        mime = _detect_mime_type_from_base64(payload) or mime
    return mime, payload


def is_video_data_url(url: str) -> bool:
    parsed = split_data_url(url)
    return parsed is not None and parsed[0].startswith("video/")


def is_image_data_url(url: str) -> bool:
    parsed = split_data_url(url)
    return parsed is not None and parsed[0].startswith("image/")


def convert_content_part(part: Dict[str, Any], image_detail: str) -> Dict[str, Any] | None:
    part_type = part.get("type")
    if part_type == "text":
        return {"type": "text", "text": str(part.get("text", ""))}
    if part_type != "image_url":
        return None
    image_url = part.get("image_url") or {}
    url = str(image_url.get("url", "")) if isinstance(image_url, dict) else ""
    if not url:
        return None
    parsed = split_data_url(url)
    if parsed is not None and parsed[0].startswith("video/"):
        mime, payload = parsed
        return {"type": "video", "base64": payload, "mime_type": mime}
    return {"type": "image_url", "image_url": {"url": url, "detail": image_detail}}


def convert_prompt_messages(
    prompt_messages: List[PromptMessage], *, image_detail: str = "high"
) -> List[BaseMessage]:
    messages: List[BaseMessage] = []
    for message in prompt_messages:
        role = message.get("role", "user")
        content = message.get("content", "")
        if role == "system":
            text = content if isinstance(content, str) else "".join(
                str(p.get("text", "")) for p in content if isinstance(p, dict) and p.get("type") == "text"
            )
            messages.append(SystemMessage(content=text))
            continue
        if role == "assistant":
            text = content if isinstance(content, str) else "".join(
                str(p.get("text", "")) for p in content if isinstance(p, dict) and p.get("type") == "text"
            )
            messages.append(AIMessage(content=text))
            continue
        if isinstance(content, str):
            messages.append(HumanMessage(content=content))
            continue
        parts: List[Any] = []
        for raw_part in content:
            if not isinstance(raw_part, dict):
                continue
            converted = convert_content_part(raw_part, image_detail)
            if converted is not None:
                parts.append(converted)
        messages.append(HumanMessage(content=parts))
    return messages


def split_system_prompt(messages: List[BaseMessage]) -> tuple[str, List[BaseMessage]]:
    """Deep agents take the system prompt separately from the conversation."""
    if messages and isinstance(messages[0], SystemMessage):
        system = messages[0].content
        return (system if isinstance(system, str) else str(system)), messages[1:]
    return "", messages


def extract_input_images(prompt_messages: List[PromptMessage]) -> List[str]:
    """Still-image data URLs from the prompt (inputs for ``extract_assets``)."""
    images: List[str] = []
    for message in prompt_messages:
        content = message.get("content")
        if not isinstance(content, list):
            continue
        for part in content:
            if not isinstance(part, dict) or part.get("type") != "image_url":
                continue
            image_url = part.get("image_url")
            if not isinstance(image_url, dict):
                continue
            url = image_url.get("url")
            if isinstance(url, str) and url.startswith("data:image/") and "," in url:
                images.append(url)
    return images
