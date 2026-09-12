"""OpenAI-shaped prompt dicts → LangChain messages for the deep agent."""

import base64
from typing import Any, cast

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from prompts.create.video import build_video_prompt_messages
from prompts.to_langchain import (
    convert_prompt_messages,
    extract_input_images,
    is_video_data_url,
    split_system_prompt,
)
from prompts.update.from_history import build_update_prompt_from_history

PNG = "data:image/png;base64," + base64.b64encode(b"\x89PNG\r\n\x1a\n" + b"0" * 16).decode()
WEBM = "data:video/webm;base64," + base64.b64encode(b"\x1aE\xdf\xa3" + b"0" * 16).decode()
OCTET_MP4 = "data:application/octet-stream;base64," + base64.b64encode(b"\x00\x00\x00\x18ftypisom" + b"0" * 8).decode()


def test_image_prompt_keeps_wire_format_and_applies_detail() -> None:
    messages = convert_prompt_messages(
        [
            {"role": "system", "content": "SYS"},
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": PNG, "detail": "high"}},
                    {"type": "text", "text": "Build it"},
                ],
            },
        ],
        image_detail="original",
    )
    system, rest = split_system_prompt(messages)
    assert system == "SYS"
    assert isinstance(rest[0], HumanMessage)
    assert rest[0].content[0] == {"type": "image_url", "image_url": {"url": PNG, "detail": "original"}}
    assert rest[0].content[1] == {"type": "text", "text": "Build it"}


def test_video_prompt_becomes_video_content_block() -> None:
    prompt = build_video_prompt_messages(WEBM, "html_tailwind", "", True)
    messages = convert_prompt_messages(prompt)
    _, rest = split_system_prompt(messages)
    video_part = cast(Any, rest[0].content[0])
    assert video_part["type"] == "video"
    assert video_part["mime_type"] == "video/webm"
    assert video_part["base64"] == WEBM.split(",", 1)[1]
    # a video is not an extractable still image
    assert extract_input_images(prompt) == []


def test_octet_stream_video_is_sniffed() -> None:
    assert is_video_data_url(OCTET_MP4)
    messages = convert_prompt_messages(
        [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": OCTET_MP4}}]}]
    )
    assert cast(Any, messages[0].content[0])["mime_type"] == "video/mp4"


def test_history_roles_map_to_message_classes() -> None:
    prompt = build_update_prompt_from_history(
        stack="html_tailwind",
        history=[
            {"role": "user", "text": "make a page", "images": [PNG], "videos": []},
            {"role": "assistant", "text": "<html><body>v1</body></html>", "images": [], "videos": []},
            {"role": "user", "text": "make it blue", "images": [], "videos": []},
        ],
        image_generation_enabled=True,
    )
    messages = convert_prompt_messages(prompt)
    assert isinstance(messages[0], SystemMessage)
    assert isinstance(messages[1], HumanMessage)
    assert isinstance(messages[2], AIMessage)
    assert '<file path="index.html">' in messages[2].content
    assert isinstance(messages[3], HumanMessage) and messages[3].content == "make it blue"
    assert extract_input_images(prompt) == [PNG]
