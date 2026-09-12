"""Unit tests for the tool implementations and the tool-message builder."""

import json
from typing import Any

import pytest

from agent.state import AgentFileState
from agent.tools.emit import build_tool_messages
from agent.tools.files import create_file_impl, edit_file_impl
from agent.tools.options import retrieve_option_impl
from agent.tools.registry import ToolCaps, build_tools, resolve_tool_caps
from agent.tools.types import ToolExecutionResult, ToolMultimodalPart


def test_create_file_extracts_html_from_fenced_content() -> None:
    fs = AgentFileState()
    result = create_file_impl(fs, {"content": "```html\n<html><body>a</body></html>\n```"})
    assert result.ok
    assert fs.content == "<html><body>a</body></html>"
    assert result.summary["path"] == "index.html"
    assert result.updated_content == fs.content


def test_create_file_requires_content() -> None:
    fs = AgentFileState()
    result = create_file_impl(fs, {"content": ""})
    assert not result.ok and "non-empty" in result.result["error"]


def test_edit_file_batch_and_diff() -> None:
    fs = AgentFileState(content="<html>\n<p>a</p>\n<p>a</p>\n</html>")
    result = edit_file_impl(
        fs,
        {"edits": [{"old_text": "<p>a</p>", "new_text": "<p>b</p>", "count": -1}]},
    )
    assert result.ok
    assert fs.content.count("<p>b</p>") == 2
    assert result.summary["edits"][0]["replaced"] == 2
    assert result.summary["firstChangedLine"] == 1  # hunk header of the unified diff
    assert "-<p>a</p>" in result.summary["diff"]


def test_edit_file_missing_old_text_reports_error() -> None:
    fs = AgentFileState(content="<html></html>")
    result = edit_file_impl(fs, {"old_text": "zzz", "new_text": "y"})
    assert not result.ok
    assert result.result["error"] == "old_text not found"
    assert fs.content == "<html></html>"


def test_retrieve_option_bounds() -> None:
    ok = retrieve_option_impl(["<html>1</html>", ""], {"option_number": 1})
    assert ok.ok and ok.result["code"] == "<html>1</html>"
    empty = retrieve_option_impl(["<html>1</html>", ""], {"option_number": 2})
    assert not empty.ok
    out_of_range = retrieve_option_impl(["<html>1</html>"], {"option_number": 5})
    assert not out_of_range.ok and out_of_range.result["available"] == 1


def test_tool_messages_attach_images_only_on_success() -> None:
    parts = [ToolMultimodalPart(display_name="a.png", mime_type="image/png", data=b"\x89PNG")]
    ok = ToolExecutionResult(ok=True, result={"images": 1}, summary={}, multimodal_parts=parts)
    [message] = build_tool_messages("call_1", "generate_images", ok)
    assert isinstance(message.content, list)
    assert message.content[0] == {"type": "text", "text": json.dumps({"images": 1})}
    image_part = [p for p in message.content if p.get("type") == "image_url"][0]
    assert image_part["image_url"]["url"].startswith("data:image/png;base64,")
    assert message.tool_call_id == "call_1"

    failed = ToolExecutionResult(ok=False, result={"error": "x"}, summary={}, multimodal_parts=parts)
    [message] = build_tool_messages("call_2", "generate_images", failed)
    assert message.content == json.dumps({"error": "x"})
    assert message.status == "error"


def test_tool_messages_follow_up_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    import agent.tools.emit as emit

    monkeypatch.setattr(emit, "IMAGES_IN_TOOL_MESSAGE", False)
    parts = [ToolMultimodalPart(display_name="a.png", mime_type="image/png", image_url="https://x/a.png")]
    ok = ToolExecutionResult(ok=True, result={}, summary={}, multimodal_parts=parts)
    tool_message, human = build_tool_messages("call_1", "screenshot_preview", ok)
    assert isinstance(tool_message.content, str)
    image_parts = [p for p in human.content if p.get("type") == "image_url"]
    assert image_parts[0]["image_url"]["url"] == "https://x/a.png"


def test_tool_caps_follow_original_gating(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("agent.tools.registry.is_screenshot_preview_available", lambda: False)
    caps = resolve_tool_caps(
        should_generate_images=True,
        should_extract_assets=True,
        has_input_images=False,  # e.g. video-only prompt
        openrouter_api_key="k",
        replicate_api_key=None,
    )
    names = [t.name for t in build_tools(caps)]
    assert names == [
        "create_file",
        "edit_file",
        "generate_images",
        "remove_backgrounds",
        "edit_images",
        "save_assets",
        "retrieve_option",
    ]
    assert "extract_assets" not in names and "screenshot_preview" not in names

    everything = [t.name for t in build_tools(ToolCaps())]
    assert everything == [
        "create_file",
        "edit_file",
        "generate_images",
        "remove_backgrounds",
        "edit_images",
        "extract_assets",
        "screenshot_preview",
        "save_assets",
        "retrieve_option",
    ]


def test_tool_schemas_keep_original_argument_names() -> None:
    from langchain_core.utils.function_calling import convert_to_openai_tool

    expected: dict[str, Any] = {
        "create_file": {"path", "content"},
        "edit_file": {"path", "old_text", "new_text", "count", "edits"},
        "generate_images": {"prompts"},
        "remove_backgrounds": {"image_urls"},
        "edit_images": {"edits"},
        "extract_assets": {"asset_descriptions"},
        "screenshot_preview": set(),
        "save_assets": {"asset_ids"},
        "retrieve_option": {"option_number"},
    }
    for tool in build_tools(ToolCaps()):
        schema = convert_to_openai_tool(tool)["function"]
        assert set(schema["parameters"].get("properties", {})) == expected[tool.name], tool.name
        # the injected ToolRuntime must never leak into the model-facing schema
        assert "runtime" not in schema["parameters"].get("properties", {})
