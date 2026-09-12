"""Frontend protocol produced by the deep-agent runner (parity with the old engine)."""

import base64
import io
from typing import Any

import pytest
from PIL import Image

from llm import Llm
from tests.scripted_model import Recorder, ScriptedModel


def _png_data_url() -> str:
    buf = io.BytesIO()
    Image.new("RGB", (40, 20), "blue").save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def _runner(recorder: Recorder, **kwargs: Any) -> Any:
    from agent.deep_runner import DeepAgentRunner

    return DeepAgentRunner(
        send_message=recorder.send,
        variant_index=2,
        openrouter_api_key="key",
        should_generate_images=False,
        **kwargs,
    )


@pytest.mark.asyncio
async def test_create_flow_streams_expected_events(monkeypatch: pytest.MonkeyPatch) -> None:
    from prompts.create.image import build_image_prompt_messages

    html = "<!DOCTYPE html><html><head><title>t</title></head><body>" + "x" * 900 + "</body></html>"
    model = ScriptedModel(
        turns=[
            {"thinking": "let me look", "tool": "create_file", "id": "call_A", "args": {"path": "index.html", "content": html}},
            {"text": "Built the page."},
        ]
    )
    monkeypatch.setattr("agent.deep_runner.make_chat_model", lambda *a, **k: model)

    ws = Recorder()
    prompt = build_image_prompt_messages([_png_data_url()], "html_tailwind", "", False)
    result = await _runner(ws).run(Llm.CLAUDE_OPUS_5_MEDIUM, prompt)

    assert result == html
    types = [e["type"] for e in ws.events]
    assert types[0] == "thinking"
    assert "toolStart" in types and "toolResult" in types and "assistant" in types
    # toolStart precedes the first setCode which precedes toolResult
    assert types.index("toolStart") < types.index("setCode") < types.index("toolResult")
    # every message carries this variant's index
    assert {e["variantIndex"] for e in ws.events} == {2}

    tool_start = ws.of_type("toolStart")[0]
    assert tool_start["eventId"] == "call_A"
    assert tool_start["data"]["name"] == "create_file"
    assert tool_start["data"]["input"]["path"] == "index.html"

    tool_result = ws.of_type("toolResult")[0]
    assert tool_result["eventId"] == "call_A"
    assert tool_result["data"]["ok"] is True
    assert tool_result["data"]["output"]["contentLength"] == len(html)

    # incremental previews grow monotonically and end with the full file
    set_codes = [e["value"] for e in ws.of_type("setCode")]
    assert set_codes[-1] == html
    assert all(len(a) <= len(b) for a, b in zip(set_codes, set_codes[1:]))

    # the model saw the tools we built, and none of the deep-agent built-ins
    assert "create_file" in model.bound_tools and "edit_file" in model.bound_tools
    assert not {"task", "read_file", "write_file", "ls", "glob", "grep", "execute"} & set(model.bound_tools)
    assert "generate_images" not in model.bound_tools  # disabled for this run
    assert "extract_assets" in model.bound_tools  # still image present + key


@pytest.mark.asyncio
async def test_update_flow_edits_seeded_file(monkeypatch: pytest.MonkeyPatch) -> None:
    from prompts.update.from_file_snapshot import build_update_prompt_from_file_snapshot

    current = "<html><head><title>Old</title></head><body><h1>Hi</h1></body></html>"
    model = ScriptedModel(
        turns=[
            {"tool": "edit_file", "id": "call_E", "args": {"old_text": "<title>Old</title>", "new_text": "<title>New</title>"}},
            {"text": "Renamed the title."},
        ]
    )
    monkeypatch.setattr("agent.deep_runner.make_chat_model", lambda *a, **k: model)

    ws = Recorder()
    prompt = build_update_prompt_from_file_snapshot(
        stack="html_tailwind",
        prompt={"text": "Rename the title", "images": [], "videos": []},
        file_state={"path": "index.html", "content": current},
        image_generation_enabled=False,
    )
    runner = _runner(ws, initial_file_state={"path": "index.html", "content": current})
    result = await runner.run(Llm.GEMINI_3_FLASH_PREVIEW_MINIMAL, prompt)

    assert "<title>New</title>" in result
    tool_result = ws.of_type("toolResult")[0]
    assert tool_result["data"]["name"] == "edit_file"
    assert tool_result["data"]["output"]["edits"][0]["replaced"] == 1
    assert "firstChangedLine" in tool_result["data"]["output"]
    assert ws.of_type("setCode")[-1]["value"] == result
    # video-less, still-image-less prompt: extract_assets must not be offered
    assert "extract_assets" not in model.bound_tools


@pytest.mark.asyncio
async def test_failed_tool_call_reports_error_result(monkeypatch: pytest.MonkeyPatch) -> None:
    model = ScriptedModel(
        turns=[
            {"tool": "edit_file", "id": "call_X", "args": {"old_text": "nope", "new_text": "x"}},
            {"tool": "create_file", "id": "call_Y", "args": {"content": "<html><body>ok</body></html>"}},
            {"text": "done"},
        ]
    )
    monkeypatch.setattr("agent.deep_runner.make_chat_model", lambda *a, **k: model)
    ws = Recorder()
    result = await _runner(ws).run(
        Llm.GPT_5_5_LOW, [{"role": "system", "content": "s"}, {"role": "user", "content": "make a page"}]
    )
    assert result == "<html><body>ok</body></html>"
    results = ws.of_type("toolResult")
    assert results[0]["data"]["ok"] is False
    assert "No file to edit" in results[0]["data"]["output"]["error"]
    assert results[1]["data"]["ok"] is True
    # the model received the error as a tool message and continued
    assert model.calls == 3


@pytest.mark.asyncio
async def test_step_limit_aborts_runaway_loop(monkeypatch: pytest.MonkeyPatch) -> None:
    from agent.middleware import AgentStepLimitError

    model = ScriptedModel(
        turns=[{"tool": "retrieve_option", "id": "call_L", "args": {"option_number": 1}}]
    )
    monkeypatch.setattr("agent.deep_runner.make_chat_model", lambda *a, **k: model)
    ws = Recorder()
    runner = _runner(ws, option_codes=["<html>opt</html>"], max_steps=3)
    with pytest.raises(AgentStepLimitError, match="max tool turns"):
        await runner.run(Llm.GPT_5_5_LOW, [{"role": "system", "content": "s"}, {"role": "user", "content": "x"}])
    assert model.calls == 3


@pytest.mark.asyncio
async def test_final_html_in_assistant_text_is_extracted(monkeypatch: pytest.MonkeyPatch) -> None:
    model = ScriptedModel(turns=[{"text": "Here you go:\n```html\n<html><body>inline</body></html>\n```"}])
    monkeypatch.setattr("agent.deep_runner.make_chat_model", lambda *a, **k: model)
    ws = Recorder()
    result = await _runner(ws).run(
        Llm.GPT_5_5_LOW, [{"role": "system", "content": "s"}, {"role": "user", "content": "x"}]
    )
    assert result == "<html><body>inline</body></html>"
    assert ws.of_type("setCode")[-1]["value"] == result


@pytest.mark.asyncio
async def test_fake_model_mode_runs_without_any_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("models.factory.S2C_FAKE_MODEL", True)
    from agent.deep_runner import DeepAgentRunner

    ws = Recorder()
    runner = DeepAgentRunner(send_message=ws.send, variant_index=0, openrouter_api_key=None)
    result = await runner.run(
        Llm.GEMINI_3_FLASH_PREVIEW_HIGH,
        [{"role": "system", "content": "s"}, {"role": "user", "content": "a landing page"}],
    )
    assert "<html" in result
    assert ws.of_type("toolResult")[0]["data"]["name"] == "create_file"
