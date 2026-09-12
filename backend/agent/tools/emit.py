# pyright: reportUnknownVariableType=false, reportUnknownMemberType=false, reportUnknownArgumentType=false
"""Turn a ``ToolExecutionResult`` into the graph update a deep-agent tool returns.

Every tool emits two things:

1. Custom stream events (``runtime.stream_writer``) that the stream bridge
   forwards to the frontend as ``toolResult`` / ``setCode`` messages — the
   same wire protocol as the original backend.
2. A ``Command`` whose ``messages`` update carries the ``ToolMessage`` the
   model sees. Successful results with images attach them as OpenRouter
   ``image_url`` content parts (OpenRouter's tool-message schema accepts a
   list of content parts; ``ChatOpenRouter`` passes ``ToolMessage.content``
   through verbatim, so the parts are written in wire format here).
"""

import base64
import json
from typing import Any, Dict, List, Optional

from langchain.tools import ToolRuntime

from agent.state import S2CContext, S2CState
from langchain_core.messages import HumanMessage, ToolMessage
from langgraph.types import Command

from agent.tools.types import ToolExecutionResult, ToolMultimodalPart

# When True, images are attached to the ToolMessage itself (OpenRouter schema:
# tool content may be a list of parts). When False, they are sent as a
# follow-up user message right after the (text-only) tool result. Flip this if
# an upstream provider rejects image parts inside tool messages.
IMAGES_IN_TOOL_MESSAGE = True


def _image_part(part: ToolMultimodalPart, detail: str = "high") -> Dict[str, Any]:
    if part.image_url:
        url = part.image_url
    else:
        assert part.data is not None
        url = f"data:{part.mime_type};base64,{base64.b64encode(part.data).decode('ascii')}"
    return {"type": "image_url", "image_url": {"url": url, "detail": detail}}


def build_tool_messages(
    tool_call_id: str,
    tool_name: str,
    result: ToolExecutionResult,
    *,
    image_detail: str = "high",
) -> List[Any]:
    """Messages appended to the conversation for one executed tool call."""
    result_json = json.dumps(result.result)
    parts = result.multimodal_parts or []
    status = "success" if result.ok else "error"

    # Failed calls never carry images (Anthropic rejects non-text tool_result
    # content when is_error is set).
    if not parts or not result.ok:
        return [ToolMessage(content=result_json, tool_call_id=tool_call_id, name=tool_name, status=status)]

    image_parts: List[Dict[str, Any]] = []
    for part in parts:
        image_parts.append({"type": "text", "text": part.display_name})
        image_parts.append(_image_part(part, image_detail))

    if IMAGES_IN_TOOL_MESSAGE:
        content: List[Any] = [{"type": "text", "text": result_json}, *image_parts]
        return [ToolMessage(content=content, tool_call_id=tool_call_id, name=tool_name, status=status)]

    return [
        ToolMessage(content=result_json, tool_call_id=tool_call_id, name=tool_name, status=status),
        HumanMessage(
            content=[
                {"type": "text", "text": f"Images returned by the {tool_name} tool are attached."},
                *image_parts,
            ]
        ),
    ]


def emit_result(
    runtime: ToolRuntime[S2CContext, S2CState],
    tool_name: str,
    result: ToolExecutionResult,
    *,
    updated_file: Optional[Dict[str, str]] = None,
) -> Command:
    """Stream UI events for ``result`` and return the graph update."""
    tool_call_id = runtime.tool_call_id or ""

    if updated_file is not None and updated_file.get("content"):
        runtime.stream_writer(
            {
                "type": "setCode",
                "tool_call_id": tool_call_id,
                "content": updated_file["content"],
            }
        )
    runtime.stream_writer(
        {
            "type": "toolResult",
            "tool_call_id": tool_call_id,
            "name": tool_name,
            "ok": result.ok,
            "summary": result.summary,
            "result": result.result,
            "multimodal_parts": [
                {
                    "display_name": p.display_name,
                    "mime_type": p.mime_type,
                    "data_len": len(p.data) if p.data is not None else None,
                    "image_url": p.image_url,
                }
                for p in (result.multimodal_parts or [])
            ],
        }
    )

    update: Dict[str, Any] = {"messages": build_tool_messages(tool_call_id, tool_name, result)}
    if updated_file is not None:
        update["html_file"] = {"path": updated_file["path"], "content": updated_file["content"]}
    return Command(update=update)
