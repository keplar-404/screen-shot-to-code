# pyright: reportUnknownVariableType=false, reportUnknownMemberType=false, reportUnknownArgumentType=false
"""Announce tool execution start as a custom stream event.

LangGraph streams the model's finished turn (with its tool calls) *before*
``after_model`` hooks run, so emitting ``toolStart`` from that update would
announce tools that a budget abort then never executes. Emitting from the
tool-call boundary instead gives the original semantics: a tool is announced
exactly when it starts running.
"""

from typing import Any, Awaitable, Callable

from langchain.agents.middleware.types import AgentMiddleware, ToolCallRequest
from langgraph.config import get_stream_writer


def _emit_tool_start(request: ToolCallRequest) -> None:
    tool_call = request.tool_call
    try:
        writer = get_stream_writer()
    except Exception:
        return
    writer(
        {
            "type": "toolStart",
            "tool_call_id": str(tool_call.get("id") or ""),
            "name": str(tool_call.get("name") or "unknown_tool"),
            "args": dict(tool_call.get("args") or {}),
        }
    )


class ToolStartEventsMiddleware(AgentMiddleware):
    def wrap_tool_call(self, request: ToolCallRequest, handler: Callable[[ToolCallRequest], Any]) -> Any:
        _emit_tool_start(request)
        return handler(request)

    async def awrap_tool_call(
        self, request: ToolCallRequest, handler: Callable[[ToolCallRequest], Awaitable[Any]]
    ) -> Any:
        _emit_tool_start(request)
        return await handler(request)
