# pyright: reportUnknownVariableType=false, reportUnknownMemberType=false, reportUnknownArgumentType=false
"""Translate deep-agent stream events into the frontend WebSocket protocol.

The original engine drove the provider stream itself; here LangGraph drives
the loop and we observe it through ``agent.astream(stream_mode=[...])``:

- ``messages``  → token deltas: ``thinking``, ``assistant`` and, for a
  streaming ``create_file`` call, incremental ``setCode`` built from the
  partial JSON of the ``content`` argument.
- ``updates``   → the finished model turn (``toolStart`` for every tool call,
  cosmetic code preview stream for ``create_file``) and tool-node results
  for calls that errored before emitting their own result.
- ``custom``    → ``toolResult`` / ``setCode`` events written by the tools.
- ``values``    → the final graph state.

Message types, payload shapes and eventId conventions match the original
backend so the existing frontend works unchanged.
"""

import asyncio
import json
import uuid
from typing import Any, Awaitable, Callable, Dict, List, Optional, Set

from langchain_core.messages import AIMessage, ToolMessage

from agent.events import StreamEvent
from agent.state import AgentFileState
from agent.tools.parsing import extract_content_from_args, extract_path_from_args
from agent.tools.summaries import summarize_text, summarize_tool_args
from agent.tools.types import ToolCall, ToolExecutionResult
from codegen.utils import extract_html_content
from fs_logging.agent_runs import AgentRunRecorder

SendMessage = Callable[
    [str, Optional[str], int, Optional[Dict[str, Any]], Optional[str]],
    Awaitable[None],
]


def _text_of(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: List[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and block.get("type") == "text":
                parts.append(str(block.get("text", "")))
        return "".join(parts)
    return ""


def _reasoning_text(message: AIMessage) -> str:
    extras = message.additional_kwargs or {}
    reasoning = extras.get("reasoning_content")
    if isinstance(reasoning, str) and reasoning:
        return reasoning
    details = extras.get("reasoning_details")
    if isinstance(details, list):
        chunks: List[str] = []
        for detail in details:
            if isinstance(detail, dict):
                text = detail.get("text") or detail.get("summary")
                if isinstance(text, str):
                    chunks.append(text)
        return "".join(chunks)
    return ""


class StreamBridge:
    def __init__(
        self,
        *,
        send_message: SendMessage,
        variant_index: int,
        file_state: AgentFileState,
        recorder: Optional[AgentRunRecorder] = None,
    ) -> None:
        self.send_message = send_message
        self.variant_index = variant_index
        self.file_state = file_state
        self.recorder = recorder

        self._assistant_event_id = self._next_event_id("assistant")
        self._thinking_event_id = self._next_event_id("thinking")
        # Streaming tool-call argument buffers for the current turn, keyed by
        # the provider's tool_call index.
        self._arg_buffers: Dict[Any, Dict[str, Any]] = {}
        self._started_tool_ids: Set[str] = set()
        self._streamed_lengths: Dict[str, int] = {}
        self._tool_preview_lengths: Dict[str, int] = {}
        self._pending_tool_calls: Dict[str, ToolCall] = {}
        self._result_emitted: Set[str] = set()
        self.final_state: Dict[str, Any] = {}
        self.last_assistant_text: str = ""

    # ------------------------------------------------------------- helpers

    def _next_event_id(self, prefix: str) -> str:
        return f"{prefix}-{self.variant_index}-{uuid.uuid4().hex[:8]}"

    async def _send(
        self,
        msg_type: str,
        value: Optional[str] = None,
        data: Optional[Dict[str, Any]] = None,
        event_id: Optional[str] = None,
    ) -> None:
        await self.send_message(msg_type, value, self.variant_index, data, event_id)

    def _record(self, event: StreamEvent, event_id: Optional[str]) -> None:
        if self.recorder is not None:
            self.recorder.record_stream_event(event, event_id)

    def _mark_preview_length(self, tool_event_id: str, length: int) -> None:
        current = self._tool_preview_lengths.get(tool_event_id, 0)
        if length > current:
            self._tool_preview_lengths[tool_event_id] = length

    async def _stream_code_preview(self, tool_event_id: str, content: str) -> None:
        """Cosmetic progressive reveal of a fully-received ``create_file`` body."""
        if not content:
            return
        already_sent = self._tool_preview_lengths.get(tool_event_id, 0)
        total_len = len(content)
        if already_sent >= total_len:
            return

        max_chunks = 18
        min_step = 200
        step = max(min_step, total_len // max_chunks)
        start = already_sent if already_sent > 0 else 0

        for end in range(start + step, total_len, step):
            await self._send("setCode", content[:end])
            self._mark_preview_length(tool_event_id, end)
            await asyncio.sleep(0.01)

        await self._send("setCode", content)
        self._mark_preview_length(tool_event_id, total_len)
        if self.recorder is not None:
            self.recorder.record_set_code(total_len, "stream_preview")

    # ------------------------------------------------------- messages mode

    async def _handle_streamed_create_file(self, entry: Dict[str, Any]) -> None:
        tool_event_id = entry.get("id")
        if not tool_event_id:
            return
        content = extract_content_from_args(entry["args"])
        if content is None:
            return

        if tool_event_id not in self._started_tool_ids:
            path = (
                extract_path_from_args(entry["args"])
                or self.file_state.path
                or "index.html"
            )
            await self._send(
                "toolStart",
                data={
                    "name": "create_file",
                    "input": {
                        "path": path,
                        "contentLength": len(content),
                        "preview": summarize_text(content, 200),
                    },
                },
                event_id=tool_event_id,
            )
            self._started_tool_ids.add(tool_event_id)

        last_len = self._streamed_lengths.get(tool_event_id, 0)
        if last_len == 0 and content:
            self._streamed_lengths[tool_event_id] = len(content)
            await self._send("setCode", content)
            self._mark_preview_length(tool_event_id, len(content))
        elif len(content) - last_len >= 40:
            self._streamed_lengths[tool_event_id] = len(content)
            await self._send("setCode", content)
            self._mark_preview_length(tool_event_id, len(content))

    async def on_message_chunk(self, message: Any, metadata: Dict[str, Any]) -> None:
        if not isinstance(message, AIMessage):
            return
        if metadata.get("langgraph_node") not in (None, "model"):
            return

        thinking = _reasoning_text(message)
        if thinking:
            self._record(StreamEvent(type="thinking_delta", text=thinking), self._thinking_event_id)
            await self._send("thinking", thinking, event_id=self._thinking_event_id)

        text = _text_of(message.content)
        if text:
            self._record(StreamEvent(type="assistant_delta", text=text), self._assistant_event_id)
            await self._send("assistant", text, event_id=self._assistant_event_id)

        chunks = getattr(message, "tool_call_chunks", None) or []
        for chunk in chunks:
            index = chunk.get("index")
            key = index if index is not None else chunk.get("id") or len(self._arg_buffers)
            entry = self._arg_buffers.setdefault(key, {"id": None, "name": None, "args": ""})
            if chunk.get("id"):
                entry["id"] = chunk["id"]
            if chunk.get("name"):
                entry["name"] = chunk["name"]
            piece = chunk.get("args")
            if isinstance(piece, str) and piece:
                entry["args"] += piece
            self._record(
                StreamEvent(
                    type="tool_call_delta",
                    tool_call_id=entry["id"],
                    tool_name=entry["name"],
                    tool_arguments=entry["args"],
                ),
                entry["id"],
            )
            if entry["name"] == "create_file":
                await self._handle_streamed_create_file(entry)

    # -------------------------------------------------------- updates mode

    async def on_model_turn(self, message: AIMessage) -> None:
        """A model turn finished: remember its tool calls and start a new turn."""
        text = _text_of(message.content)
        if text:
            self.last_assistant_text = text
        for call in message.tool_calls:
            tool_event_id = str(call.get("id") or self._next_event_id("tool"))
            self._pending_tool_calls[tool_event_id] = ToolCall(
                id=tool_event_id,
                name=str(call.get("name") or "unknown_tool"),
                arguments=dict(call.get("args") or {}),
            )
        # New event ids for the next turn.
        self._assistant_event_id = self._next_event_id("assistant")
        self._thinking_event_id = self._next_event_id("thinking")
        self._arg_buffers = {}

    async def on_tool_start(self, tool_call_id: str, name: str, args: Dict[str, Any]) -> None:
        """A tool is about to execute (custom event from ToolStartEventsMiddleware)."""
        tool_event_id = tool_call_id or self._next_event_id("tool")
        tool_call = ToolCall(id=tool_event_id, name=name, arguments=args)
        self._pending_tool_calls[tool_event_id] = tool_call

        if tool_event_id not in self._started_tool_ids:
            await self._send(
                "toolStart",
                data={"name": name, "input": summarize_tool_args(name, args, self.file_state)},
                event_id=tool_event_id,
            )
            self._started_tool_ids.add(tool_event_id)

        if name == "create_file":
            content = extract_content_from_args(args)
            if content:
                await self._stream_code_preview(tool_event_id, content)

        # Timing starts here, after the cosmetic preview stream, so tool
        # durations measure execution only.
        if self.recorder is not None:
            self.recorder.record_tool_start(tool_event_id, tool_call)

    async def on_tool_node_messages(self, messages: List[Any]) -> None:
        """Results for tool calls that never emitted a custom event (tool crashed
        or its arguments failed validation): surface them as failed results."""
        for message in messages:
            if not isinstance(message, ToolMessage):
                continue
            tool_call_id = str(message.tool_call_id or "")
            if not tool_call_id or tool_call_id in self._result_emitted:
                continue
            pending = self._pending_tool_calls.get(tool_call_id)
            name = message.name or (pending.name if pending else "unknown_tool")
            error_text = _text_of(message.content) or "Tool failed"
            result = ToolExecutionResult(
                ok=False,
                result={"error": error_text},
                summary={"error": summarize_text(error_text, 300)},
            )
            await self._emit_tool_result(tool_call_id, name, result)

    async def on_update(self, update: Dict[str, Any]) -> None:
        for node_name, node_update in update.items():
            if not isinstance(node_update, dict):
                continue
            messages = node_update.get("messages")
            if not isinstance(messages, list):
                continue
            if node_name == "model":
                for message in messages:
                    if isinstance(message, AIMessage):
                        await self.on_model_turn(message)
            elif node_name == "tools":
                await self.on_tool_node_messages(messages)

    # --------------------------------------------------------- custom mode

    async def _emit_tool_result(self, tool_call_id: str, name: str, result: ToolExecutionResult) -> None:
        self._result_emitted.add(tool_call_id)
        if self.recorder is not None:
            tool_call = self._pending_tool_calls.get(tool_call_id) or ToolCall(
                id=tool_call_id, name=name, arguments={}
            )
            self.recorder.record_tool_end(tool_call_id, tool_call, result)
        await self._send(
            "toolResult",
            data={"name": name, "output": result.summary, "ok": result.ok},
            event_id=tool_call_id,
        )

    async def on_custom(self, event: Any) -> None:
        if not isinstance(event, dict):
            return
        kind = event.get("type")
        tool_call_id = str(event.get("tool_call_id") or "")
        if kind == "toolStart":
            await self.on_tool_start(
                tool_call_id,
                str(event.get("name") or "unknown_tool"),
                dict(event.get("args") or {}),
            )
            return
        if kind == "setCode":
            content = event.get("content")
            if isinstance(content, str) and content:
                await self._send("setCode", content)
                if self.recorder is not None:
                    self.recorder.record_set_code(len(content), "tool_result")
            return
        if kind == "toolResult":
            name = str(event.get("name") or "unknown_tool")
            result = ToolExecutionResult(
                ok=bool(event.get("ok")),
                result=dict(event.get("result") or {}),
                summary=dict(event.get("summary") or {}),
            )
            await self._emit_tool_result(tool_call_id, name, result)

    # ------------------------------------------------------------ driver

    async def handle(self, mode: str, chunk: Any) -> None:
        if mode == "messages":
            message, metadata = chunk
            await self.on_message_chunk(message, metadata or {})
        elif mode == "updates":
            if isinstance(chunk, dict):
                await self.on_update(chunk)
        elif mode == "custom":
            await self.on_custom(chunk)
        elif mode == "values":
            if isinstance(chunk, dict):
                self.final_state = chunk

    async def finalize(self) -> str:
        """The HTML to return for this variant (mirrors the original finalize step)."""
        if self.file_state.content:
            return self.file_state.content

        assistant_text = self.last_assistant_text
        messages = self.final_state.get("messages") or []
        for message in reversed(messages):
            if isinstance(message, AIMessage) and _text_of(message.content):
                assistant_text = _text_of(message.content)
                break

        html = extract_html_content(assistant_text) if assistant_text else ""
        if html and html.strip() and "<html" in html.lower():
            self.file_state.content = html
            await self._send("setCode", html)
            if self.recorder is not None:
                self.recorder.record_set_code(len(html), "finalize")
        return self.file_state.content
