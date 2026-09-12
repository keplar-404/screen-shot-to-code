"""A scripted chat model for exercising the deep-agent runner in tests.

Each *turn* is either ``{"text": "..."}`` or ``{"tool": name, "args": {...},
"id": "call_x"}`` (plus optional ``"cost"`` in USD and ``"thinking"``). Tool
turns are streamed as tool-call argument chunks, the way OpenRouter streams
them, so the bridge's incremental ``setCode`` path is exercised too.
"""

import json
from typing import Any, AsyncIterator, Iterator, List, Optional

from langchain_core.callbacks import AsyncCallbackManagerForLLMRun, CallbackManagerForLLMRun
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, AIMessageChunk, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult


class ScriptedModel(BaseChatModel):
    turns: List[dict]
    calls: int = 0
    seen_requests: List[List[BaseMessage]] = []
    bound_tools: List[Any] = []
    chunk_size: int = 16

    @property
    def _llm_type(self) -> str:
        return "scripted-test-model"

    def bind_tools(self, tools: Any, **kwargs: Any) -> "ScriptedModel":  # type: ignore[override]
        self.bound_tools = [
            getattr(t, "name", None) or (t.get("name") if isinstance(t, dict) else None) for t in tools
        ]
        return self

    def _get_ls_params(self, stop: Optional[list[str]] = None, **kwargs: Any) -> Any:
        params = super()._get_ls_params(stop=stop, **kwargs)
        params["ls_provider"] = "openrouter"
        return params

    def _next_turn(self, messages: List[BaseMessage]) -> dict:
        self.seen_requests.append(list(messages))
        turn = self.turns[min(self.calls, len(self.turns) - 1)]
        self.calls += 1
        return turn

    def _chunks(self, turn: dict) -> List[AIMessageChunk]:
        chunks: List[AIMessageChunk] = []
        meta = {"cost": turn["cost"]} if "cost" in turn else {}
        usage = {"input_tokens": 100, "output_tokens": 50, "total_tokens": 150}
        if turn.get("thinking"):
            chunks.append(AIMessageChunk(content="", additional_kwargs={"reasoning_content": turn["thinking"]}))
        if "tool" in turn:
            args = json.dumps(turn.get("args", {}))
            first = True
            for start in range(0, len(args), self.chunk_size):
                chunks.append(
                    AIMessageChunk(
                        content="",
                        tool_call_chunks=[
                            {
                                "name": turn["tool"] if first else None,
                                "args": args[start : start + self.chunk_size],
                                "id": turn.get("id", "call_1") if first else None,
                                "index": 0,
                            }
                        ],
                    )
                )
                first = False
        else:
            text = turn.get("text", "")
            if text:
                for start in range(0, len(text), self.chunk_size):
                    chunks.append(AIMessageChunk(content=text[start : start + self.chunk_size]))
        if not chunks:
            chunks.append(AIMessageChunk(content=""))
        chunks[-1] = AIMessageChunk(
            content=chunks[-1].content,
            tool_call_chunks=chunks[-1].tool_call_chunks,
            additional_kwargs=chunks[-1].additional_kwargs,
            response_metadata=meta,
            usage_metadata=usage,  # type: ignore[arg-type]
        )
        return chunks

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        turn = self._next_turn(messages)
        merged: Optional[AIMessageChunk] = None
        for chunk in self._chunks(turn):
            merged = chunk if merged is None else merged + chunk
        assert merged is not None
        message = AIMessage(
            content=merged.content,
            tool_calls=merged.tool_calls,
            additional_kwargs=merged.additional_kwargs,
            response_metadata=merged.response_metadata,
            usage_metadata=merged.usage_metadata,
        )
        return ChatResult(generations=[ChatGeneration(message=message)])

    def _stream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        for chunk in self._chunks(self._next_turn(messages)):
            yield ChatGenerationChunk(message=chunk)

    async def _astream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> AsyncIterator[ChatGenerationChunk]:
        for chunk in self._chunks(self._next_turn(messages)):
            yield ChatGenerationChunk(message=chunk)


class Recorder:
    """Collects websocket-style messages sent by the runner."""

    def __init__(self) -> None:
        self.events: List[dict] = []

    async def send(
        self,
        message_type: str,
        value: Optional[str],
        variant_index: int,
        data: Optional[dict],
        event_id: Optional[str],
    ) -> None:
        self.events.append(
            {"type": message_type, "value": value, "variantIndex": variant_index, "data": data, "eventId": event_id}
        )

    def of_type(self, message_type: str) -> List[dict]:
        return [e for e in self.events if e["type"] == message_type]
