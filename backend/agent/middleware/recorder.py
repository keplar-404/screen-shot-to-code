# pyright: reportUnknownVariableType=false, reportUnknownMemberType=false, reportUnknownArgumentType=false
"""Feed the on-disk run recorder / prompt reports from inside the agent loop.

The stream bridge records deltas, tool calls and the final result; this
middleware records what only the model-call boundary can see: the exact
request (messages, tools, settings) and the final response with token usage.
"""

from typing import Any, Awaitable, Callable, List, Optional

from langchain.agents.middleware.types import AgentMiddleware, ModelRequest, ModelResponse
from langchain_core.messages import AIMessage
from langchain_core.utils.function_calling import convert_to_openai_tool

from agent.tools.types import ToolCall
from costs.token_usage import token_usage_from_message
from fs_logging.agent_runs import AgentRunRecorder
from fs_logging.prompt_reports import PromptReportLogger
from llm import Llm, get_openrouter_slug


def _serialize_tools(tools: Any) -> List[Any]:
    serialized: List[Any] = []
    for tool in tools or []:
        try:
            serialized.append(convert_to_openai_tool(tool))
        except Exception:
            serialized.append(getattr(tool, "name", str(tool)))
    return serialized


def _extract_ai_message(response: Any) -> Optional[AIMessage]:
    result = getattr(response, "result", None)
    if isinstance(result, list):
        for message in reversed(result):
            if isinstance(message, AIMessage):
                return message
    if isinstance(response, AIMessage):
        return response
    return None


class RunRecorderMiddleware(AgentMiddleware):
    def __init__(self, model: Llm, recorder: Optional[AgentRunRecorder]) -> None:
        self.model = model
        self.recorder = recorder
        self.slug = get_openrouter_slug(model)
        self.prompt_reports = PromptReportLogger(
            provider="openrouter", model=model, api_model_name=self.slug
        )

    def _before(self, request: ModelRequest) -> None:
        payload = {
            "model": self.slug,
            "system_message": request.system_message,
            "messages": list(request.messages),
            "tools": _serialize_tools(request.tools),
            "tool_choice": request.tool_choice,
            "model_settings": request.model_settings,
        }
        self.prompt_reports.record_request(payload)
        if self.recorder is not None:
            self.recorder.record_llm_request("openrouter", self.slug, payload)

    def _after(self, response: Any) -> None:
        message = _extract_ai_message(response)
        if message is None:
            return
        usage = token_usage_from_message(message)
        if usage is not None:
            self.prompt_reports.record_usage(usage)
        if self.recorder is not None:
            tool_calls = [
                ToolCall(id=str(call.get("id") or ""), name=str(call.get("name") or ""), arguments=dict(call.get("args") or {}))
                for call in message.tool_calls
            ]
            self.recorder.record_llm_response(message.text or "", tool_calls, usage)

    def wrap_model_call(self, request: ModelRequest, handler: Callable[[ModelRequest], ModelResponse]) -> Any:
        self._before(request)
        response = handler(request)
        self._after(response)
        return response

    async def awrap_model_call(
        self, request: ModelRequest, handler: Callable[[ModelRequest], Awaitable[ModelResponse]]
    ) -> Any:
        self._before(request)
        response = await handler(request)
        self._after(response)
        return response
