# pyright: reportUnknownVariableType=false, reportUnknownMemberType=false, reportUnknownArgumentType=false
"""Run one code-generation variant with a LangChain Deep Agent over OpenRouter.

``DeepAgentRunner.run`` has the same contract as the original ``Agent.run``:
it takes the provider-neutral prompt messages, streams UI events through
``send_message`` and returns the final HTML.
"""

import traceback
import uuid
from datetime import datetime
from typing import Any, Awaitable, Callable, Dict, List, Optional

from deepagents import FilesystemMiddleware, create_deep_agent
import agent.profiles  # noqa: F401  (registers the OpenRouter harness profile)
from agent.middleware import (
    AnthropicImageLimitsMiddleware,
    BudgetMiddleware,
    RunRecorderMiddleware,
    StepLimitMiddleware,
    ToolStartEventsMiddleware,
)
from agent.state import AgentFileState, S2CContext, S2CState, seed_file_state_from_messages
from agent.stream_bridge import StreamBridge
from agent.tools.registry import build_tools, resolve_tool_caps
from config import AGENT_MAX_STEPS, GENERATION_MAX_COST_USD
from fs_logging.agent_runs import AgentRunRecorder
from llm import Llm, get_image_detail, get_model_spec, get_openrouter_slug
from models.factory import make_chat_model
from prompts.prompt_types import PromptMessage
from prompts.to_langchain import (
    convert_prompt_messages,
    extract_input_images,
    split_system_prompt,
)


class EmptyOutputError(Exception):
    """Raised when a run finishes without producing any HTML.

    Some models occasionally run asset tools and then stop without calling
    create_file. Treating that as success poisons evals, so it is a normal,
    retryable failure.
    """

    def __init__(self) -> None:
        super().__init__("Generation finished without producing any output.")


def new_generation_id() -> str:
    return f"gen_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"


class DeepAgentRunner:
    def __init__(
        self,
        send_message: Callable[
            [str, Optional[str], int, Optional[Dict[str, Any]], Optional[str]],
            Awaitable[None],
        ],
        variant_index: int,
        openrouter_api_key: Optional[str],
        replicate_api_key: Optional[str] = None,
        should_generate_images: bool = True,
        should_extract_assets: bool = True,
        asset_base_url: str = "",
        initial_file_state: Optional[Dict[str, str]] = None,
        option_codes: Optional[List[str]] = None,
        recorder: Optional[AgentRunRecorder] = None,
        openrouter_base_url: Optional[str] = None,
        max_steps: int = AGENT_MAX_STEPS,
        max_cost_usd: float = GENERATION_MAX_COST_USD,
        user_id: Optional[str] = None,
    ):
        self.send_message = send_message
        self.variant_index = variant_index
        self.recorder = recorder
        self.openrouter_api_key = openrouter_api_key
        self.openrouter_base_url = openrouter_base_url
        self.replicate_api_key = replicate_api_key
        self.should_generate_images = should_generate_images
        self.should_extract_assets = should_extract_assets
        self.asset_base_url = asset_base_url
        self.option_codes = option_codes or []
        self.max_steps = max_steps
        self.max_cost_usd = max_cost_usd
        self.user_id = user_id

        self.file_state = AgentFileState()
        if initial_file_state and initial_file_state.get("content"):
            self.file_state.path = initial_file_state.get("path") or "index.html"
            self.file_state.content = initial_file_state["content"]

    # ----------------------------------------------------------- assembly

    def build_middleware(self, model: Llm) -> List[Any]:
        middleware: List[Any] = [
            # Keep the built-in filesystem tooling out of the way: the model
            # writes through create_file/edit_file. The eviction thresholds are
            # disabled because this app legitimately sends very large human
            # messages (the current file) and tool results (retrieve_option).
            FilesystemMiddleware(
                tools=["read_file"],
                tool_token_limit_before_evict=None,
                human_message_token_limit_before_evict=None,
            ),
            StepLimitMiddleware(self.max_steps),
            BudgetMiddleware(self.max_cost_usd, pricing_key=get_openrouter_slug(model)),
            ToolStartEventsMiddleware(),
        ]
        if get_model_spec(model).family == "anthropic":
            middleware.append(AnthropicImageLimitsMiddleware())
        middleware.append(RunRecorderMiddleware(model, self.recorder))
        return middleware

    def build_agent(self, model: Llm, tools: List[Any], system_prompt: str) -> Any:
        chat_model = make_chat_model(model, self.openrouter_api_key, base_url=self.openrouter_base_url)
        return create_deep_agent(
            model=chat_model,
            tools=tools,
            system_prompt=system_prompt,
            state_schema=S2CState,
            context_schema=S2CContext,
            middleware=self.build_middleware(model),
            name=f"screenshot-to-code-variant-{self.variant_index}",
        )

    # ---------------------------------------------------------------- run

    async def run(self, model: Llm, prompt_messages: List[PromptMessage]) -> str:
        input_images = extract_input_images(prompt_messages)
        seed_file_state_from_messages(self.file_state, prompt_messages)

        if self.recorder is not None:
            self.recorder.record_run_start(model, prompt_messages)

        caps = resolve_tool_caps(
            should_generate_images=self.should_generate_images,
            should_extract_assets=self.should_extract_assets,
            has_input_images=bool(input_images),
            openrouter_api_key=self.openrouter_api_key,
            replicate_api_key=self.replicate_api_key,
        )
        tools = build_tools(caps)

        lc_messages = convert_prompt_messages(prompt_messages, image_detail=get_image_detail(model))
        system_prompt, conversation = split_system_prompt(lc_messages)

        agent = self.build_agent(model, tools, system_prompt)
        context = S2CContext(
            file_state=self.file_state,
            variant_index=self.variant_index,
            asset_base_url=self.asset_base_url,
            input_images=input_images,
            option_codes=list(self.option_codes),
            should_generate_images=self.should_generate_images,
            openrouter_api_key=self.openrouter_api_key,
            replicate_api_key=self.replicate_api_key,
            user_id=self.user_id,
        )
        bridge = StreamBridge(
            send_message=self.send_message,
            variant_index=self.variant_index,
            file_state=self.file_state,
            recorder=self.recorder,
        )
        initial_state: Dict[str, Any] = {
            "messages": conversation,
            "html_file": (
                {"path": self.file_state.path, "content": self.file_state.content}
                if self.file_state.content
                else None
            ),
            "model_calls": 0,
            "spent_usd": 0.0,
        }

        try:
            async for mode, chunk in agent.astream(
                initial_state,
                context=context,
                config={"recursion_limit": self.max_steps * 10 + 25},
                stream_mode=["messages", "updates", "custom", "values"],
            ):
                await bridge.handle(mode, chunk)

            result = await bridge.finalize()
            if not result:
                raise EmptyOutputError()
            if self.recorder is not None:
                await self.recorder.record_run_end("completed", final_html=result)
            return result
        # BaseException so cancellation (client disconnect) still finalizes
        # the run record instead of leaving it stuck at "running".
        except BaseException as exc:
            if self.recorder is not None:
                await self.recorder.record_run_end(
                    "failed",
                    error="".join(traceback.format_exception_only(type(exc), exc)).strip(),
                )
            raise


# Name kept for callers that imported the original class.
Agent = DeepAgentRunner
