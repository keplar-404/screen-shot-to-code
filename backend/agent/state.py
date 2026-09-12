# pyright: reportUnknownVariableType=false, reportUnknownMemberType=false, reportUnknownArgumentType=false
"""Agent state shared between the deep agent graph and the tools."""

from dataclasses import dataclass, field
from typing import Any, List, Optional

from deepagents import DeepAgentState
from typing_extensions import NotRequired, TypedDict

from codegen.utils import extract_html_content
from prompts.prompt_types import PromptMessage


class HtmlFile(TypedDict):
    path: str
    content: str


class S2CState(DeepAgentState):
    """Deep agent state plus the single HTML file the agent is building.

    ``html_file`` is written by the ``create_file`` / ``edit_file`` tools via
    ``Command(update=...)`` and read back by ``edit_file``, ``screenshot_preview``
    and the runner when the run finishes.
    """

    html_file: NotRequired[Optional[HtmlFile]]
    # Running totals maintained by middleware.
    model_calls: NotRequired[int]
    spent_usd: NotRequired[float]


@dataclass
class AgentFileState:
    path: str = "index.html"
    content: str = ""


@dataclass(frozen=True)
class S2CContext:
    """Per-variant, read-only configuration available to tools via ``runtime.context``."""

    # Mutable working file shared by the file tools of one variant.
    file_state: AgentFileState = field(default_factory=AgentFileState)
    variant_index: int = 0
    asset_base_url: str = ""
    input_images: List[str] = field(default_factory=list)
    option_codes: List[str] = field(default_factory=list)
    should_generate_images: bool = True
    openrouter_api_key: Optional[str] = None
    replicate_api_key: Optional[str] = None
    user_id: Optional[str] = None


def ensure_str(value: Any) -> str:
    if value is None:
        return ""
    return str(value)


def extract_text_content(message: PromptMessage) -> str:
    content = message.get("content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        for part in content:
            if isinstance(part, dict) and part.get("type") == "text":
                return ensure_str(part.get("text"))
    return ""


def seed_file_state_from_messages(
    file_state: AgentFileState,
    prompt_messages: List[PromptMessage],
) -> None:
    if file_state.content:
        return

    for message in reversed(prompt_messages):
        if message.get("role") != "assistant":
            continue
        raw_text = extract_text_content(message)
        if not raw_text:
            continue
        extracted = extract_html_content(raw_text)
        file_state.content = extracted or raw_text
        if not file_state.path:
            file_state.path = "index.html"
        return

    if not prompt_messages:
        return

    system_message = prompt_messages[0]
    if system_message.get("role") != "system":
        return

    system_text = extract_text_content(system_message)
    markers = [
        "Here is the code of the app:",
    ]
    for marker in markers:
        if marker not in system_text:
            continue
        raw_text = system_text.split(marker, 1)[1].strip()
        extracted = extract_html_content(raw_text)
        file_state.content = extracted or raw_text
        if not file_state.path:
            file_state.path = "index.html"
        return
