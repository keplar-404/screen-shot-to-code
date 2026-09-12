# pyright: reportUnknownVariableType=false, reportUnknownMemberType=false, reportUnknownArgumentType=false
"""``retrieve_option`` — fetch the full HTML of another variant."""

from typing import Any, Dict, List, Optional

from langchain.tools import ToolRuntime, tool
from langgraph.types import Command
from pydantic import BaseModel, Field

from agent.state import S2CContext, S2CState, ensure_str
from agent.tools.emit import emit_result
from agent.tools.summaries import summarize_text
from agent.tools.types import ToolExecutionResult


class RetrieveOptionArgs(BaseModel):
    option_number: int = Field(
        description="1-based option number to retrieve (Option 1, Option 2, etc.)."
    )


def retrieve_option_impl(option_codes: List[str], args: Dict[str, Any]) -> ToolExecutionResult:
    raw_option_number = args.get("option_number")
    raw_index = args.get("index")

    def coerce_int(value: Any) -> Optional[int]:
        if value is None:
            return None
        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    option_number = coerce_int(raw_option_number)
    index = coerce_int(raw_index)

    if option_number is None and index is None:
        return ToolExecutionResult(
            ok=False,
            result={"error": "retrieve_option requires option_number"},
            summary={"error": "Missing option_number"},
        )

    resolved_index = index if option_number is None else option_number - 1
    if resolved_index is None:
        return ToolExecutionResult(
            ok=False,
            result={"error": "Invalid option_number"},
            summary={"error": "Invalid option_number"},
        )

    if resolved_index < 0 or resolved_index >= len(option_codes):
        return ToolExecutionResult(
            ok=False,
            result={
                "error": "Option index out of range",
                "option_number": resolved_index + 1,
                "available": len(option_codes),
            },
            summary={"error": "Option index out of range", "available": len(option_codes)},
        )

    code = ensure_str(option_codes[resolved_index])
    if not code.strip():
        return ToolExecutionResult(
            ok=False,
            result={
                "error": "Option code is empty or unavailable",
                "option_number": resolved_index + 1,
            },
            summary={"error": "Option code unavailable"},
        )

    summary = {
        "option_number": resolved_index + 1,
        "contentLength": len(code),
        "preview": summarize_text(code, 200),
    }
    return ToolExecutionResult(
        ok=True, result={"option_number": resolved_index + 1, "code": code}, summary=summary
    )


@tool(
    "retrieve_option",
    args_schema=RetrieveOptionArgs,
    description=(
        "Retrieve the full HTML for a specific option (variant) so you can reference it."
    ),
)
def retrieve_option(option_number: int, runtime: ToolRuntime[S2CContext, S2CState]) -> Command:
    result = retrieve_option_impl(runtime.context.option_codes, {"option_number": option_number})
    return emit_result(runtime, "retrieve_option", result)
