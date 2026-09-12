# pyright: reportUnknownVariableType=false, reportUnknownMemberType=false, reportUnknownArgumentType=false
"""``screenshot_preview`` — render the current HTML in headless Chromium so the
model can visually verify its work."""

from langchain.tools import ToolRuntime, tool
from langgraph.types import Command

from agent.state import S2CContext, S2CState
from agent.tools.emit import emit_result
from agent.tools.screenshot_preview import run_screenshot_preview


# No args_schema on purpose: an explicit empty schema prevents LangChain from
# injecting ``runtime`` into an argument-less tool; the inferred schema is
# ``{"type": "object", "properties": {}}`` either way.
@tool(
    "screenshot_preview",
    description=(
        "Render the current HTML file in a headless browser and return "
        "full-page desktop and mobile screenshots so you can visually "
        "verify your work. Use after creating or substantially editing "
        "the file to check layout, spacing, and fidelity to the "
        "requested design. Screenshots are returned as attached images."
    ),
)
async def screenshot_preview(runtime: ToolRuntime[S2CContext, S2CState]) -> Command:
    result = await run_screenshot_preview({}, file_state=runtime.context.file_state)
    return emit_result(runtime, "screenshot_preview", result)
