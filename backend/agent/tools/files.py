# pyright: reportUnknownVariableType=false, reportUnknownMemberType=false, reportUnknownArgumentType=false
"""``create_file`` and ``edit_file`` — the only way the model writes code."""

import difflib
from typing import Any, Dict, List, Optional, Tuple

from langchain.tools import ToolRuntime, tool
from langgraph.types import Command
from pydantic import BaseModel, Field

from agent.state import AgentFileState, S2CContext, S2CState, ensure_str
from agent.tools.emit import emit_result
from agent.tools.summaries import summarize_text
from agent.tools.types import ToolExecutionResult
from codegen.utils import extract_html_content


class CreateFileArgs(BaseModel):
    path: Optional[str] = Field(
        default=None,
        description="Path for the main HTML file. Use index.html if unsure.",
    )
    content: str = Field(description="Full HTML for the single-file app.")


class EditItem(BaseModel):
    old_text: str
    new_text: str
    count: Optional[int] = None


class EditFileArgs(BaseModel):
    path: Optional[str] = Field(default=None, description="Path for the main HTML file.")
    old_text: Optional[str] = Field(
        default=None, description="Exact text to replace. Must match the file contents."
    )
    new_text: Optional[str] = Field(default=None, description="Replacement text.")
    count: Optional[int] = Field(
        default=None, description="How many occurrences to replace. Use -1 for all."
    )
    edits: Optional[List[EditItem]] = Field(
        default=None, description="Batch of independent replacements applied in order."
    )


def file_state_from_runtime(runtime: ToolRuntime[S2CContext, S2CState]) -> AgentFileState:
    """The per-variant file the agent is editing.

    The mutable ``AgentFileState`` lives on the context so that tools called
    in the same turn observe each other's writes (graph state is only
    committed between steps). The graph's ``html_file`` key mirrors it.
    """
    return runtime.context.file_state


def create_file_impl(file_state: AgentFileState, args: Dict[str, Any]) -> ToolExecutionResult:
    path = ensure_str(args.get("path") or file_state.path or "index.html")
    content = ensure_str(args.get("content"))
    if not content:
        return ToolExecutionResult(
            ok=False,
            result={"error": "create_file requires non-empty content"},
            summary={"error": "Missing content"},
        )

    extracted = extract_html_content(content)
    file_state.path = path
    file_state.content = extracted or content

    summary = {
        "path": file_state.path,
        "contentLength": len(file_state.content),
        "preview": summarize_text(file_state.content, 320),
    }
    result = {
        "content": f"Successfully created file at {file_state.path}.",
        "details": {
            "path": file_state.path,
            "contentLength": len(file_state.content),
        },
    }
    return ToolExecutionResult(
        ok=True, result=result, summary=summary, updated_content=file_state.content
    )


def generate_diff(old_content: str, new_content: str, path: str) -> Dict[str, Any]:
    """Generate a unified diff between old and new content."""
    old_lines = old_content.splitlines(keepends=True)
    new_lines = new_content.splitlines(keepends=True)
    diff_lines = list(difflib.unified_diff(old_lines, new_lines, fromfile=path, tofile=path))
    diff_str = "".join(diff_lines)

    first_changed_line: Optional[int] = None
    for line in diff_lines:
        if not line.startswith("@@"):
            continue
        try:
            plus_part = line.split("+")[1].split("@@")[0].strip()
            first_changed_line = int(plus_part.split(",")[0])
        except (IndexError, ValueError):
            pass
        break

    return {"diff": diff_str, "firstChangedLine": first_changed_line}


def apply_single_edit(
    content: str, old_text: str, new_text: str, count: Optional[int]
) -> Tuple[str, int]:
    if old_text not in content:
        return content, 0

    if count is None:
        replace_count = 1
    elif count < 0:
        replace_count = content.count(old_text)
    else:
        replace_count = count

    updated = content.replace(old_text, new_text, replace_count)
    return updated, min(replace_count, content.count(old_text))


def edit_file_impl(file_state: AgentFileState, args: Dict[str, Any]) -> ToolExecutionResult:
    if not file_state.content:
        return ToolExecutionResult(
            ok=False,
            result={"error": "No file exists yet. Call create_file first."},
            summary={"error": "No file to edit"},
        )

    edits = args.get("edits")
    if not edits:
        old_text = ensure_str(args.get("old_text"))
        new_text = ensure_str(args.get("new_text"))
        count = args.get("count")
        edits = [{"old_text": old_text, "new_text": new_text, "count": count}]

    if not isinstance(edits, list):
        return ToolExecutionResult(
            ok=False,
            result={"error": "edits must be a list"},
            summary={"error": "Invalid edits payload"},
        )

    content = file_state.content
    original_content = content
    summary_edits: List[Dict[str, Any]] = []
    for edit in edits:
        if not isinstance(edit, dict):
            edit = edit.model_dump() if hasattr(edit, "model_dump") else {}
        old_text = ensure_str(edit.get("old_text"))
        new_text = ensure_str(edit.get("new_text"))
        count = edit.get("count")
        if not old_text:
            return ToolExecutionResult(
                ok=False,
                result={"error": "edit_file requires old_text"},
                summary={"error": "Missing old_text"},
            )

        content, replaced = apply_single_edit(content, old_text, new_text, count)
        if replaced == 0:
            return ToolExecutionResult(
                ok=False,
                result={"error": "old_text not found", "old_text": old_text},
                summary={
                    "error": "old_text not found",
                    "old_text": summarize_text(old_text, 160),
                },
            )

        summary_edits.append(
            {
                "old_text": summarize_text(old_text, 140),
                "new_text": summarize_text(new_text, 140),
                "replaced": replaced,
            }
        )

    file_state.content = content
    path = file_state.path or "index.html"
    diff_info = generate_diff(original_content, content, path)
    summary = {
        "path": path,
        "edits": summary_edits,
        "contentLength": len(file_state.content),
        "diff": diff_info["diff"],
        "firstChangedLine": diff_info["firstChangedLine"],
    }
    result = {
        "content": f"Successfully edited file at {path}.",
        "details": {
            "diff": diff_info["diff"],
            "firstChangedLine": diff_info["firstChangedLine"],
        },
    }
    return ToolExecutionResult(
        ok=True, result=result, summary=summary, updated_content=file_state.content
    )


@tool(
    "create_file",
    args_schema=CreateFileArgs,
    description=(
        "Create the main HTML file for the app. Use exactly once to write the "
        "full HTML. Returns a success message and file metadata."
    ),
)
def create_file(
    content: str,
    runtime: ToolRuntime[S2CContext, S2CState],
    path: Optional[str] = None,
) -> Command:
    file_state = file_state_from_runtime(runtime)
    result = create_file_impl(file_state, {"path": path, "content": content})
    updated = (
        {"path": file_state.path, "content": file_state.content} if result.ok else None
    )
    return emit_result(runtime, "create_file", result, updated_file=updated)


@tool(
    "edit_file",
    args_schema=EditFileArgs,
    description=(
        "Edit the main HTML file using exact string replacements. Do not "
        "regenerate the entire file. Returns a success message plus edit "
        "details, including a unified diff and first changed line."
    ),
)
def edit_file(
    runtime: ToolRuntime[S2CContext, S2CState],
    path: Optional[str] = None,
    old_text: Optional[str] = None,
    new_text: Optional[str] = None,
    count: Optional[int] = None,
    edits: Optional[List[EditItem]] = None,
) -> Command:
    file_state = file_state_from_runtime(runtime)
    args: Dict[str, Any] = {
        "path": path,
        "old_text": old_text,
        "new_text": new_text,
        "count": count,
        "edits": [e.model_dump() if isinstance(e, EditItem) else e for e in edits] if edits else None,
    }
    result = edit_file_impl(file_state, args)
    updated = (
        {"path": file_state.path, "content": file_state.content} if result.ok else None
    )
    return emit_result(runtime, "edit_file", result, updated_file=updated)
